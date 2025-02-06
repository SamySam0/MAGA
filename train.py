import torch, yaml, random, time, os
from datetime import datetime
import torch.nn.functional as F
import torch.optim as optim

from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

from model.vqvae import VQVAE


# ===== Initialisation =====
torch.manual_seed(42)
RUN_NAME = f"{datetime.now().strftime('%m_%d_%H_%M')}"
print(f'Run name: {RUN_NAME}')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Create log dir if does not exist
os.makedirs(config['train']['checkpoint_dir'], exist_ok=True)

# Save config in log file
with open(f"{config['train']['checkpoint_dir']}/{RUN_NAME}_logs.txt", 'w') as f:
    f.write('===== Config =====\n')
    yaml.dump(config, f, default_flow_style=False)
    f.write('\n===== Training logs =====')

# ===== Prepare dataset =====
dataset = TUDataset(root=config['data']['path'], name=config['data']['name'])

# Fix the size of all adjacency matrices to the biggest one of the dataset
max_nodes = max(max(data.edge_index.max() for data in dataset), 
                max(data.edge_index.shape[0] for data in dataset)) + 1
if max_nodes % 2 == 1: max_nodes += 1

# Get adjacency matrix from list of edges
adj_matrices = []
for data in dataset:
    adj_matrix = torch.zeros(max_nodes, max_nodes)
    edges = data.edge_index.T
    adj_matrix[edges[:, 0], edges[:, 1]] = 1
    adj_matrices.append(adj_matrix)
adj_matrices = torch.stack(adj_matrices).unsqueeze(-1).permute(0, 3, 1, 2)

# Get splits for training and validation
dataset_size = len(adj_matrices)
train_size = int(config['data']['train_split'] * dataset_size)
indices = torch.randperm(dataset_size)
train_indices, valid_indices = indices[:train_size], indices[train_size:]

# Load adjacency dataset into data-loader
training_loader = DataLoader(
    adj_matrices[train_indices], 
    batch_size=config['train']['batch_size'], 
    shuffle=True
)

valid_loader = DataLoader(
    adj_matrices[valid_indices],
    batch_size=config['train']['batch_size'],
    shuffle=False
)


# ===== Model setup =====
model = VQVAE(
    channel_in=config['model']['channel_in'],
    num_hiddens=config['model']['num_hiddens'],
    num_res_layers=config['model']['num_res_layers'],
    num_res_hiddens=config['model']['num_res_hiddens'],
    codebook_size=config['model']['codebook_size'],
    embedding_dim=config['model']['embedding_dim'],
    commitment_cost=config['model']['commitment_cost'],
).to(device)


# ===== Training and Validation =====
optimizer = optim.Adam(model.parameters(), lr=config['train']['lr'])

train_res_recon_error, train_res_codebook_loss = [], []
valid_res_recon_error, valid_res_codebook_loss = [], []
start_time = time.time()

for epoch in range(config['train']['epochs']):

    # Train
    model.train()
    epoch_res_recon_error, epoch_res_codebook_loss = 0, 0
    for batch in training_loader:
        data = batch.to(device)
        optimizer.zero_grad()

        # Get model's prediction and apply sigmoid for BCE
        vq_loss, recon = model(data)
        recon = F.sigmoid(recon)

        # Cut down lower-left triangular part of the ajdacency matrix
        # This accounts for symmetry of undirected graphs when backpropagating
        pred, y = recon.squeeze(1), data.squeeze(1)
        mask_upper_tri = torch.triu(torch.ones(max_nodes, max_nodes, dtype=torch.bool), diagonal=1)
        pred = pred[:, mask_upper_tri]
        y    = y[:, mask_upper_tri]

        # Calculate loss and backpropagate
        recon_error = F.binary_cross_entropy(pred, y)
        loss = recon_error + vq_loss
        loss.backward()

        optimizer.step()

        epoch_res_recon_error += recon_error.item()
        epoch_res_codebook_loss += vq_loss.item()
    
    train_res_recon_error.append(epoch_res_recon_error)
    train_res_codebook_loss.append(epoch_res_codebook_loss)

    print(f"Epoch {epoch+1} : Training Recon error: {round(epoch_res_recon_error/len(training_loader), 4)} | Training VQ loss: {round(epoch_res_codebook_loss/len(training_loader), 4)}", end='')
    with open(f"{config['train']['checkpoint_dir']}/{RUN_NAME}_logs.txt", 'a') as f:
        f.write(f"\nEpoch {epoch+1} : Training Recon error: {round(epoch_res_recon_error/len(training_loader), 4)} | Training VQ loss: {round(epoch_res_codebook_loss/len(training_loader), 4)}")

    # Validation
    model.eval()
    epoch_res_recon_error, epoch_res_codebook_loss = 0, 0
    for batch in valid_loader:
        data = batch.to(device)

        with torch.no_grad():
            vq_loss, recon = model(data)
            recon = F.sigmoid(recon)
            
            # Cut down lower-left triangular part
            pred, y = recon.squeeze(1), data.squeeze(1)
            mask_upper_tri = torch.triu(torch.ones(max_nodes, max_nodes, dtype=torch.bool), diagonal=1)
            pred = pred[:, mask_upper_tri]
            y    = y[:, mask_upper_tri]
            
            # Calculate validation losses
            recon_error = F.binary_cross_entropy(pred, y)
            
            epoch_res_recon_error += recon_error.item()
            epoch_res_codebook_loss += vq_loss.item()

    valid_res_recon_error.append(epoch_res_recon_error)
    valid_res_codebook_loss.append(epoch_res_codebook_loss)
    
    print(f" | Valid Recon error: {round(epoch_res_recon_error/len(valid_loader), 4)} | Valid VQ loss: {round(epoch_res_codebook_loss/len(valid_loader), 4)}")
    with open(f"{config['train']['checkpoint_dir']}/{RUN_NAME}_logs.txt", 'a') as f:
        f.write(f" | Valid Recon error: {round(epoch_res_recon_error/len(valid_loader), 4)} | Valid VQ loss: {round(epoch_res_codebook_loss/len(valid_loader), 4)}")

    # Save model checkpoint
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_recon_loss': train_res_recon_error,
        'train_vq_loss': train_res_codebook_loss,
        'valid_recon_loss': valid_res_recon_error,
        'valid_vq_loss': valid_res_codebook_loss,
    }, f"{config['train']['checkpoint_dir']}/{RUN_NAME}_model_checkpoint.pt")

# Time logs
with open(f"{config['train']['checkpoint_dir']}/{RUN_NAME}_logs.txt", 'a') as f:
    f.write('\n\n===== Time logs =====')
    f.write(f"\nDevice: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    f.write(f"\nTraining time: {round(time.time() - start_time, 1)} seconds")
    print(f"Training time: {round(time.time() - start_time, 1)} seconds")


# TODO: Early stopping/scheduler 
# (must change/check which checkpoint is saved!)
# must save scheduler state when saving checkpoint model
