import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml
from model.vqvae import VQVAE
from easydict import EasyDict as edict
from utils.losses import get_losses
from utils.func import discretize, get_edge_target, get_edge_masks

class Trainer:
    def __init__(self, dataloaders, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = VQVAE(config=config).to(self.device)
        self.train_loader, self.valid_loader = dataloaders
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.train.lr, betas=(config.train.beta1, config.train.beta2))
        self.gamma = config.train.gamma
        self.n_epochs = config.train.epochs
        self._log_model_parameters()

    def _log_model_parameters(self):
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Number of trainable parameters in the model: {num_params}")

    def step(self, batch, train: bool):
        if train:
            self.optimizer.zero_grad()

        commitment_loss, q_latent_loss, nodes_recon, edges_recon, node_masks = self.model(batch)
        masks = node_masks.detach(), get_edge_masks(node_masks) 

        recon_loss, rec_losses = get_losses(
            batch=batch, 
            nodes_rec=nodes_recon, 
            edges_rec=edges_recon, 
            node_masks=node_masks.unsqueeze(-1), 
            annotated_nodes=True, 
            annotated_edges=True, 
            max_node_num=9, 
            n_node_feat=11, 
            edge_masks=masks[1],
        )

        loss = recon_loss + (commitment_loss + q_latent_loss) * self.gamma
        if train:
            loss.backward()
            self.optimizer.step()
        
        return recon_loss.item()
    
    def train_step(self):
        self.model.train()
        batch_recon_loss = []
        for batch in self.train_loader:
            loss = self.step(batch.to(self.device), train=True)
            batch_recon_loss.append(loss)
        return np.mean(batch_recon_loss)

    def valid_step(self):
        self.model.eval()
        batch_recon_loss = []
        for batch in self.valid_loader:
            with torch.no_grad():
                loss = self.step(batch.to(self.device), train=False)
            batch_recon_loss.append(loss)
        return np.mean(batch_recon_loss)
    
    def train(self):
        for epoch in range(1, self.n_epochs+1):
            train_loss = self.train_step()
            valid_loss = self.valid_step()
            if epoch % 10 == 0:
                print(f"Epoch: {epoch}, Train Loss: {train_loss}, Valid Loss: {valid_loss}")








def run_experiment(model, train_loader, val_loader, test_loader, gamma, n_epochs=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # LR scheduler: decays LR when validation error plateaus.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.9, patience=5, min_lr=1e-5
    )
    
    print("\nStart training:")
    best_val_error = None
    test_error = None
    perf_per_epoch = []  # Records (test_error, val_error, epoch, model_name) for plotting later.
        
    for epoch in range(1, n_epochs + 1):
        # Current learning rate.
        lr = optimizer.param_groups[0]['lr']
        
        # Train for one epoch.
        loss = train(model, train_loader, optimizer, gamma, device)
        
        # Evaluate on validation set.
        val_error = evaluate(model, val_loader, gamma, device)
        
        # If validation improves, evaluate on test set.
        if best_val_error is None or val_error <= best_val_error:
            test_error = evaluate(model, test_loader, gamma, device)
            best_val_error = val_error
        
        # Print status every 10 epochs.
        if epoch % 10 == 0:
            print(f"Epoch: {epoch:03d}, LR: {lr:.6f}, Train Loss: {loss:.7f}, "
                  f"Val: {val_error:.7f}, Test: {test_error:.7f}")
        
        # Step the scheduler using the validation error.
        scheduler.step(val_error)
        perf_per_epoch.append((test_error, val_error, epoch))
    
    print(f"Best validation: {best_val_error:.7f}, corresponding test: {test_error:.7f}.")

    return best_val_error, test_error, train_time, perf_per_epoch
