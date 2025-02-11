import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader
from models.graph_var import GraphVAR
from models.graph_encoder import GraphEncoder
from torch_geometric.data import DataLoader
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader


def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for data in dataloader:
        data = data.to(device)
        optimizer.zero_grad()

        # Pass entire data object
        logits = model(data)  

        # Compute loss using ground truth labels
        loss = criterion(logits, data.y)  
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


# Example Usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the QM9 dataset
dataset = QM9(root="data/QM9")

graph_encoder = GraphEncoder(in_dim=dataset.num_features, embed_dim=512).to(device)
model = GraphVAR(graph_encoder, num_nodes=100, embed_dim=512).to(device)

# Define train, validation, and test splits (optional)
train_dataset = dataset[:10000]  # First 10,000 molecules for training
val_dataset = dataset[10000:11000]  # Next 1,000 for validation
test_dataset = dataset[11000:]  # Remaining for testing

# Create PyTorch Geometric DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Train for multiple epochs
for epoch in range(10):
    loss = train(model, train_loader, optimizer, criterion, device)
    print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
