import torch
import os
from torch import nn, optim
from torch_geometric.data import Data#, DataLoader
from torch_geometric.loader import DataLoader
import numpy as np

# Define a dummy dataset
class DummyGraphDataset:
    def __init__(self, num_graphs=100):
        self.data = [Data(x=torch.randn(10, 4), edge_index=torch.randint(0, 10, (2, 20)), edge_attr=torch.randn(20, 3)) for _ in range(num_graphs)]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

# Define dummy Generator and Discriminator for demonstration
class DummyGenerator(nn.Module):
    def __init__(self, input_dim):
        super(DummyGenerator, self).__init__()
        self.fc = nn.Linear(input_dim, 40)  # Arbitrary dimension

    def forward(self, z):
        return {'x': torch.randn(len(z), 10, 4), 'edge_index': torch.randint(0, 10, (2, 20), dtype=torch.long), 'edge_attr': torch.randn(20, 3)}

class DummyDiscriminator(nn.Module):
    def __init__(self):
        super(DummyDiscriminator, self).__init__()
        self.fc = nn.Linear(4, 1)

    def forward(self, data):
        return torch.sigmoid(self.fc(data.x)).mean()  # Dummy forward pass

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize dummy models
generator = DummyGenerator(input_dim=100).to(device)
discriminator = DummyDiscriminator().to(device)

# Setup the trainer
from trainer import GANTrainer  # Assuming GANTrainer is in a file named GANTrainer.py

train_folder = './train_results'
if not os.path.exists(train_folder):
    os.makedirs(train_folder)

trainer = GANTrainer(
    d=discriminator,
    g=generator,
    rand_dim=100,  # Dimension of the noise vector
    train_folder=train_folder,
    tot_epoch_num=5,  # For demo purposes
    batch_size=10,
    device=device
)

# Setup data loader
dataset = DummyGraphDataset(num_graphs=50)
data_loader = DataLoader(dataset, batch_size=10, shuffle=True)

# Run the training
trainer.train(data_loader)
