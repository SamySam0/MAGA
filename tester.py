''' This is a tutorial to run a forward pass on the VQVAE model for graphs. '''

import yaml, torch
from model.vqvae import VQVAE
from easydict import EasyDict as edict
from utils.dataset_loader import get_dataloaders
from utils.test_permutation import PermutationEvaluation
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch

# Load in config
config_path = f'config.yaml'
config = edict(yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader))

# Load data and select next batch
train_dataloader, val_dataloader, test_dataloader = get_dataloaders(config)
batch = next(iter(train_dataloader))

# Load model
model = VQVAE(config=config)

# Forward pass
commitment_loss, q_latent_loss, nodes_recon, edges_recon, node_masks = model(batch)
print(commitment_loss, q_latent_loss, nodes_recon.shape, edges_recon.shape, node_masks.shape)

# Test equivariance of model
dataset = QM9(config.data.path, transform=None)
permutation_tester = PermutationEvaluation(model=model, dataset=dataset)
permutation_tester.evaluate()
