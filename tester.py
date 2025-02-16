''' This is a tutorial to run a forward pass on the VQVAE model for graphs. '''

import yaml, torch
from model.vqvae import VQVAE
from easydict import EasyDict as edict
from data.dataset import load_qm9_data
from utils.test_permutation import PermutationEvaluation
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch

# Load in config
config_path = f'config.yaml'
config = edict(yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader))

# Load data and select next batch
data_root = config.data.path
batch_size = config.train.batch_size
num_workers = 0
train_val_test_split = config.data.train_val_test_split
dataset_size = config.data.dataset_size

train_loader, val_loader, test_loader = load_qm9_data(
        transforms=[],
        root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        train_val_test_split=train_val_test_split,
        dataset_size=dataset_size
    )

batch = next(iter(train_loader))

# Load model
model = VQVAE(config=config)

# Forward pass
commitment_loss, q_latent_loss, nodes_recon, edges_recon, node_masks = model(batch)
print(commitment_loss, q_latent_loss, nodes_recon.shape, edges_recon.shape, node_masks.shape)

# Test equivariance of model
dataset = QM9(config.data.path, transform=None)
permutation_tester = PermutationEvaluation(model=model, dataset=dataset)
permutation_tester.evaluate()
