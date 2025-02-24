import torch
from rdkit.Chem import Draw
import os

from eval.mol_utils import gen_mol
from eval.evaluation import get_mol_metric


###### Dummy Examples ######

#   - annots: (batch_size, n_max, n_feat)
#   - adjs: (batch_size, n_edge_types, n_max, n_max)
batch_size = 32      # e.g., 32 sampled molecules
n_max = 9           # maximum number of nodes per graph
n_feat = 5          # number of node annotation features
n_edge_types = 4    # number of edge types

# annots = torch.randint(0, 2, (batch_size, n_max, n_feat), dtype=torch.float)
# adjs = torch.randint(0, 2, (batch_size, n_edge_types, n_max, n_max), dtype=torch.float)

# Define one-hot vectors for atoms based on our mapping:
# Carbon: [1, 0, 0, 0, 0]
# Oxygen: [0, 0, 1, 0, 0]
# Padding: [0, 0, 0, 0, 1]
carbon  = torch.tensor([1, 0, 0, 0, 0], dtype=torch.float)
oxygen  = torch.tensor([0, 0, 1, 0, 0], dtype=torch.float)
padding = torch.tensor([0, 0, 0, 0, 1], dtype=torch.float)

# Create annotation tensor: shape (batch_size, n_max, n_feat)
annots = torch.zeros(batch_size, n_max, n_feat)
for i in range(batch_size):
    # For ethanol "CCO", set first three nodes to C, C, O.
    annots[i, 0] = carbon   # Atom 0: Carbon
    annots[i, 1] = carbon   # Atom 1: Carbon
    annots[i, 2] = oxygen   # Atom 2: Oxygen
    # Fill the rest with padding
    annots[i, 3:] = padding

# Create adjacency tensor: shape (batch_size, n_edge_types, n_max, n_max)
# Start by filling every edge with "no bond": one-hot vector [1, 0, 0, 0]
adjs = torch.zeros(batch_size, n_edge_types, n_max, n_max)
adjs[:, 0, :, :] = 1.0  # all edges set to "no bond" by default

# For ethanol, add bonds:
# Bond between atom 0 and atom 1, and between atom 1 and atom 2.
# To encode a single bond, we remove the "no bond" (channel 0) and set channel 1 to 1.
for i in range(batch_size):
    # Bond between atom 0 and atom 1 (both directions for an undirected graph)
    adjs[i, 0, 0, 1] = 0.0
    adjs[i, 1, 0, 1] = 1.0
    adjs[i, 0, 1, 0] = 0.0
    adjs[i, 1, 1, 0] = 1.0
    
    # Bond between atom 1 and atom 2 (both directions)
    adjs[i, 0, 1, 2] = 0.0
    adjs[i, 1, 1, 2] = 1.0
    adjs[i, 0, 2, 1] = 0.0
    adjs[i, 1, 2, 1] = 1.0

###### Run the Pipeline ########
dataset = "QM9"  
gen_mols, num_no_correct = gen_mol(annots, adjs, dataset)
metrics = get_mol_metric(gen_mols, dataset, num_no_correct)

print("Generated Molecules (SMILES):")
output_dir = "output_images"
os.makedirs(output_dir, exist_ok=True)

for idx, mol in enumerate(gen_mols):
    img = Draw.MolToImage(mol, size=(300, 300))
    file_path = os.path.join("output_images", f"molecule_{idx}.png")
    img.save(file_path)

print("\nEvaluation Metrics:")
for key, value in metrics.items():
    print(f"{key}: {value}")