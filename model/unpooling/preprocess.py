fop = open('model/unpooling/qm9.txt')
mols = []
mols += [j.strip() for j in fop.readlines()]
fop.close()

len(mols), mols[:2]

from rdkit import Chem
import pandas as pd
from util_richer import dataFromSmile
from rdkit import RDLogger    
RDLogger.DisableLog('rdApp.*') # avoid warning messages.
all_graphs = []
for smile_str in mols:
    used = dataFromSmile(smile_str, without_aroma=True, node_dim=4)
    all_graphs.append(used)

from util_molecular import MolFromTorchGraphData_enriched
all_smiles = []
for j in all_graphs:
    mol_comp = MolFromTorchGraphData_enriched(j, without_aroma=True, node_dim=4)
    all_smiles.append(Chem.MolToSmiles(mol_comp))

import torch
torch.save(all_graphs, 'qm9_data_noar.pt')
torch.save(all_smiles, 'qm9_smiles_noar.pt')