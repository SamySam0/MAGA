import os
from rdkit import Chem
from rdkit.Chem import Draw
from eval.mol_eval import construct_mol, correct_mol, valid_mol_can_with_seg


def save_molecules(annots_recon, adjs_recon, dataset_name, viz_dir, epoch):
    viz_dir = viz_dir + f'/{dataset_name}'
    if dataset_name == 'qm9':
        atomic_num_list = [6, 7, 8, 9, 0]                       # C, N, O, F, None
    elif dataset_name == 'zinc':
        atomic_num_list = [6, 7, 8, 9, 15, 16, 17, 35, 53, 0]   # C, N, O, F, P, S, Cl, Br, I, None
    
    # Convert to numpy
    node_one_hot = annots_recon.detach().cpu().numpy()
    adj_one_hot = adjs_recon.detach().cpu().numpy()
    
    # Generate molecules
    molecules = []
    unique_smiles = set()  # To track unique molecules

    for x, a in zip(node_one_hot, adj_one_hot):
        # Stop if we already have 25 unique molecules
        if len(unique_smiles) >= 25:
            break
            
        mol = construct_mol(x, a, atomic_num_list)
        if mol is not None:
            c_mol, _ = correct_mol(mol)
            vc_mol = valid_mol_can_with_seg(c_mol, largest_connected_comp=False)
            if vc_mol is not None:
                # Check if this molecule is unique
                smiles = Chem.MolToSmiles(vc_mol)
                if smiles not in unique_smiles:
                    unique_smiles.add(smiles)
                    molecules.append(vc_mol)
    
    # Ensure the directory exists
    os.makedirs(viz_dir, exist_ok=True)
    
    # Save a grid of molecules (if any were generated)
    if molecules:
        # No need to limit as we already have at most 25 unique molecules
        img = Draw.MolsToGridImage(molecules, molsPerRow=5, subImgSize=(500, 500))
        img.save(f"{viz_dir}/molecule_grid-epoch_{epoch}.png")
        
        # Save SMILES strings to a file
        with open(f"{viz_dir}/smiles-epoch_{epoch}.txt", "w") as f:
            for i, mol in enumerate(molecules):
                f.write(f"{i}: {Chem.MolToSmiles(mol)}\n")
