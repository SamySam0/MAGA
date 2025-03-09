# easy import
import numpy as np
import pandas as pd
import json
import networkx as nx
import re
from datetime import datetime
from sklearn.metrics.pairwise import pairwise_kernels
import torch
import torch.nn.functional as F

# risky imports
from rdkit import Chem, RDLogger
from eden.graph import vectorize
from eval.metrics.metrics import get_all_metrics
RDLogger.DisableLog('rdApp.*')

##### Constants #####

ATOM_VALENCY = {6: 4, 7: 3, 8: 2, 9: 1, 15: 3, 16: 2, 17: 1, 35: 1, 53: 1}
bond_decoder = {1: Chem.rdchem.BondType.SINGLE, 2: Chem.rdchem.BondType.DOUBLE, 3: Chem.rdchem.BondType.TRIPLE}
AN_TO_SYMBOL = {6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I'}

##### rdkit utils functions #####

def load_smiles(dataset='qm9', subset=None):
    if dataset == 'qm9':
        col = 'SMILES1'
    elif dataset == 'zinc':
        col = 'smiles'
    else:
        raise ValueError('wrong dataset name in load_smiles')
    
    df = pd.read_csv(f'data/{dataset.lower()}/{dataset.lower()}.csv')

    with open(f'data/{dataset.lower()}/valid_idx_{dataset.lower()}.json') as f:
        test_idx = json.load(f)
    
    if dataset == 'qm9':
        test_idx = test_idx['valid_idxs']
        test_idx = [int(i) for i in test_idx]

    train_idx = [i for i in range(len(df)) if i not in test_idx]

    if subset is not None:
        # DEBUG mode, select a subset of the dataset
        assert isinstance(subset, int)
        df[col] = df[col].loc[:subset]
        train_idx = list(range(subset))
        test_idx = train_idx

    return list(df[col].loc[train_idx]), list(df[col].loc[test_idx])

def canonicalize_smiles(smiles):
    return [Chem.MolToSmiles(Chem.MolFromSmiles(smi)) for smi in smiles]

def mols_to_smiles(mols):
    return [Chem.MolToSmiles(mol) for mol in mols]

def construct_mol(x, adj, atomic_num_list):
    # x: 9, 5; adj: 4, 9, 9
    mol = Chem.RWMol()
    atoms = np.argmax(x, axis=1)
    atoms_exist = (atoms != len(atomic_num_list) - 1)
    atoms = atoms[atoms_exist]              # 9,
    for atom in atoms:
        mol.AddAtom(Chem.Atom(int(atomic_num_list[atom])))
    adj = np.argmax(adj, axis=0)            # 9, 9
    adj = adj[atoms_exist, :][:, atoms_exist]
    for start, end in zip(*np.nonzero(adj)):
        if start > end:
            mol.AddBond(int(start), int(end), bond_decoder[adj[start, end]])
            # add formal charge to atom: e.g. [O+], [N+], [S+]
            # not support [O-], [N-], [S-], [NH+] etc.
            flag, atomid_valence = check_valency(mol)
            if flag:
                continue
            else:
                assert len(atomid_valence) == 2
                idx = atomid_valence[0]
                v = atomid_valence[1]
                an = mol.GetAtomWithIdx(idx).GetAtomicNum()
                if an in (7, 8, 16) and (v - ATOM_VALENCY[an]) == 1:
                    mol.GetAtomWithIdx(idx).SetFormalCharge(1)
    return mol

def check_valency(mol):
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        return True, None
    except ValueError as e:
        e = str(e)
        p = e.find('#')
        e_sub = e[p:]
        atomid_valence = list(map(int, re.findall(r'\d+', e_sub)))
        return False, atomid_valence

def correct_mol(m):
    # xsm = Chem.MolToSmiles(x, isomericSmiles=True)
    mol = m

    #####
    no_correct = False
    flag, _ = check_valency(mol)
    if flag:
        no_correct = True

    while True:
        flag, atomid_valence = check_valency(mol)
        if flag:
            break
        else:
            assert len(atomid_valence) == 2
            idx = atomid_valence[0]
            v = atomid_valence[1]
            queue = []
            for b in mol.GetAtomWithIdx(idx).GetBonds():
                queue.append((b.GetIdx(), int(b.GetBondType()), b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
            queue.sort(key=lambda tup: tup[1], reverse=True)
            if len(queue) > 0:
                start = queue[0][2]
                end = queue[0][3]
                t = queue[0][1] - 1
                mol.RemoveBond(start, end)
                if t >= 1:
                    mol.AddBond(start, end, bond_decoder[t])
    return mol, no_correct

def valid_mol_can_with_seg(m, largest_connected_comp=True):
    if m is None:
        return None
    sm = Chem.MolToSmiles(m, isomericSmiles=True)
    if largest_connected_comp and '.' in sm:
        vsm = [(s, len(s)) for s in sm.split('.')]  # 'C.CC.CCc1ccc(N)cc1CCC=O'.split('.')
        vsm.sort(key=lambda tup: tup[1], reverse=True)
        mol = Chem.MolFromSmiles(vsm[0][0])
    else:
        mol = Chem.MolFromSmiles(sm)
    return mol

def mols_to_nx(mols):
    nx_graphs = []
    for mol in mols:
        G = nx.Graph()
        if mol is None:
            continue
        for atom in mol.GetAtoms():
            G.add_node(atom.GetIdx(),
                       label=atom.GetSymbol())
                    #    atomic_num=atom.GetAtomicNum(),
                    #    formal_charge=atom.GetFormalCharge(),
                    #    chiral_tag=atom.GetChiralTag(),
                    #    hybridization=atom.GetHybridization(),
                    #    num_explicit_hs=atom.GetNumExplicitHs(),
                    #    is_aromatic=atom.GetIsAromatic())
                    
        for bond in mol.GetBonds():
            G.add_edge(bond.GetBeginAtomIdx(),
                       bond.GetEndAtomIdx(),
                       label=int(bond.GetBondTypeAsDouble()))
                    #    bond_type=bond.GetBondType())
        
        nx_graphs.append(G)
    return nx_graphs

##### NSPDK #####

def smiles_to_mols(smiles):
    return [Chem.MolFromSmiles(s) for s in smiles]

def nspdk_stats(graph_ref_list, graph_pred_list, is_parallel=False):
    graph_pred_list_remove_empty = [G for G in graph_pred_list if not G.number_of_nodes() == 0]

    prev = datetime.now()
    mmd_dist = compute_nspdk_mmd(graph_ref_list, graph_pred_list_remove_empty, n_jobs=20)
    elapsed = datetime.now() - prev
    return mmd_dist

def eval_graph_list(graph_ref_list, grad_pred_list):
    """
    Evaluate the graph statistics given networkx graphs of reference and generated graphs.
    @param graph_ref_list: list of networkx graphs
    @param grad_pred_list: list of networkx graphs
    @param methods: list of methods to evaluate
    @return: a dictionary of results
    """
    results = nspdk_stats(graph_ref_list, grad_pred_list, is_parallel=False)
    return results

def compute_nspdk_mmd(samples1, samples2, n_jobs=None):
    """
    Compute MMD between two sets of samples using NSPDK kernel.
    Code adapted from https://github.com/idea-iitd/graphgen/blob/master/metrics/mmd.py
    """
    def _kernel_compute(x, y=None, n_jobs_=None):
        x = vectorize(x, complexity=4, discrete=True)
        if y is not None:
            y = vectorize(y, complexity=4, discrete=True)
        return pairwise_kernels(x, y, metric='linear', n_jobs=n_jobs_)

    X = _kernel_compute(samples1, n_jobs_=n_jobs)
    Y = _kernel_compute(samples2, n_jobs_=n_jobs)
    Z = _kernel_compute(samples1, y=samples2, n_jobs_=n_jobs)

    return np.average(X) + np.average(Y) - 2 * np.average(Z)

##### Make final metric function ####

# [B, N, num_node_types] for node_recon, [B, num_bond_types, N, N] for adj_recon
def get_evaluation_metrics(node_one_hot, adj_one_hot, dataset_name):
    """
    Evaluate molecules from one-hot encoded node and adjacency tensors
    Args:
        node_one_hot: One-hot encoded node features tensor [B, N, num_node_type]
        adj_one_hot: One-hot encoded adjacency tensor [B, num_adj_type, N, N]
        dataset_name: Name of the dataset ('qm9' or 'zinc')
    """
    node_one_hot = node_one_hot.detach().cpu().numpy()
    adj_one_hot  = adj_one_hot.detach().cpu().numpy()

    train_smiles, test_smiles = load_smiles(dataset_name)
    train_smiles = canonicalize_smiles(train_smiles)
    test_smiles = canonicalize_smiles(test_smiles)
    # test_graph_list = mols_to_nx(smiles_to_mols(test_smiles))

    if dataset_name == 'qm9':
        atomic_num_list = [6, 7, 8, 9, 0]
    elif dataset_name == 'zinc':
        atomic_num_list = [6, 7, 8, 9, 15, 16, 17, 35, 53, 0]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    gen_mols, num_no_correct = [], 0
    for x, a in zip(node_one_hot, adj_one_hot):
        mol = construct_mol(x, a, atomic_num_list)
        c_mol, no_correct = correct_mol(mol)
        if no_correct:
           num_no_correct += 1
        vc_mol = valid_mol_can_with_seg(c_mol, largest_connected_comp=False)
        if vc_mol is not None:
            gen_mols.append(vc_mol)

    ##### Convert to SMILES #####
    gen_mols = [mol for mol in gen_mols if mol is not None]  # remove None molecules
    gen_smiles = mols_to_smiles(gen_mols)
    gen_smiles = [smi for smi in gen_smiles if len(smi)]  # remove empty smiles

    # Evaluate metrics
    scores = get_all_metrics(gen=gen_smiles, k=len(gen_smiles), device='cpu', n_jobs=1, test=test_smiles, train=train_smiles)
    # scores_nspdk = eval_graph_list(test_graph_list, mols_to_nx(gen_mols))

    valid_wo_correction = float(num_no_correct / len(gen_mols))
    uniqueness = scores[f'unique@{len(gen_smiles)}']
    novelty = scores['Novelty']
    # fcd = scores['FCD/Test']

    return valid_wo_correction, uniqueness, novelty
