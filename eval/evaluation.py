from chem_func import mols_to_smiles
from mol_utils import mols_to_nx, load_smiles, gen_mol
from sample import sample
from func import get_edge_target

from graph_stats.stats import eval_graph_list
from fcd_torch import FCD
import rdkit.Chem as Chem
from rdkit import rdBase
import pickle
import torch
from torch_geometric.utils import to_dense_batch
from torchmetrics import MeanMetric
import random

# def init_autoencoder_running_metrics(annotated_nodes):
#     metric_names = ['loss', 'edge_loss', 'edge_acc', 'edge_acc',
#                     'graph_rec', 'commit', 'codebook', 'perplexity', 'recon_loss']
#     if annotated_nodes:
#         metric_names += ['node_loss', 'node_acc']

#     metrics = {}
#     metrics['train'] = {}
#     metrics['iter'] = {}
#     metrics['val'] = {}
#     for metric_step in metrics:
#         for metric in metric_names:
#             metrics[metric_step][metric] = MeanMetric()
#     return metrics

# def init_prior_running_metrics():
#     metric_names = ['loss']
#     metrics = {}
#     metrics['train'] = {}
#     metrics['iter'] = {}
#     metrics['val'] = {}
#     for metric_step in metrics:
#         for metric in metric_names:
#             metrics[metric_step][metric] = MeanMetric()
#     return metrics

# def eval_sample(transformer, quantizer, decoder, dataset):
#     print(dataset)
#     if dataset == 'zinc' or dataset == 'qm9':
#         N_SAMPLE = 1000
#         annots, adjs = sample(N_SAMPLE, transformer, quantizer, decoder)
#         gen_mols, num_no_correct = gen_mol(annots, adjs, dataset)
#         metrics = get_mol_metric(gen_mols, dataset, num_no_correct)
#         log_iter.log(metrics)
#         log_iter.save_prior(metrics['nspdk'], transformer, opt, scheduler, step,
#                                  name='nspdk')
#     else:
#         N_SAMPLE = 117
#         annots, adjs = sample(N_SAMPLE, transformer, quantizer, decoder)
#         ref = to_dense_adj(batch.edge_index, batch=batch.batch, max_num_nodes=max_node_num)
#         metrics = eval_torch_batch(ref_batch=ref, pred_batch=adjs, methods=None)
#         metrics['avg'] = sum(metrics.values()) / 3
#         log_iter.log(metrics)
#         log_iter.save_prior(metrics['avg'], transformer, opt, scheduler, step,
#                                  name='Avg')


def reconstruction_stats(batch, edges_rec, nodes_rec, masks_nodes, masks_edges, n_node_feat):
    '''
    Args:
        edges_rec (tensor): The binary dense tensors of the reconstructed edge types (batch size x n x n x #edge_types)
        edges_rec (tensor): The dense tensors of the true edge types (batch size x n x n)
        edges_rec (tensor): The binary dense tensor of the true edge types (n x n x #edge_types)
        masks (tensor, bool): The dense tensor masking where there is possible edges.
    Return:
        tensor: the number of edges correctly reconstructed by graph.
        tensor: the rate of reconstruction by graph.
        int: the number of graphs, for which the edges are reconstructed totally correctly
    '''

    edges_true = get_edge_target(batch)
    if nodes_rec is not None:
        max_node_num = nodes_rec.shape[1]
        dense_nodes, _ = to_dense_batch(batch.x, batch.batch, max_num_nodes=max_node_num)
        nodes_true = dense_nodes[:, :, :n_node_feat].argmax(-1)
        nodes_rec = nodes_rec.argmax(-1)
        nodes_rec[(1 - masks_nodes.int()).bool()] = 0
        correct_nodes = nodes_rec == nodes_true
        n_nodes_corr = correct_nodes.sum(-1)
        all_nodes_corr = n_nodes_corr == nodes_rec.shape[-1]

        n_potential_nodes = nodes_rec.shape[0]*nodes_rec.shape[1]
        n_nodes = masks_nodes.sum()

        err_nodes = n_potential_nodes - n_nodes_corr.sum()
        acc_nodes = 1-err_nodes/n_nodes

    if edges_rec.shape[-1] > 1:
        edges_rec = edges_rec.argmax(-1)
    else:
        edges_rec = edges_rec.round()
    edges_rec[(1-masks_edges).squeeze().bool()] = 0

    edge_rec = edges_rec.squeeze() == edges_true

    n_edges_corr = edge_rec.sum([1, 2])
    n_potential_edges = edge_rec.shape[0] * edges_rec.shape[1] * edges_rec.shape[2]
    n_edges = masks_edges.sum()
    err_edges = n_potential_edges - n_edges_corr.sum()

    acc_edges = 1 - err_edges / n_edges

    all_edges_corr = (n_edges_corr == edges_rec.shape[1] ** 2)
    if nodes_rec is not None:
        graphs_corr = (all_nodes_corr*all_edges_corr)
    else:
        graphs_corr = all_edges_corr.sum()
        err_nodes, acc_nodes = None, None
    n_graphs_corr = graphs_corr.sum()
    return err_edges, acc_edges, err_nodes, acc_nodes, n_graphs_corr, graphs_corr

def get_mol(smiles_or_mol):
    '''
    Loads SMILES/molecule into RDKit's object
    '''
    if isinstance(smiles_or_mol, str):
        if len(smiles_or_mol) == 0:
            return None
        mol = Chem.MolFromSmiles(smiles_or_mol)
        if mol is None:
            return None
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            return None
        return mol
    return smiles_or_mol


def canonic_smiles(smiles_or_mol):
    mol = get_mol(smiles_or_mol)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)

def fraction_unique(gen, k=None, check_validity=True):
    """
    Computes a number of unique molecules
    Parameters:
        gen: list of SMILES
        k: compute unique@k
        n_jobs: number of threads for calculation
        check_validity: raises ValueError if invalid molecules are present
    """
    if k is not None:
        if len(gen) < k:
            raise ValueError(f"Can't compute unique@{k} gen contains only {len(gen)} molecules")
        gen = gen[:k]
    canonic = set(map(canonic_smiles, gen))
    if None in canonic and check_validity:
        raise ValueError("Invalid molecule passed to unique@k")
    return len(canonic) / len(gen)

def remove_invalid(gen, canonize=True):
    """
    Removes invalid molecules from the dataset
    """

    if not canonize:
        mols = get_mol(gen)
        return [gen_ for gen_, mol in zip(gen, mols) if mol is not None]
    return [x for x in map(canonic_smiles, gen) if x is not None]

def fraction_valid(gen):
    """
    Computes a number of valid molecules
    Parameters:
        gen: list of SMILES
    """
    gen = [mol for mol in map(get_mol, gen)]
    return 1 - gen.count(None) / len(gen)

def novelty(gen, train):
    gen_smiles = []
    for smiles in gen:
        gen_smiles.append(canonic_smiles(smiles))
    gen_smiles_set = set(gen_smiles) - {None}
    train_set = set(train)
    return len(gen_smiles_set - train_set) / len(gen_smiles_set)

def get_mol_metric(gen_mols, dataset, num_no_correct, train_smiles=None):
    '''
    Args:
        - graphs(list of torch_geometric.Data)
        - train_smiles (list of smiles from the training set)
    Return:
        - Dict with key valid, unique, novel nspdk
    '''
    metrics = {}
    rdBase.DisableLog('rdApp.*')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_smiles, test_smiles = load_smiles(dataset=dataset) # 'QM9' or 'ZINC'
    gen_smiles = mols_to_smiles(gen_mols)
    metrics['valid'] = num_no_correct
    gen_valid = remove_invalid(gen_smiles)
    metrics['unique'] = fraction_unique(gen_valid, k=None, check_validity=True)
    if train_smiles is not None:
        metrics['novel'] = novelty(gen_valid, train_smiles)
    else:
        metrics['novel'] = None

    # with open(f'./data/{dataset.lower()}_test_nx.pkl', 'rb') as f:
    #     test_graph_list = pickle.load(f)
    #     random.Random(42).shuffle(test_graph_list)
    # metrics['nspdk'] = eval_graph_list(test_graph_list[:1000], mols_to_nx(gen_mols), methods=['nspdk'])['nspdk']
    # TODO: Make the computation of fcd more efficient
    # print(gen_smiles)
    metrics['fcd'] = FCD(n_jobs=0, device=device)(ref=test_smiles, gen=gen_smiles)
    metrics['valid_with_corr'] = len(gen_valid)
    return metrics

def qm9_eval(node_recon, edge_recon, dataset='qm9'):
    gen_mols, num_no_correct = gen_mol(node_recon, edge_recon, dataset)
    metrics = get_mol_metric(gen_mols, dataset, num_no_correct)
    valid, unique, novel, fcd, valid_w_corr = metrics.values()
    return valid, unique, novel, fcd, valid_w_corr
    

############################################################

if __name__ == '__main__':
    import torch
    import random
    import rdkit.Chem as Chem
    from torchmetrics import MeanMetric
    from torch_geometric.utils import to_dense_adj
    from torch_geometric.data import Data, Batch

    # # Test the functions that don't depend on external data first
    # print("=== Testing init_autoencoder_running_metrics ===")
    # metrics_auto = init_autoencoder_running_metrics(annotated_nodes=True)
    # print("Autoencoder metrics keys:", list(metrics_auto.keys()))

    # print("\n=== Testing init_prior_running_metrics ===")
    # metrics_prior = init_prior_running_metrics()
    # print("Prior metrics keys:", list(metrics_prior.keys()))

    print("\n=== Testing get_mol and canonic_smiles ===")
    test_smiles = "CCO"  # ethanol
    mol = get_mol(test_smiles)
    print("Molecule from SMILES:", mol)
    canon = canonic_smiles(test_smiles)
    print("Canonical SMILES:", canon)

    print("\n=== Testing fraction_valid and fraction_unique ===")
    test_smiles_list = ["CCO", "CCO", "OCC", "invalid", ""]
    try:
        valid_fraction = fraction_valid(test_smiles_list)
        print("Fraction valid:", valid_fraction)
    except Exception as e:
        print("fraction_valid error:", e)

    try:
        unique_fraction = fraction_unique(test_smiles_list, k=len(test_smiles_list), check_validity=False)
        print("Fraction unique:", unique_fraction)
    except Exception as e:
        print("fraction_unique error:", e)

    print("\n=== Testing remove_invalid ===")
    cleaned_smiles = remove_invalid(test_smiles_list, canonize=True)
    print("Cleaned SMILES:", cleaned_smiles)

    print("\n=== Testing novelty ===")
    train_set = {"CCO", "OCC"}
    novelty_val = novelty(test_smiles_list, train_set)
    print("Novelty:", novelty_val)

    print("\n=== Testing eval_sample for QM9 dataset (dummy run) ===")
    # eval_sample depends on external components (e.g., transformer, quantizer, decoder, log_iter, and the proper QM9 data files).
    # Here we use dummy arguments and wrap in try/except to catch errors if the environment isn't fully set up.
    try:
        eval_sample(None, None, None, 'qm9')
    except Exception as e:
        print("eval_sample (dummy QM9) encountered an error (expected if dependencies are missing):", e)

    print("\n=== Testing reconstruction_stats with dummy tensors ===")
    try:
        # Create a dummy torch_geometric Data object
        x = torch.tensor([[1, 0], [0, 1], [1, 0]], dtype=torch.float)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index)
        batch_obj = Batch.from_data_list([data])
        num_graphs, n_nodes, num_edge_types = batch_obj.num_graphs, 3, 2
        edges_rec = torch.randint(0, 2, (num_graphs, n_nodes, n_nodes, num_edge_types), dtype=torch.float)
        nodes_rec = torch.rand(num_graphs, n_nodes, 2)
        masks_nodes = torch.ones(num_graphs, n_nodes, dtype=torch.bool)
        masks_edges = torch.ones(num_graphs, n_nodes, n_nodes, dtype=torch.bool)
        n_node_feat = 2
        recon_stats = reconstruction_stats(batch_obj, edges_rec, nodes_rec, masks_nodes, masks_edges, n_node_feat)
        print("Reconstruction stats:", recon_stats)
    except Exception as e:
        print("reconstruction_stats encountered an error:", e)

    print("\n=== Testing get_mol_metric (dummy run) ===")
    try:
        # For this dummy test, create a few molecules using RDKit and evaluate molecular metrics.
        dummy_gen_mols = [Chem.MolFromSmiles("CCO") for _ in range(5)]
        print(dummy_gen_mols)
        dummy_num_no_correct = 5
        # This call will try to load './data/qm9_test_nx.pkl'. Ensure the file exists or catch the error.
        metrics_mol = get_mol_metric(dummy_gen_mols, 'QM9', dummy_num_no_correct)
        print("Molecular metrics:", metrics_mol)
    except Exception as e:
        print("get_mol_metric (dummy QM9) encountered an error (expected if external dependencies/files are missing):", e)
