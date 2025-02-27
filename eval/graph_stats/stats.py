
import concurrent.futures
import os
import subprocess as sp
from datetime import datetime

from scipy.linalg import eigvalsh
import networkx as nx
import numpy as np
import copy

from .mmd import process_tensor, compute_mmd, gaussian, gaussian_emd, compute_nspdk_mmd
PRINT_TIME = True
ORCA_DIR = './graph_stats/orca'  # the relative path to the orca dir


def degree_worker(G):
    return np.array(nx.degree_histogram(G))


def add_tensor(x, y):
    x, y = process_tensor(x, y)
    return x + y


def degree_stats(graph_ref_list, graph_pred_list, is_parallel=True):
    ''' Compute the distance between the degree distributions of two unordered sets of graphs.
    Args:
      graph_ref_list, graph_target_list: two lists of networkx graphs to be evaluated
    '''
    sample_ref = []
    sample_pred = []
    # in case an empty graph is generated
    graph_pred_list_remove_empty = [G for G in graph_pred_list if not G.number_of_nodes() == 0]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for deg_hist in executor.map(degree_worker, graph_ref_list):
                sample_ref.append(deg_hist)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for deg_hist in executor.map(degree_worker, graph_pred_list_remove_empty):
                sample_pred.append(deg_hist)

    else:
        for i in range(len(graph_ref_list)):
            degree_temp = np.array(nx.degree_histogram(graph_ref_list[i]))
            sample_ref.append(degree_temp)
        for i in range(len(graph_pred_list_remove_empty)):
            degree_temp = np.array(nx.degree_histogram(graph_pred_list_remove_empty[i]))
            sample_pred.append(degree_temp)
    mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd)
    #mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian)
    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print('Time computing degree mmd: ', elapsed)
    return mmd_dist


###############################################################################

def spectral_worker(G):
    # eigs = nx.laplacian_spectrum(G)
    eigs = eigvalsh(nx.normalized_laplacian_matrix(G).todense())
    spectral_pmf, _ = np.histogram(eigs, bins=200, range=(-1e-5, 2), density=False)
    spectral_pmf = spectral_pmf / spectral_pmf.sum()
    # from scipy import stats
    # kernel = stats.gaussian_kde(eigs)
    # positions = np.arange(0.0, 2.0, 0.1)
    # spectral_density = kernel(positions)

    # import pdb; pdb.set_trace()
    return spectral_pmf


def spectral_stats(graph_ref_list, graph_pred_list, is_parallel=True):
    ''' Compute the distance between the degree distributions of two unordered sets of graphs.
    Args:
      graph_ref_list, graph_target_list: two lists of networkx graphs to be evaluated
    '''
    sample_ref = []
    sample_pred = []
    # in case an empty graph is generated
    graph_pred_list_remove_empty = [
        G for G in graph_pred_list if not G.number_of_nodes() == 0
    ]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for spectral_density in executor.map(spectral_worker, graph_ref_list):
                sample_ref.append(spectral_density)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for spectral_density in executor.map(spectral_worker, graph_pred_list_remove_empty):
                sample_pred.append(spectral_density)

        # with concurrent.futures.ProcessPoolExecutor() as executor:
        #   for spectral_density in executor.map(spectral_worker, graph_ref_list):
        #     sample_ref.append(spectral_density)
        # with concurrent.futures.ProcessPoolExecutor() as executor:
        #   for spectral_density in executor.map(spectral_worker, graph_pred_list_remove_empty):
        #     sample_pred.append(spectral_density)
    else:
        for i in range(len(graph_ref_list)):
            spectral_temp = spectral_worker(graph_ref_list[i])
            sample_ref.append(spectral_temp)
        for i in range(len(graph_pred_list_remove_empty)):
            spectral_temp = spectral_worker(graph_pred_list_remove_empty[i])
            sample_pred.append(spectral_temp)
    # print(len(sample_ref), len(sample_pred))

    mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd)
    # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=emd)
    # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd)
    # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian)

    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print('Time computing degree mmd: ', elapsed)
    return mmd_dist


###############################################################################

def clustering_worker(param):
    G, bins = param
    clustering_coeffs_list = list(nx.clustering(G).values())
    hist, _ = np.histogram(
        clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
    return hist


def clustering_stats(graph_ref_list, graph_pred_list, bins=100, is_parallel=True):
    sample_ref = []
    sample_pred = []
    graph_pred_list_remove_empty = [G for G in graph_pred_list if not G.number_of_nodes() == 0]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for clustering_hist in executor.map(clustering_worker,
                                                [(G, bins) for G in graph_ref_list]):
                sample_ref.append(clustering_hist)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for clustering_hist in executor.map(clustering_worker,
                                                [(G, bins) for G in graph_pred_list_remove_empty]):
                sample_pred.append(clustering_hist)
        # check non-zero elements in hist
        # total = 0
        # for i in range(len(sample_pred)):
        #    nz = np.nonzero(sample_pred[i])[0].shape[0]
        #    total += nz
        # print(total)
    else:
        for i in range(len(graph_ref_list)):
            clustering_coeffs_list = list(nx.clustering(graph_ref_list[i]).values())
            hist, _ = np.histogram(
                clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
            sample_ref.append(hist)

        for i in range(len(graph_pred_list_remove_empty)):
            clustering_coeffs_list = list(nx.clustering(graph_pred_list_remove_empty[i]).values())
            hist, _ = np.histogram(
                clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
            sample_pred.append(hist)
    mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd,
                          sigma=1.0 / 10, distance_scaling=bins)
    # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian,
    #                        sigma=1.0 / 10)
    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print('Time computing clustering mmd: ', elapsed)
    return mmd_dist


# maps motif/orbit name string to its corresponding list of indices from orca output
motif_to_indices = {
    '3path': [1, 2],
    '4cycle': [8],
}
COUNT_START_STR = 'orbit counts: \n'


def edge_list_reindexed(G):
    idx = 0
    id2idx = dict()
    for u in G.nodes():
        id2idx[str(u)] = idx
        idx += 1

    edges = []
    for (u, v) in G.edges():
        edges.append((id2idx[str(u)], id2idx[str(v)]))
    return edges


def orca(graph):
    tmp_file_path = os.path.join(ORCA_DIR, 'tmp.txt')
    print(tmp_file_path)
    f = open(tmp_file_path, 'w')
    f.write(str(graph.number_of_nodes()) + ' ' + str(graph.number_of_edges()) + '\n')
    for (u, v) in edge_list_reindexed(graph):
        f.write(str(u) + ' ' + str(v) + '\n')
    f.close()
    output = sp.check_output([os.path.join(ORCA_DIR, 'orca'), 'node', '4', tmp_file_path, 'std'])
    #output = sp.check_output([os.path.join(ORCA_DIR), 'node', '4', tmp_file_path, 'std'])
    print(output)
    output = output.decode('utf8').strip()

    idx = output.find(COUNT_START_STR) + len(COUNT_START_STR)
    output = output[idx:]
    node_orbit_counts = np.array([list(map(int, node_cnts.strip().split(' ')))
                                  for node_cnts in output.strip('\n').split('\n')])

    try:
        os.remove(tmp_file_path)
    except OSError:
        pass

    return node_orbit_counts


def orbit_stats_all(graph_ref_list, graph_pred_list):
    total_counts_ref = []
    total_counts_pred = []

    # graph_pred_list_remove_empty = [G for G in graph_pred_list if not G.number_of_nodes() == 0]

    for G in graph_ref_list:
        try:
            orbit_counts = orca(G)
            print(orbit_counts)
        except Exception as e:
            print(e)
            continue
        orbit_counts_graph = np.sum(orbit_counts, axis=0) / G.number_of_nodes()
        total_counts_ref.append(orbit_counts_graph)

    for G in graph_pred_list:
        try:
            orbit_counts = orca(G)
        except:
            continue
        orbit_counts_graph = np.sum(orbit_counts, axis=0) / G.number_of_nodes()
        total_counts_pred.append(orbit_counts_graph)

    total_counts_ref = np.array(total_counts_ref)
    total_counts_pred = np.array(total_counts_pred)
    print(total_counts_ref, total_counts_pred)
    mmd_dist = compute_mmd(total_counts_ref, total_counts_pred, kernel=gaussian,
                           is_hist=False, sigma=30.0)

    print('-------------------------')
    print(np.sum(total_counts_ref, axis=0) / len(total_counts_ref))
    print('...')
    print(np.sum(total_counts_pred, axis=0) / len(total_counts_pred))
    print('-------------------------')
    return mmd_dist


def adjs_to_graphs(adjs, node_flags=None):
    graph_list = []
    for adj in adjs:
        G = nx.from_numpy_array(adj)
        G.remove_edges_from(nx.selfloop_edges(G))
        G.remove_nodes_from(list(nx.isolates(G)))
        if not nx.is_empty(G):
            largest_component = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest_component).copy()
        else:
            G.add_node(1)
        graph_list.append(G)
    return graph_list


def eval_acc_lobster_graph(G_list):
    G_list = [copy.deepcopy(gg) for gg in G_list]

    count = 0
    for gg in G_list:
        if is_lobster_graph(gg):
            count += 1

    return count / float(len(G_list))


def is_lobster_graph(G):
    """
    Check a given graph is a lobster graph or not
    Removing leaf nodes twice:
    lobster -> caterpillar -> path
  """
    ### Check if G is a tree
    if nx.is_tree(G):
        # import pdb; pdb.set_trace()
        ### Check if G is a path after removing leaves twice
        leaves = [n for n, d in G.degree() if d == 1]
        G.remove_nodes_from(leaves)

        leaves = [n for n, d in G.degree() if d == 1]
        G.remove_nodes_from(leaves)

        num_nodes = len(G.nodes())
        num_degree_one = [d for n, d in G.degree() if d == 1]
        num_degree_two = [d for n, d in G.degree() if d == 2]

        if sum(num_degree_one) == 2 and sum(num_degree_two) == 2 * (num_nodes - 2):
            return True
        elif sum(num_degree_one) == 0 and sum(num_degree_two) == 0:
            return True
        else:
            return False
    else:
        return False

def nspdk_stats(graph_ref_list, graph_pred_list, is_parallel=False):
    graph_pred_list_remove_empty = [G for G in graph_pred_list if not G.number_of_nodes() == 0]

    prev = datetime.now()
    mmd_dist = compute_nspdk_mmd(graph_ref_list, graph_pred_list_remove_empty, n_jobs=None)
    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print('Time computing nspdk mmd: ', elapsed)
    return mmd_dist


METHOD_NAME_TO_FUNC = {
    'degree': degree_stats,
    'cluster': clustering_stats,
    'orbit': orbit_stats_all,
    'spectral': spectral_stats,
    'nspdk': nspdk_stats
}


def eval_torch_batch(ref_batch, pred_batch, methods=None):
    graph_ref_list = adjs_to_graphs(ref_batch.detach().cpu().numpy())
    graph_pred_list = adjs_to_graphs(pred_batch.detach().cpu().numpy())
    results = eval_graph_list(graph_ref_list, graph_pred_list, methods=methods)
    return results


# -------- Evaluate generated generic graphs --------
# def eval_graph_list(graph_ref_list, graph_pred_list, methods=None, kernels=None):
#     if methods is None:
#         methods = ['degree', 'cluster', 'orbit']
#     results = {}
#     kernels = {'degree': gaussian_emd,
#                'cluster': gaussian_emd,
#                'orbit': gaussian,
#                'spectral': gaussian_emd}
#     for method in methods:
#         if method == 'nspdk':
#             results[method] = METHOD_NAME_TO_FUNC[method](graph_ref_list, graph_pred_list)
#         else:
#             results[method] = round(METHOD_NAME_TO_FUNC[method](graph_ref_list, graph_pred_list), 6)
#         print('\033[91m' + f'{method:9s}' + '\033[0m' + ' : ' + '\033[94m' +  f'{results[method]:.6f}' + '\033[0m')
#     return results
def eval_graph_list(graph_ref_list, grad_pred_list, methods=None):
    """
    Evaluate the graph statistics given networkx graphs of reference and generated graphs.
    @param graph_ref_list: list of networkx graphs
    @param grad_pred_list: list of networkx graphs
    @param methods: list of methods to evaluate
    @return: a dictionary of results
    """
    if methods is None:
        methods = ['degree', 'cluster', 'orbit', 'spectral']
    print("Size of reference graphs: {:d}, size of generated graphs: {:d}".format(
        len(graph_ref_list), len(grad_pred_list)))
    results = {}
    for method in methods:
        results[method] = METHOD_NAME_TO_FUNC[method](graph_ref_list, grad_pred_list)
    results['average'] = np.mean(list(results.values()))
    print(results)
    return results

##################

if __name__ == '__main__':
    import networkx as nx
    import numpy as np

    # Create sample graphs for testing
    num_graphs = 5
    graph_ref_list = [nx.erdos_renyi_graph(50, 0.1, seed=i) for i in range(num_graphs)]
    graph_pred_list = [nx.erdos_renyi_graph(50, 0.1, seed=i+num_graphs) for i in range(num_graphs)]

    print("Testing degree statistics:")
    degree_mmd = degree_stats(graph_ref_list, graph_pred_list, is_parallel=False)
    print("Degree MMD:", degree_mmd)

    print("\nTesting spectral statistics:")
    spectral_mmd = spectral_stats(graph_ref_list, graph_pred_list, is_parallel=False)
    print("Spectral MMD:", spectral_mmd)

    print("\nTesting clustering statistics:")
    clustering_mmd = clustering_stats(graph_ref_list, graph_pred_list, bins=10, is_parallel=False)
    print("Clustering MMD:", clustering_mmd)

    print("\nTesting orbit statistics (requires ORCA set up correctly):")
    try:
        orbit_mmd = orbit_stats_all(graph_ref_list, graph_pred_list)
        print("Orbit MMD:", orbit_mmd)
    except Exception as e:
        print("Orbit statistics test failed (check ORCA setup):", e)

    print("\nTesting eval_graph_list:")
    results = eval_graph_list(graph_ref_list, graph_pred_list, methods=['degree', 'cluster', 'spectral'])
    print("Evaluation Results:", results)

    # Optionally, if you have PyTorch installed, test eval_torch_batch
    try:
        import torch
        # Convert graphs to adjacency matrices and then to torch tensors.
        adjs = []
        for g in graph_ref_list:
            A = nx.to_numpy_array(g)
            # Ensure the matrix is symmetric and has no self-loops.
            np.fill_diagonal(A, 0)
            adjs.append(A)
        ref_batch = torch.tensor(np.array(adjs), dtype=torch.float32)
        pred_batch = torch.tensor(np.array(adjs), dtype=torch.float32)
        print("\nTesting eval_torch_batch:")
        torch_results = eval_torch_batch(ref_batch, pred_batch, methods=['degree', 'cluster', 'spectral'])
        print("Torch Batch Evaluation Results:", torch_results)
    except ImportError:
        print("\nPyTorch is not installed; skipping eval_torch_batch test.")
    