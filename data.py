import numpy as np
import networkx as nx
import torch
import random
import pickle as pkl

def generate_adjacency_vector_sequence(g, node_sequence):
    """
    Generate the adjacency vector sequence for a given graph and node sequence.
    Input:
        g: NX graph             Graph to generate sequence
        node_sequence: List     Node sequence to generate sequence
    Output:
        adjacency_vector_sequence: np.ndarray     Adjacency vector sequence (lower triangular)
    """

    return np.tril(nx.adjacency_matrix(g, node_sequence).toarray())

def bfs(g):
    """
    Perform a BFS on a permuted version of g.

    Input:
        g: NX graph             Graph to find BFS sequence
    Output:
        permuted_graph, traversal
    """

    a = nx.to_numpy_array(g)
    reordering = np.random.permutation(g.number_of_nodes())
    permuted_graph = nx.from_numpy_array(a[reordering, :][:, reordering])

    comps = [list(comp) for comp in nx.connected_components(permuted_graph)]
    traversal = []

    # Perform BFS on each connected component
    for comp in comps:
        successor_listing = [node[1] for node in nx.bfs_successors(permuted_graph, comp[0])]
        traversal.append(comp[0])
        for successor_list in successor_listing:
            for successor in successor_list:
                traversal.append(successor)

    return permuted_graph, traversal


class GraphDataSet(torch.utils.data.Dataset):
    """
    A PyTorch Dataset for Graph Data that supports BFS-based reordering
    and flexible splitting into train, val, and test sets.
    """

    def __init__(
        self, 
        dataset: str, 
        m: int = None, 
        use_bfs: bool = True, 
        split: str = "train", 
        train_split: float = 0.8, 
        val_split: float = 0.1
    ):
        """
        Initialize the GraphDataSet.

        Args:
            dataset (str):         Name of dataset to load ("community", "community-small", etc.)
            m (int):               Precalculated M-value (Ref. paper), controls the window size for adjacency vectors.
            use_bfs (bool):        Whether to use BFS reordering for node sequences.
            split (str):           Which subset to load: "train", "val", or "test".
            train_split (float):   Fraction of data to use for training.
            val_split (float):     Fraction of data to use for validation.
                                   The remainder goes to test = 1 - train_split - val_split.

        Raises:
            Exception: If dataset not supported or if splits are invalid.
        """

        self.m = m
        self.use_bfs = use_bfs
        np.random.seed(42)  # For reproducibility of random splits in BFS reordering, etc.

        # Load the graphs
        if dataset == 'community':
            self.graphs = self.load_community_dataset()
        elif dataset == 'community-small':
            self.graphs = self.load_community_dataset(min_nodes=12, max_nodes=20)
        else:
            raise Exception(f"No data-loader for dataset `{dataset}`")

        # Remove any self-loops (if present)
        for g in self.graphs:
            g.remove_edges_from(list(nx.selfloop_edges(g)))

        # Check splits
        if train_split + val_split > 1.0:
            raise ValueError(
                f"train_split + val_split = {train_split + val_split} which is > 1.0. "
                f"Please ensure they sum to <= 1.0"
            )
        
        data_size = len(self.graphs)
        train_size = int(data_size * train_split)
        val_size = int(data_size * val_split)
        test_size = data_size - train_size - val_size  # the remainder

        # Shuffle indices to randomize which graphs end up in train/val/test
        all_indices = np.arange(data_size)
        np.random.shuffle(all_indices)

        # Partition the dataset indices
        train_indices = all_indices[:train_size]
        val_indices = all_indices[train_size : train_size + val_size]
        test_indices = all_indices[train_size + val_size :]

        # Based on `split` argument, pick the correct subset
        if split == "train":
            self.indices = train_indices
        elif split == "val":
            self.indices = val_indices
        elif split == "test":
            self.indices = test_indices
        else:
            raise ValueError("split must be one of: ['train', 'val', 'test']")

        # Compute the maximum number of nodes in any graph (for zero-padding)
        self.max_node_count = 0
        for idx in self.indices:
            G = self.graphs[idx]
            if G.number_of_nodes() > self.max_node_count:
                self.max_node_count = G.number_of_nodes()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """
        Fetch a single graph sample from the dataset.

        Returns:
            {
              'x': 2D np.ndarray of shape (max_node_count-1, m) [padded adjacency vectors]
              'len': int, actual number of adjacency vectors used (num_nodes - 1)
            }
        """
        # Convert the 'idx' to the actual index in the underlying graphs list
        true_index = self.indices[idx]
        G = self.graphs[true_index]

        # Either BFS permute or use an integer labeling + random permutation
        if self.use_bfs:
            permuted_g, bfs_seq = bfs(G)
            adjacency_vector_seq = generate_adjacency_vector_sequence(permuted_g, bfs_seq)
        else:
            # Convert nodes to 0..n-1
            G_int = nx.convert_node_labels_to_integers(G, label_attribute=None)
            nodes = list(G_int.nodes())
            np.random.shuffle(nodes)  # random permutation
            adjacency_vector_seq = generate_adjacency_vector_sequence(G_int, nodes)

        # Build feature vectors from adjacency triangular matrix
        scratch = []
        n = adjacency_vector_seq.shape[0]  # number of nodes
        # If self.m is None, you might decide a default or compute automatically
        m_val = self.m if self.m is not None else n - 1
        
        for i in range(1, n):
            # Only the last i columns up to i, but we only take a window of size m_val
            critical_strip = adjacency_vector_seq[i, max(i - m_val, 0) : i]
            # Reverse + zero-pad to length m_val
            padded_strip = np.pad(critical_strip, (m_val - len(critical_strip), 0))[::-1]
            scratch.append(padded_strip)

        # Convert to numpy
        result = np.array(scratch)  # shape = (n-1, m_val)
        
        # Pad to (max_node_count - 1, m_val)
        pad_rows = self.max_node_count - result.shape[0]
        # If the graph has only 1 node, then result is shape (0, m_val), so we pad carefully
        if pad_rows < 0:
            raise ValueError(
                "Encountered a graph with more nodes than max_node_count. "
                "Check logic or reduce BFS logic or m dimension."
            )
        
        padded_result = np.pad(
            result,
            ((0, pad_rows), (0, 0)),  # pad along the row dimension
            mode='constant',
            constant_values=0
        )

        sample = {
            "x": padded_result,          # shape: (max_node_count-1, m_val)
            "len": result.shape[0]       # actual number of node-based rows
        }

        return sample

    def community_dataset(self, c_sizes, p_inter=0.05, p_intra=0.3):
        """Generate a random 2-community graph based on the Erdős-Rényi model."""
        # Create subgraphs
        g_sub = [nx.gnp_random_graph(c_sizes[i], p=p_intra, directed=False) for i in range(len(c_sizes))]
        G = nx.disjoint_union_all(g_sub)

        # Connect the communities minimally
        g1 = list(g_sub[0].nodes())
        g2 = list(g_sub[1].nodes())

        # Add one inter-community edge to ensure connectivity
        n1 = random.choice(g1)
        n2 = random.choice(g2) + len(g1)
        G.add_edge(n1, n2)

        # Add extra inter-community edges
        V = sum(c_sizes)
        for _ in range(int(p_inter * V)):
            n1 = random.choice(g1)
            n2 = random.choice(g2) + len(g1)
            G.add_edge(n1, n2)

        return G

    def load_community_dataset(
        self, 
        graph_count=500, 
        min_nodes=60, 
        max_nodes=160, 
        num_communities=2,
        p_inter=0.05, 
        p_intra=0.3
    ):
        """Generate `graph_count` random graphs (2-community) in a given node range."""
        retval = []
        for _ in range(graph_count):
            c_sizes = np.random.choice(
                list(range(int(min_nodes/2), int(max_nodes/2)+1)), 
                num_communities
            )
            retval.append(self.community_dataset(c_sizes, p_inter, p_intra))
        return retval
