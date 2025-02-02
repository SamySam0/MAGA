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
        adjacency_vector_sequence: List     Adjacency vector sequence
    """

    return np.tril(nx.adjacency_matrix(g, node_sequence).toarray())


def bfs(g):
    """
    There were two ways to do this. One is that people start from the min node values, one is randomise then do BFS.
    Input:
        g: NX graph             Graph to find reduced BFS sequence
    Output:
        bfs_sequence: List      BFS sequence
    """

    a = nx.to_numpy_array(g)

    reordering = np.random.permutation(g.number_of_nodes())
    permuted_graph = nx.from_numpy_array(a[reordering, :][:, reordering])

    comps = [list(comp) for comp in nx.connected_components(permuted_graph)]

    traversal = []

    # Perform BFS Traversal on Each Component
    for comp in comps:
        successor_listing = [node[1] for node in nx.bfs_successors(permuted_graph, comp[0])]
        
        traversal.append(comp[0])

        for successor_list in successor_listing:
            for successor in successor_list:
                traversal.append(successor)
    
    return permuted_graph, traversal


class GraphDataSet(torch.utils.data.Dataset):
    # ==========================================================
    # PyTorch Dataset for Graph Data (GraphDataSet)
    # ==========================================================
    # This class extends `torch.utils.data.Dataset` to handle 
    # graph-based datasets for deep learning models.
    #
    # Key Features:
    # - Loads various graph datasets (grid, BA, protein, etc.).
    # - Supports BFS-based node reordering for preprocessing.
    # - Splits data into training & testing sets.
    # - Implements `__len__()` to return dataset size.
    # - Implements `__getitem__()` to fetch a graph sample.
    # - Returns adjacency sequences padded to the largest graph.
    #
    # Integration with DataLoader:
    # - Enables efficient batch processing & shuffling.
    # - Supports lazy loading to avoid memory overload.
    # - Works with PyTorch for deep learning models.
    #
    # Usage Example:
    # dataset = GraphDataSet(dataset="grid")
    # data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    # for batch in data_loader:
    #     print(batch["x"].shape, batch["len"])
    #
    # ==========================================================
    def __init__(self, dataset, m=None, bfs=True, training=True, train_split=0.8):
        """
        Initialize the GraphDataSet.
        Input:
            dataset: String      Name of dataset to load
            m: Int               Precalculated M-value (Ref. paper)
            bfs: Bool            Whether to use BFS reordering
            training: Bool       Whether to load training or testing data
            train_split: Float   Proportion of data to use for training
        """

        self.max_node_count = -1
        self.training = training
        self.bfs = bfs
        self.m = m

        np.random.seed(42)

        if dataset == 'community':
            self.graphs = self.load_community_dataset()
        elif dataset == 'community-small':
            self.graphs = self.load_community_dataset(min_nodes=12, max_nodes=20)
        else:
            raise Exception(f"No data-loader for dataset `{dataset}`")

        for g in self.graphs:
            g.remove_edges_from(list(nx.selfloop_edges(g)))

        train_size = int(len(self.graphs) * train_split)
        self.start_idx = 0 if training else train_size
        self.length = train_size if training else len(self.graphs) - train_size

    def __len__(self):
        return  self.length

    def __getitem__(self, idx):
        """
        To use like sample = dataset[0]  # This calls __getitem__(self, idx)
        n.b. Random BFS traversal happens at this stage

        :return :   {'x': <M-length sequence vectors paddded to fit largest graph>,
                    'len': <Number of sequnce vectors actually containing data>}
        """
        g = self.graphs[self.start_idx + idx]

        if self.bfs:
            permuted_g, bfs_seq = bfs(g)
            adjacency_vector_seq = generate_adjacency_vector_sequence(permuted_g, bfs_seq)
        else:
            g = nx.convert_node_labels_to_integers(g)
            adjacency_vector_seq = generate_adjacency_vector_sequence(g, np.random.permutation(g.nodes))

        scratch = []

        for i in range(1, adjacency_vector_seq.shape[0]):
            # Data that actually can have 1s:
            critical_strip = adjacency_vector_seq[i, max(i-self.m, 0):i]
            m_dash = len(critical_strip)
            scratch.append(np.pad(critical_strip, (self.m - m_dash, 0))[::-1])

        result = np.array(scratch)
        return {'x': np.pad(result, [(0, self.max_node_count - result.shape[0]), (0,0)]), 'len': result.shape[0]}
    
    def community_dataset(self, c_sizes, p_inter=0.05, p_intra=0.3):
        """Helper function to a generate random graph using the Erdős-Rényi model.

        :param c_sizes numpy_ndarray:     1-D array of number of nodes in each community in a graph
        :param p_inter Int:               Number of intercommunity edges between communities in a graph
       
        :return graph:                    Random graph generated using the Erdős-Rényi model and given params
        """    
    
        g = [nx.gnp_random_graph(c_sizes[i], p=p_intra, directed=False) for i in range(len(c_sizes))]

        G = nx.disjoint_union_all(g)

        g1 = list(g[0].nodes())
        g2 = list(g[1].nodes())

        # Adding one inter-community edge by default
        # This ensures that we have a connected graph
        n1 = random.choice(g1)
        n2 = random.choice(g2) + len(g1)
        G.add_edge(n1,n2)

        V = sum(c_sizes)
        for i in range(int(p_inter*V)):

            n1 = random.choice(g1)
            n2 = random.choice(g2) + len(g1)
            G.add_edge(n1,n2)


        return G

    def load_community_dataset(self, graph_count=500, min_nodes=60, max_nodes=160, num_communities=2, p_inter=0.05, p_intra=0.3):
        """Generate `graph_count` random graphs using the Erdős-Rényi model.

        :param graph_count Int:     Number of graphs to produce
        :param min_nodes Int:       Minimum number of nodes in any graph
        :param max_nodes Int:       Maximum number of nodes in any graph
        :param p_inter Int:         Number of intercommunity edges in any graph

        :return List:               List of random graphs generated using the Erdős-Rényi model and given params
        """

        retval = []

        for _ in range(graph_count):
            c_sizes = np.random.choice(list(range(int(min_nodes/2),int(max_nodes/2)+1)), num_communities) 
            retval.append(self.community_dataset(c_sizes, p_inter, p_intra))
            self.max_node_count = max(self.max_node_count, sum(c_sizes))

        return retval  

# import torch
# import torch.nn as nn
# import torch.optim as optim

# # Dummy model example (adjust based on your problem)
# class SimpleGraphModel(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super(SimpleGraphModel, self).__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(hidden_dim, output_dim)
    
#     def forward(self, x, seq_lens):
#         # x is of shape (batch_size, max_nodes, m_value)
#         # You might want to process only the valid sequence entries per graph using seq_lens.
#         # For simplicity, we'll just flatten and process all entries.
#         batch_size, max_nodes, m_value = x.size()
#         x = x.view(batch_size * max_nodes, m_value)
#         out = self.fc1(x)
#         out = self.relu(out)
#         out = self.fc2(out)
#         # Reshape back if needed
#         out = out.view(batch_size, max_nodes, -1)
#         return out

# # Define hyperparameters for the model
# input_dim = m_value         # since each input vector has length m_value
# hidden_dim = 64
# output_dim = 10             # adjust based on your task

# # Initialize model, loss function, and optimizer
# model = SimpleGraphModel(input_dim, hidden_dim, output_dim)
# loss_fn = nn.MSELoss()      # example loss; replace with one suitable for your problem
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# num_epochs = 10

# # Training Loop
# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0
#     for batch in train_loader:
#         # batch is a dictionary with keys 'x' and 'len'
#         # 'x' is a numpy array; convert it to a torch tensor (and to float if necessary)
#         x = torch.tensor(batch['x'], dtype=torch.float32)
#         seq_lens = batch['len']  # this tells you the actual sequence length per graph

#         # Zero the parameter gradients
#         optimizer.zero_grad()
        
#         # Forward pass
#         outputs = model(x, seq_lens)
        
#         # Create dummy target for demonstration purposes.
#         # In practice, use your actual target.
#         target = torch.zeros_like(outputs)
        
#         # Compute loss
#         loss = loss_fn(outputs, target)
        
#         # Backward pass and optimize
#         loss.backward()
#         optimizer.step()
        
#         running_loss += loss.item()
    
#     print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")
    
#     # Evaluation on test set
#     model.eval()
#     test_loss = 0.0
#     with torch.no_grad():
#         for batch in test_loader:
#             x = torch.tensor(batch['x'], dtype=torch.float32)
#             seq_lens = batch['len']
#             outputs = model(x, seq_lens)
            
#             # Again, using a dummy target here
#             target = torch.zeros_like(outputs)
#             loss = loss_fn(outputs, target)
#             test_loss += loss.item()
#     print(f"Test Loss: {test_loss/len(test_loader):.4f}")
