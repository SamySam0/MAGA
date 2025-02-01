import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque

def n_community(v_size, p_inter=0.05):
    """
    Generates a two-community graph using the Erdős-Rényi model.
    
    Parameters:
        v_size (int): Total number of nodes in the graph (split evenly between two communities).
        p_inter (float): Fraction of inter-community edges relative to |V|.
    
    Returns:
        NetworkX Graph: A graph with two communities.
    """
    # Split the total nodes into two equal communities
    n = v_size // 2
    
    # Generate Erdős-Rényi graphs for both communities
    # each pair of nodes within the same community has a 30% chance of being connected.
    G1 = nx.erdos_renyi_graph(n, 0.3, seed=1)
    G2 = nx.erdos_renyi_graph(n, 0.3, seed=2)
    
    # Merge both communities into a single graph
    G = nx.disjoint_union(G1, G2)
    
    # Get node lists for both communities
    nodes1 = list(range(n))  # Nodes from the first community
    nodes2 = list(range(n, 2 * n))  # Nodes from the second community
    
    # Number of inter-community edges to add
    # we add exactly 0.05×∣V∣ edges between the two communities
    num_inter_edges = int(p_inter * v_size)
    
    # Add inter-community edges at random
    for _ in range(num_inter_edges):
        n1 = random.choice(nodes1)
        n2 = random.choice(nodes2)
        G.add_edge(n1, n2)
    
    return G

def generate_community_graphs(num_graphs=500):
    """
    Generates a dataset of community graphs.
    
    Parameters:
        num_graphs (int): Number of graphs to generate.
    
    Returns:
        list: A list of generated NetworkX graphs.
    """
    graphs = []
    
    for _ in range(num_graphs):
        # Select a random total size |V| between 60 and 160 (even numbers only)
        v_size = random.choice(range(60, 161, 2))
        
        # Generate a two-community graph
        graphs.append(n_community(v_size, p_inter=0.05))
    
    return graphs

def bfs_ordering(graph, start_node=None):
    """
    Perform BFS on a given graph and return the node ordering π.
    
    Parameters:
        graph (networkx.Graph): Input graph.
        start_node (int, optional): The starting node for BFS. If None, picks the smallest node.
        
    Returns:
        list: BFS node ordering π.
    """

    if start_node is None:
        start_node = min(graphs.nodes)

    visited = set()
    queue = deque([start_node])
    pi = []
    while queue:
        node = queue.popleft()
        if node not in visited:
            visited.add(node)
            pi.append(node)
            queue.extend(neighbor for neighbor in graph.neighbors(node) if neighbor not in visited)
    return pi

def graph_to_S_pi(graph, pi):
    """
    Convert a graph to its adjacency sequence representation S^pi using BFS ordering.
    
    Parameters:
        graph (networkx.Graph): Input graph.
        pi (list): BFS node ordering.
    
    Returns:
        list: List of adjacency vectors S^pi.
    """
    node_index = {node: i for i, node in enumerate(pi)}  # Map nodes to BFS positions
    S_pi = []

    for i, node in enumerate(pi):
        adjacency_vector = [0] * i  # Initialize vector for previous nodes
        for neighbor in graph.neighbors(node):
            if neighbor in node_index and node_index[neighbor] < i:
                adjacency_vector[node_index[neighbor]] = 1
        S_pi.append(adjacency_vector)

    return S_pi

def prepare_dataset(type="community"):
    """
    Convert a list of graphs into BFS orderings and adjacency sequences S^pi.
    
    Parameters:
        graphs (list of networkx.Graph): List of input graphs.
    
    Returns:
        list: List of adjacency sequences S^pi.
    """
    if type == "community":
        graphs = generate_community_graphs(num_graphs=500)
    
    dataset = []

    for graph in graphs:
        start_node = min(graph.nodes)  # Pick smallest node as BFS start for consistency
        # print(f"Start Node: {start_node}")
        pi = bfs_ordering(graph, start_node)  # Step 1: Get BFS ordering
        # print(f"BFS ordering is: {pi}")
        S_pi = graph_to_S_pi(graph, pi)  # Step 2: Convert to adjacency sequence
        # print(f"Adjacency sequence is: {S_pi}")
        dataset.append(S_pi)

    return dataset

