import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv  # Graph Transformer

class GraphEncoder(nn.Module):
    def __init__(self, in_dim, embed_dim, num_layers=3, num_heads=8):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerConv(embed_dim, embed_dim // num_heads, heads=num_heads)
            for _ in range(num_layers)
        ])
        self.input_proj = nn.Linear(in_dim, embed_dim)  # Project node features
        self.output_proj = nn.Linear(embed_dim, embed_dim)  # Project to embedding space

    def forward(self, x, edge_index):
        """
        :param x: Node features (N, in_dim)
        :param edge_index: Edge list (2, num_edges)
        :return: Node embeddings (N, embed_dim)
        """
        x = self.input_proj(x)  # Project input
        for layer in self.layers:
            x = layer(x, edge_index)  # Apply graph transformer layers
            x = torch.relu(x)  # Activation

        return self.output_proj(x)  # Final projection
