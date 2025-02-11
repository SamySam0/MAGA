from torch_geometric.nn import GCNConv, GATConv
import torch
import torch.nn as nn
from models.basic_var import AdaLNSelfAttn
from functools import partial


class GraphVAR(nn.Module):
    def __init__(self, 
                 graph_encoder,  # Replace VQVAE with a Graph Encoder
                 num_nodes,      # Max number of nodes (or dynamic)
                 embed_dim=512, 
                 num_heads=8, 
                 depth=6,
                 drop_rate=0.1):
        super().__init__()

        # Graph Encoder (e.g., GCN, GAT, or Transformer-based)
        self.graph_encoder = graph_encoder  

        # Node & Edge Embeddings
        self.node_embed = nn.Embedding(num_nodes, embed_dim)
        self.edge_embed = nn.Embedding(num_nodes * num_nodes, embed_dim)  # If needed

        # Define normalization layer
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        # Compute dropout rates per block (stochastic depth)
        dpr = [x.item() for x in torch.linspace(0, drop_rate, depth)]

        # Transformer Blocks for Autoregression (Fixed)
        self.blocks = nn.ModuleList([
            AdaLNSelfAttn(
                cond_dim=embed_dim,
                shared_aln=False,
                norm_layer=norm_layer,
                num_heads=num_heads,
                embed_dim=embed_dim,
                drop=drop_rate,
                attn_drop=drop_rate,  # Assuming attn_drop_rate is same as drop_rate
                drop_path=dpr[block_idx],  # Dropout schedule
                block_idx=block_idx,
                last_drop_p=dpr[block_idx - 1] if block_idx > 0 else 0  # Avoid index error
            ) for block_idx in range(depth)
        ])

        # Output layer (predicts next node or adjacency)
        self.head = nn.Linear(embed_dim, num_nodes)


    def forward(self, data):
        """
        :param data: A PyG Data object containing x (nodes), edge_index, y (labels)
        :return: Logits predicting the next node
        """
        nodes = data.x  # Node features
        edge_index = data.edge_index  # Edge connections

        # Graph Encoding
        node_features = self.graph_encoder(nodes.long(), edge_index)

        # Autoregressive Transformer
        x = self.node_embed(nodes) + node_features  # Combine embeddings
        for block in self.blocks:
            x = block(x)  # Pass through transformer blocks

        # Predict Next Node
        return self.head(x)

    

    def autoregressive_infer_cfg(self, graph_data, steps=10):
        """
        Generates a graph in an autoregressive manner
        :param graph_data: Partial graph (to condition on)
        :return: Generated graph structure
        """
        nodes, edges, adj = graph_data

        for step in range(steps):
            logits = self.forward((nodes, edges, adj))
            next_node = torch.argmax(logits, dim=-1)  # Sample next node
            
            # Append new node to the graph
            nodes = torch.cat([nodes, next_node.unsqueeze(0)], dim=0)
            
            # Predict edges (if needed)
            edge_logits = self.head(edges)
            next_edge = torch.argmax(edge_logits, dim=-1)
            edges = torch.cat([edges, next_edge.unsqueeze(0)], dim=0)

        return nodes, edges
