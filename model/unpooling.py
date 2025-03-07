from torch import nn
from model.vgae_blocks import MPNNLayer, GNNLayer, AttnLatentProj
from torch_geometric.nn import TransformerConv

class TransformerGraphDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = TransformerConv(in_channels, out_channels, heads=4)

    def forward(self, x, edge_index):
        return self.conv1(x, edge_index)

# decoder = TransformerGraphDecoder(16, 16)
# x_unpooled = decoder(x_pooled, edge_index_pooled)


# import torch
# import torch.nn.functional as F
# from torch_geometric.nn import GCNConv

# class GraphDecoder(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels):
#         super(GraphDecoder, self).__init__()
#         self.conv1 = GCNConv(in_channels, hidden_channels)
#         self.conv2 = GCNConv(hidden_channels, out_channels)

#     def forward(self, x, edge_index):
#         x = F.relu(self.conv1(x, edge_index))
#         x = self.conv2(x, edge_index)
#         return x

# # Example usage
# decoder = GraphDecoder(in_channels=16, hidden_channels=32, out_channels=16)
# x_unpooled = decoder(x_pooled, edge_index_pooled)
