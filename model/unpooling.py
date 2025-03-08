from torch import nn
from torch_geometric.nn import TransformerConv


class TransformerGraphDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = TransformerConv(in_channels, out_channels, heads=4)

    def forward(self, x, edge_index):
        return self.conv1(x, edge_index)
