from torch import nn
from model.vgae_blocks import MPNNLayer, GNNLayer, AttnLatentProj
from torch_geometric.nn import SAGPooling


class Unpooling(nn.Module):
    def __init__(self, node_dim, pooling_to_size):
        super().__init__()
        self.pool = SAGPooling(node_dim, ratio=int(pooling_to_size))
    
    def forward(self, node_feat, edge_index, batch):
        node_feat, _, _, batch_idx, _, _ = self.pool(
            node_feat, edge_index, batch=batch,
        )
        return node_feat, batch_idx