import torch
from torch import nn

from model.nn_blocks import MPNNLayer, GNNLayer


class Encoder(nn.Module):
    def __init__(self, 
        n_layers, hidden_dim, emb_dim,
        in_node_feature_dim, in_edge_feature_dim, 
        out_node_feature_dim,
    ):
        super().__init__()
        layers = [MPNNLayer(node_dim=in_node_feature_dim, edge_dim=in_edge_feature_dim, hidden_dim=hidden_dim, node_emb_dim=emb_dim, edge_emb_dim=emb_dim)]
        for _ in range(1, n_layers-1):
            layers.append(MPNNLayer(node_dim=emb_dim, edge_dim=emb_dim, hidden_dim=hidden_dim, node_emb_dim=emb_dim, edge_emb_dim=emb_dim))
        layers.append(MPNNLayer(node_dim=emb_dim, edge_dim=emb_dim, hidden_dim=hidden_dim, node_emb_dim=out_node_feature_dim, edge_emb_dim=1))
        self.layers = nn.Sequential(*layers)
    
    def forward(self, batch):
        node_feat, edge_feat = self.layers[0](batch.x, batch.edge_index, batch.edge_attr)
        for i, layer in enumerate(self.layers[1:-1]):
            node_feat_new, edge_feat_new = layer(node_feat, batch.edge_index, edge_feat)
            node_feat = node_feat + node_feat_new
            edge_feat = edge_feat + edge_feat_new
        node_feat, edge_feat = self.layers[-1](node_feat, batch.edge_index, edge_feat)
        return node_feat, edge_feat


class Decoder(nn.Module):
    def __init__(self, 
        n_layers, hidden_dim, emb_dim,
        in_node_feature_dim,
        out_node_feature_dim, out_edge_feature_dim,
    ):
        super().__init__()
        layers = [GNNLayer(node_dim=in_node_feature_dim, edge_dim=0, hidden_dim=hidden_dim, node_emb_dim=emb_dim, edge_emb_dim=emb_dim)]
        for _ in range(1, n_layers-1):
            layers.append(GNNLayer(node_dim=emb_dim, edge_dim=emb_dim, hidden_dim=hidden_dim, node_emb_dim=emb_dim, edge_emb_dim=emb_dim))
        layers.append(GNNLayer(node_dim=emb_dim, edge_dim=emb_dim, hidden_dim=hidden_dim, node_emb_dim=out_node_feature_dim, edge_emb_dim=out_edge_feature_dim))
        self.layers = nn.Sequential(*layers)
    
    def forward(self, node_feat, mask=None):
        node_feat, edge_feat = self.layers[0](node_feat)
        for i, layer in enumerate(self.layers[1:-1]):
            node_feat_new, edge_feat_new = layer(node_feat, edge_feat, skip_connection=True)
            node_feat = node_feat + node_feat_new
            edge_feat = edge_feat + edge_feat_new
        node_feat, edge_feat = self.layers[-1](node_feat, edge_feat)
        if mask is not None:
            node_feat = node_feat * mask.unsqueeze(-1)
            edge_feat = edge_feat * mask.reshape(node_feat.shape[0], -1, 1, 1)
            edge_feat = edge_feat * mask.reshape(node_feat.shape[0], 1, -1, 1)
        return node_feat, edge_feat
