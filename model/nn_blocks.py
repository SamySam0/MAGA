import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter


class MPNNLayer(MessagePassing):
    def __init__(self, node_dim, edge_dim, hidden_dim, node_emb_dim, edge_emb_dim, aggr='add'):
        super().__init__(aggr=aggr)
        self.in_node_dim = node_dim
        self.node_emb_dim = node_emb_dim

        self.mlp_node_msg = nn.Sequential(
            nn.Linear(2*node_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_emb_dim),
        )
        self.node_norm = nn.BatchNorm1d(node_emb_dim)

        self.mlp_edge_msg = nn.Sequential(
            nn.Linear(2*node_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, edge_emb_dim),
        )
        self.edge_norm = nn.BatchNorm1d(edge_emb_dim)
    
    def forward(self, x, edge_index, edge_attr):
        node_out, edge_out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return node_out, edge_out
    
    def message(self, x_i, x_j, edge_attr):
        node_msg = torch.cat([x_i, x_j, edge_attr], dim=-1)
        edge_msg = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.mlp_node_msg(node_msg), self.mlp_edge_msg(edge_msg)
    
    def aggregate(self, inputs, index):
        node_inputs, edge_inputs = inputs
        return scatter(node_inputs, index, dim=self.node_dim, reduce=self.aggr), edge_inputs
    
    def update(self, aggr_out, x):
        node_aggr_out, edge_aggr_out = aggr_out
        if self.in_node_dim == self.node_emb_dim:
            node_upd = self.node_norm(x + node_aggr_out)
        else:
            node_upd = self.node_norm(node_aggr_out)
        edge_upd = self.edge_norm(edge_aggr_out)
        return node_upd, edge_upd

class GNNLayer(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, node_emb_dim, edge_emb_dim):
        super().__init__()

        self.node_linear = nn.Sequential(
            nn.Linear(2*node_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_emb_dim),
        )
        self.node_norm = nn.BatchNorm1d(node_emb_dim)

        self.edge_linear = nn.Sequential(
            nn.Linear(2*node_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, edge_emb_dim),
        )
        self.edge_norm = nn.BatchNorm2d(edge_emb_dim)
    
    def forward(self, node_feat, edge_feat=None, skip_connection=False):
        nodes2nodes = self.nodes2edges(node_feat)
        edge_feat = torch.cat([nodes2nodes, edge_feat], dim=-1) if edge_feat is not None else nodes2nodes

        edge_out = self.edge_linear(edge_feat)
        node_out = self.node_linear(edge_feat).sum(-2)
        if skip_connection:
            node_out = node_feat + node_out

        edge_out = self.edge_norm(edge_out.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        node_out = self.node_norm(node_out.permute(0, 2, 1)).permute(0, 2, 1)
        return node_out, edge_out
    
    def nodes2edges(self, nodes):
        nodes_ = nodes.unsqueeze(3).permute(0, 3, 1, 2)
        nodes_ = nodes_.repeat(1, nodes.shape[1], 1, 1)
        nodesT = nodes_.transpose(1, 2)
        return torch.cat([nodes_, nodesT], dim=3)
