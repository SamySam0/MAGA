import torch
import os
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, dense_to_sparse, to_dense_batch
import torch.optim as optim



def discretize(score, masks=None):
    '''
    Args:
        score (tensor: batch size x # nodes x ... x #node types): A tensor containing the (log)
        probabilities (normalized or not) of each type on the last dim.
        Masks (tensor: bool, size as score), with True where there is a node/edge and False if
        the is no nodes/edges as this position.
        '''
    argmax = score.argmax(-1)
    device = score.device
    rec = torch.eye(score.shape[-1]).to(device)[argmax]
    if masks is not None:
        rec = rec * masks
    return rec


def get_edge_target(batch):
    dense_edge_attr = to_dense_adj(batch.edge_index, batch=batch.batch, edge_attr=batch.edge_attr)
    if len(dense_edge_attr.shape) == 3:
        return dense_edge_attr
    else:
        no_edge = 1 - dense_edge_attr.sum(-1, keepdim=True)
        dense_edge_attr = torch.cat((no_edge, dense_edge_attr), dim=-1)
        return dense_edge_attr.argmax(-1)

def get_edge_masks(node_masks):
    device = node_masks.device
    n = node_masks.shape[1]
    batch_size = node_masks.shape[0]
    mask_reversed = (1 - node_masks.float())
    mask_reversed = mask_reversed.reshape(batch_size, -1, 1) + mask_reversed.reshape(batch_size, 1, -1)
    mask_reversed = mask_reversed + torch.eye(n).to(device)
    mask_reversed = (mask_reversed>0).float()
    masks = 1-mask_reversed
    return masks.unsqueeze(-1)