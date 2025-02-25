import torch
import torch.nn.functional as F
from torch_geometric.utils import dense_to_sparse, to_dense_adj, to_dense_batch
from utils.func import discretize, get_edge_target


def get_losses(batch, nodes_rec, edges_rec, n_node_feat, node_masks, edge_masks):
    node_loss, nodes_rec, nodes_target = get_node_loss_and_recon(
        batch, nodes_rec, node_masks, n_node_feat,
    )
    edge_loss, edges_rec = get_edge_loss_and_recon(batch, edges_rec, edge_masks)
    
    # Calculate the total loss
    tot = node_masks.sum() + edge_masks.sum()
    recon_loss = (edge_masks.sum() / tot) * edge_loss + (node_masks.sum() / tot) * node_loss
    return recon_loss, (node_loss, edge_loss, nodes_rec, edges_rec)


def get_node_loss_and_recon(batch, nodes_rec, node_masks, n_node_feat):
    '''
    Return the loss for the node features and the reconstructed and discretized instance
    '''
    dense_nodes, _ = to_dense_batch(batch.x, batch.batch)

    target = dense_nodes[:, :, :n_node_feat].argmax(-1)

    nodes_rec = nodes_rec.log_softmax(-1) * node_masks

    node_loss = F.nll_loss(nodes_rec.permute(0, 2, 1).contiguous(), target, reduction='none')
    node_loss = node_loss.mean()
    nodes_rec = discretize(nodes_rec, node_masks)
    none_type = 1-node_masks.float()
    nodes_rec = torch.cat((nodes_rec[:, :, :n_node_feat], none_type), dim=-1)
    return node_loss, nodes_rec, target

def get_edge_loss_and_recon(batch, edges_rec, edge_masks):
    '''
    Return the loss for the edge features and the reconstructed and discretized instance
    '''
    edges_rec = (edges_rec.transpose(1, 2) + edges_rec) * .5
    edges_rec = edges_rec.log_softmax(-1)
    edge_target = get_edge_target(batch)

    edge_loss = F.nll_loss(edges_rec.permute(0, 3, 1, 2).contiguous(), edge_target, reduction='none')
    edge_loss = edge_loss * edge_masks.squeeze()
    edge_loss = edge_loss.mean()
    edges_rec = discretize(edges_rec, edge_masks)
    edges_rec[:, :, :, 0] = edges_rec[:, :, :, 0] + (1-edge_masks.squeeze())
    return edge_loss, edges_rec


def discretize(score, masks=None):
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
