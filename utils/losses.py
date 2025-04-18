import torch
import torch.nn.functional as F
from torch_geometric.utils import dense_to_sparse, to_dense_adj, to_dense_batch
from utils.func import discretize, get_edge_target



def get_losses(batch, nodes_rec, edges_rec, node_masks, annotated_nodes,
               annotated_edges, max_node_num, n_node_feat, edge_masks):
    if annotated_nodes:
        node_loss, nodes_rec, nodes_target = get_node_loss_and_recon(batch, nodes_rec, node_masks,
                                                                     max_node_num, n_node_feat)

    if annotated_edges:
        edge_loss, edges_rec = get_edge_loss_and_recon(batch, edges_rec, edge_masks)
    else:
        edge_loss, edges_rec = get_unannotated_loss_and_rec(batch, edges_rec, edge_masks, max_node_num)

    # Calculate the total loss
    if annotated_nodes:
        tot = node_masks.sum() + edge_masks.sum()
        recon_loss = (edge_masks.sum() / tot) * edge_loss + (node_masks.sum() / tot) * node_loss
    else:
        recon_loss = edge_loss
        nodes_rec = None
        node_loss = None
    return recon_loss, (node_loss, edge_loss, nodes_rec, edges_rec)



def get_node_loss_and_recon(batch, nodes_rec, node_masks, max_node_num, n_node_feat):
    '''
    Return the loss for the node features and the reconstructed and discretized instance
    '''
    dense_nodes, _ = to_dense_batch(batch.x, batch.batch)
    # print('THIS IS THE GET NODE LOSS FUNCTION')
    # print(f'dense_nodes: {dense_nodes.shape}')
    target = dense_nodes[:, :, :n_node_feat].argmax(-1)
    # print(f'-----Target Shape: {target.shape}')
    # print(f'-----node_masks Shape: {node_masks.shape}')
    nodes_rec = nodes_rec.log_softmax(-1) * node_masks
    # print(f'nodes_rec: {nodes_rec.shape}')
    # print(f'-----nodes_rec.permute(0, 2, 1): {nodes_rec.permute(0, 2, 1).contiguous().shape}')
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
    # print('THIS IS THE GET EDGE LOSS FUNCTION')

    edges_rec = (edges_rec.transpose(1, 2) + edges_rec) * .5
    edges_rec = edges_rec.log_softmax(-1)
    edge_target = get_edge_target(batch)


    edge_loss = F.nll_loss(edges_rec.permute(0, 3, 1, 2).contiguous(), edge_target, reduction='none')
    edge_loss = edge_loss * edge_masks.squeeze()
    edge_loss = edge_loss.mean()
    edges_rec = discretize(edges_rec, edge_masks)
    edges_rec[:, :, :, 0] = edges_rec[:, :, :, 0] + (1-edge_masks.squeeze())
    return edge_loss, edges_rec


def get_unannotated_loss_and_rec(batch, edges_rec, edge_masks, max_node_num):
    '''
    Return the loss for the edges and the reconstructed and discretized instance
    '''
    edges_rec = (edges_rec.transpose(1, 2) + edges_rec) * .5
    adjs = to_dense_adj(batch.edge_index, batch=batch.batch, max_num_nodes=max_node_num)
    edges_rec = edges_rec.sigmoid()
    edges_rec = edges_rec * edge_masks
    edge_loss = F.binary_cross_entropy(edges_rec.flatten(1), adjs.flatten(1), reduction='mean')
    edges_rec = edges_rec.round()
    return edge_loss, edges_rec