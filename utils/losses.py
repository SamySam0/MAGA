import torch
from torch_geometric.utils import to_dense_adj, to_dense_batch


def get_losses(batch, loss_fn, node_logits, edge_logits, node_masks, n_node_feat):
    node_loss = get_node_loss(batch, node_logits, node_masks, loss_fn, n_node_feat)

    edge_masks = get_edge_masks(node_masks)
    edge_loss = get_edge_loss(batch, edge_logits, edge_masks, loss_fn)
    
    # Calculate the total loss
    tot = node_masks.sum() + edge_masks.sum()
    recon_loss = (edge_masks.sum() / tot) * edge_loss + (node_masks.sum() / tot) * node_loss
    return recon_loss, (node_loss, edge_loss)


def get_node_loss(batch, node_logits, node_masks, loss_fn, n_node_feat):
    target, _ = to_dense_batch(batch.x, batch.batch)
    target = target[:, :, :n_node_feat].argmax(-1)

    preds = node_logits * node_masks.unsqueeze(-1)

    loss = loss_fn(input=preds.permute(0,2,1).contiguous(), target=target)
    loss = loss.mean()
    return loss

def get_edge_loss(batch, edge_logits, edge_masks, loss_fn):
    # Make the adj matrix symmetric
    edge_logits = (edge_logits.transpose(1, 2) + edge_logits) / 2

    # Get edge target matrix 
    target = to_dense_adj(batch.edge_index, batch=batch.batch, edge_attr=batch.edge_attr)
    no_edge = 1 - target.sum(-1, keepdim=True)
    target = torch.cat((no_edge, target), dim=-1)
    target = target.argmax(-1)

    loss = loss_fn(input=edge_logits.permute(0, 3, 1, 2).contiguous(), target=target)
    loss = loss * edge_masks.squeeze()
    loss = loss.mean()
    return loss


def get_edge_masks(node_masks):
    batch_size, n = node_masks.shape
    mask_reversed = (1 - node_masks.float())
    mask_reversed = mask_reversed.reshape(batch_size, -1, 1) + mask_reversed.reshape(batch_size, 1, -1)
    mask_reversed = mask_reversed + torch.eye(n).to(node_masks.device)
    mask_reversed = (mask_reversed>0).float()
    masks = 1-mask_reversed
    return masks.unsqueeze(-1)
