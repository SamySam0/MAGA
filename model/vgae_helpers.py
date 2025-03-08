import torch
from utils.losses import get_edge_masks


def prepare_for_exp(annots_recon, adjs_recon, node_masks):
    # Make the adj matrix symmetric
    adjs_recon = (adjs_recon.transpose(1, 2) + adjs_recon) / 2

    # Discretize adj and remove first feature ("no edge") since qm9_exp takes 4 features where all should be 0 when no edge
    edge_masks = get_edge_masks(node_masks)
    adjs_recon = discretize(adjs_recon, masks=edge_masks)       # Set to zero edges of masked nodes
    adjs_recon[:, :, :, 0] = adjs_recon[:, :, :, 0] + (1 - edge_masks.squeeze())
    adjs_recon = adjs_recon.permute(0, 3, 1, 2).detach().cpu()

    # Discretizer nodes
    annots_recon = discretize(annots_recon, masks=node_masks.unsqueeze(-1))   # Set to zero masked nodes (others are always nodes since interpolate to desired graph size already)
    none_type = 1 - node_masks.unsqueeze(-1).float()
    annots_recon = torch.cat((annots_recon, none_type), dim=-1).detach().cpu()

    return annots_recon, adjs_recon


def discretize(score, masks):
    argmax = score.argmax(-1)
    rec = torch.eye(score.shape[-1]).to(score.device)[argmax]
    if masks is not None:
        rec = rec * masks
    return rec
