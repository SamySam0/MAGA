import torch
from torch.nn import functional as F
from typing import Tuple, List

from utils.losses import discretize


def prepare_for_exp(annots_recon, adjs_recon, node_masks, edge_masks):
    masks = node_masks.unsqueeze(-1)

    adjs_recon = (adjs_recon.transpose(1, 2) + adjs_recon) * 0.5

    adjs_recon = discretize(adjs_recon, masks=edge_masks)
    adjs_recon[:, :, :, 0] = adjs_recon[:, :, :, 0] + (1 - edge_masks.squeeze())
    adjs_recon = torch.cat((adjs_recon[:, :, :, 1:], torch.zeros(adjs_recon.shape[0], adjs_recon.shape[1], adjs_recon.shape[2], 1, device=adjs_recon.device)), dim=-1)
    annots_recon = discretize(annots_recon, masks=masks)
    none_type = 1 - masks.float()
    annots_recon = torch.cat((annots_recon, none_type), dim=-1).detach().cpu()
    adjs_recon = adjs_recon.permute(0, 3, 1, 2).detach().cpu()

    return annots_recon, adjs_recon


def interpolate_batch(graphs: Tuple[torch.Tensor], to_sizes: List[int], padding_size: int):
    '''
    Interpolate all graphs to desired sizes and add padding.
    Returns interpolated nodes and their masks.
    '''
    interpolated_nodes = []
    masks = []
    for graph, size in zip(graphs, to_sizes):
        graph = F.interpolate(graph.transpose(1, 2), size=(size), mode='linear')

        # Create mask for the interpolated nodes
        mask = torch.ones(1, size, device=graph.device)

        # Add padding if necessary
        padded = torch.zeros(1, graph.size(1), padding_size, device=graph.device)
        padded[:, :, :size] = graph
        graph = padded

        # Adjust mask for padding
        padded_mask = torch.zeros(1, padding_size, device=graph.device)
        padded_mask[:, :size] = mask
        mask = padded_mask

        interpolated_nodes.append(graph.transpose(1, 2).squeeze(0))
        masks.append(mask.squeeze(0))
    
    return torch.stack(interpolated_nodes), torch.stack(masks).bool()

def sizes_to_mask(original_sizes, max_size, device):
    ''' Convert list of graph sizes to a binary mask. '''
    B = len(original_sizes)
    mask = torch.zeros(B, max_size, device=device)
    for i, size in enumerate(original_sizes):
        mask[i, :size] = 1
    return mask.bool()
