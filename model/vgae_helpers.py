import torch
from torch.nn import functional as F
from typing import Tuple, List

from utils.losses import discretize


def prepare_for_exp(annots_recon, adjs_recon, node_masks, edge_masks):
    masks = node_masks.unsqueeze(-1)

    adjs_recon = (adjs_recon.transpose(1, 2) + adjs_recon) * 0.5

    adjs_recon = discretize(adjs_recon, masks=edge_masks)
    adjs_recon[:, :, :, 0] = adjs_recon[:, :, :, 0] + (1 - edge_masks.squeeze())
    adjs_recon = torch.cat((adjs_recon[:, :, :, 1:], torch.zeros(adjs_recon.shape[0], adjs_recon.shape[1], adjs_recon.shape[2], 1)), dim=-1)
    annots_recon = discretize(annots_recon, masks=masks)
    none_type = 1 - masks.float()
    annots_recon = torch.cat((annots_recon, none_type), dim=-1).detach().cpu()
    adjs_recon = adjs_recon.permute(0, 3, 1, 2).detach().cpu()

    return annots_recon, adjs_recon


def interpolate_batch(graphs: Tuple[torch.Tensor], to_sizes: List[int], padding_size: int = None):
    '''
    Interpolate all graphs to desired sizes.
    If sizes are different, add padding.
    '''
    # Interpolate all graphs to desired sizes
    interpolated_nodes = []
    for graph, size in zip(graphs, to_sizes):
        if len(graph.shape) < 3: graph = graph.unsqueeze(0)
        graph = graph.transpose(1, 2)
        graph = F.interpolate(graph, size=(size if isinstance(size, int) else size.item()), mode='linear')

        # Add padding if necessary
        if padding_size is not None:
            padded = torch.zeros(1, graph.size(1), padding_size, device=graph.device)
            padded[:, :, :size] = graph
            graph = padded

        interpolated_nodes.append(graph.transpose(1, 2).squeeze(0))
    
    return torch.stack(interpolated_nodes)

def sizes_to_mask(original_sizes, max_size, device):
    ''' Convert list of graph sizes to a binary mask. '''
    B = len(original_sizes)
    mask = torch.zeros(B, max_size, device=device)
    for i, size in enumerate(original_sizes):
        mask[i, :size] = 1
    return mask.bool()
