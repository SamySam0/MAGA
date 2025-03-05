import torch
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
    adjs_recon = adjs_recon[:, :, :, 1:]
    adjs_recon = adjs_recon.permute(0, 3, 1, 2).detach().cpu()

    return annots_recon, adjs_recon
