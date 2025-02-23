import yaml
from easydict import EasyDict as edict

from model.vqvgae import VQVAE
from model.var import VAR


def build_vqvgae_and_var(config, device, vqvgae_pretrain_path=None, var_pretrain_path=None):
    # Build (and load) VQ-VGAE
    vqvgae_model = build_vqvgae(config, device, vqvgae_pretrain_path)

    # Build (and load) VAR moodel
    var_model = VAR(vqvgae=vqvgae_model, config=config).to(device)
    if var_pretrain_path is not None:
        pass
    else:
        # var_model.init_weight()
        pass
    
    return vqvgae_model, var_model

def build_vqvgae(config, device, vqvgae_pretrain_path=None):
    vqvgae_model = VQVAE(config=config).to(device)
    if vqvgae_pretrain_path is not None:
        pass
    return vqvgae_model
