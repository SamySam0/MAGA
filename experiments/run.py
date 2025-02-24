import sys
import os
import torch
import yaml
from easydict import EasyDict as edict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model import build_vqvgae_and_var


def main():
    with open('config.yaml', 'r') as f:
        config = edict(yaml.safe_load(f))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, var_model = build_vqvgae_and_var(config=config, device=device)

    # Input parameters
    B, label_B = 3, torch.tensor([5, 7, 9])
    node_features, edge_features = var_model.autoregressive_infer_cfg(B=B, label_B=label_B, cfg=1.5, top_k=0, top_p=0)
    # print(node_features.shape)  # B x 32 x 11
    # print(edge_features.shape)  # B x 32 x 32 x 4


if __name__ == '__main__':
    main()
