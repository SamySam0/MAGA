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
    B, label_B = 4, torch.tensor([3, 5, 7, 9])
    output = var_model.autoregressive_infer_cfg(B=B, label_B=label_B, cfg=1.5, top_k=0, top_p=0)
    print(output)


if __name__ == '__main__':
    main()
