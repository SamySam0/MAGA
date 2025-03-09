import torch, yaml, os
from torch import nn
from datetime import datetime
from data.dataset import get_dataset
from model import build_vqvgae
import torch.optim as optim
from easydict import EasyDict as edict
from utils.transforms import AddSpectralFeat
from train.vqvgae_train import train


def main(config_path='config.yaml'):
    # Load config
    torch.manual_seed(42)
    config = edict(yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader))

    # Fetch appropriate data configurations
    if config.dataset.name == 'qm9':
        config.dataset.update(config.qm9)
    else:
        config.dataset.update(config.zinc)

    # Create log dir if does not exist
    os.makedirs(config.log.checkpoint_dir, exist_ok=True)

    # Save config in log file
    checkpoint_name = f"VQ-VGAE_{datetime.now().strftime('%m_%d_%H_%M')}"
    print(f'Run name: {checkpoint_name}\n')
    with open(f'{config.log.checkpoint_dir}/{checkpoint_name}_logs.txt', 'w') as f:
        f.write('===== Config =====\n')
        for section, params in config.items():
            if section in ['qm9', 'zinc']:
                continue
            f.write(f'\n[{section}]\n')
            if isinstance(params, dict):
                for key, value in params.items():
                    f.write(f'{key}: {value}\n')
            else:
                f.write(f'{params}\n')

    # Load QM9 dataset
    train_loader, val_loader = get_dataset(
        root_dir=config.dataset.path, 
        dataset_name=config.dataset.name, 
        debug=config.dataset.debug, 
        batch_size=config.train.vqvgae.batch_size, 
        transforms=[AddSpectralFeat()],
    )

    # Initialise model, optimizer and scheduler
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_vqvgae(config, device, vqvgae_pretrain_path=None)
    optimizer = optim.Adam(model.parameters(), lr=config.train.vqvgae.lr, betas=(config.train.vqvgae.beta1, config.train.vqvgae.beta2))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=config.train.vqvgae.lr_decay, patience=config.train.vqvgae.sch_patience, min_lr=2*1e-5)
    loss_fn = nn.CrossEntropyLoss(reduction='none')

    # Train model
    train(
        model=model, optimizer=optimizer, scheduler=scheduler, loss_fn=loss_fn,
        train_loader=train_loader, valid_loader=val_loader,
        n_exp_samples=config.dataset.n_exp_samples, dataset_name=config.dataset.name,
        device=device, train_gamma=config.train.vqvgae.gamma, 
        n_epochs=config.train.vqvgae.epochs, log_loss_per_n_epoch=config.log.log_loss_per_n_epoch,
        checkpoint_path=config.log.checkpoint_dir, checkpoint_name=checkpoint_name,
    )


if __name__ == '__main__':
    main()
