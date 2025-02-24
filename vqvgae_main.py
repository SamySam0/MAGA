import torch, yaml, os
from datetime import datetime
from data.dataset import load_qm9_data
from model import build_vqvgae
import torch.optim as optim
from easydict import EasyDict as edict
from utils.transforms import AddSpectralFeat
from train.vqvgae_train import train


def main(config_path='config.yaml'):
    # Load config
    torch.manual_seed(42)
    config = edict(yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader))

    # Create log dir if does not exist
    os.makedirs(config.log.checkpoint_dir, exist_ok=True)

    # Save config in log file
    checkpoint_name = f"VQ-VGAE_{datetime.now().strftime('%m_%d_%H_%M')}"
    print(f'Run name: {checkpoint_name}\n')
    with open(f'{config.log.checkpoint_dir}/{checkpoint_name}_logs.txt', 'w') as f:
        f.write('===== Config =====\n')
        for section, params in config.items():
            f.write(f'\n[{section}]\n')
            if isinstance(params, dict):
                for key, value in params.items():
                    f.write(f'{key}: {value}\n')
            else:
                f.write(f'{params}\n')

    # Load QM9 dataset
    train_loader, val_loader, test_loader = load_qm9_data(
        transforms=[AddSpectralFeat()],
        root=config.data.path,
        batch_size=config.vqvgae.train.batch_size,
        num_workers=2,
        train_val_test_split=config.data.train_val_test_split,
        dataset_size=config.data.dataset_size,
    )

    # Initialise model, optimizer and scheduler
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_vqvgae(config, device, vqvgae_pretrain_path=None)
    optimizer = optim.Adam(model.parameters(), lr=config.vqvgae.train.lr, betas=(config.vqvgae.train.beta1, config.vqvgae.train.beta2))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=config.vqvgae.train.lr_decay, patience=config.vqvgae.train.sch_patience, min_lr=2*1e-5)

    # Train model
    train(
        model=model, optimizer=optimizer, scheduler=scheduler,
        train_loader=train_loader, valid_loader=val_loader,
        device=device, train_gamma=config.vqvgae.train.gamma, 
        n_epochs=config.vqvgae.train.epochs, log_loss_per_n_epoch=config.log.log_loss_per_n_epoch,
        checkpoint_path=config.log.checkpoint_dir, checkpoint_name=checkpoint_name,
    )


if __name__ == '__main__':
    main()
