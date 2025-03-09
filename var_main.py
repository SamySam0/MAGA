import torch, yaml, os, argparse
from datetime import datetime
from data.dataset import get_dataset
from model import build_vqvgae_and_var
import torch.optim as optim
from easydict import EasyDict as edict
from utils.transforms import AddSpectralFeat
from train.var_train import train


def main(vqvgae_pretrain_path, config_path='config.yaml'):
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
    checkpoint_name = f"VAR_{datetime.now().strftime('%m_%d_%H_%M')}"
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
    vqvgae, var = build_vqvgae_and_var(config, device, vqvgae_pretrain_path=vqvgae_pretrain_path, var_pretrain_path=None)
    var_optimizer = optim.Adam(var.parameters(), lr=config.train.var.lr, betas=(config.train.var.beta1, config.train.var.beta2))
    var_scheduler = optim.lr_scheduler.ReduceLROnPlateau(var_optimizer, factor=config.train.var.lr_decay, patience=config.train.var.sch_patience, min_lr=2*1e-5)

    # Train model
    train(
        vqvgae=vqvgae, var=var, var_optimizer=var_optimizer, var_scheduler=var_scheduler, 
        train_loader=train_loader, valid_loader=val_loader, device=device,
        scales=config.vqvgae.quantizer.scales, grad_clip=config.train.var.grad_clip, label_smooth=config.train.var.label_smooth, 
        n_exp_samples=config.dataset.n_exp_samples, dataset_name=config.dataset.name,
        n_epochs=config.train.var.epochs, log_loss_per_n_epoch=config.log.log_loss_per_n_epoch,
        checkpoint_path=config.log.checkpoint_dir, checkpoint_name=checkpoint_name,
    )


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--vqvgae_pretrain_path', '-p', type=str, required=True,
                      help='Path to pretrained VQVGAE model')
    
    args = parser.parse_args()
    main(vqvgae_pretrain_path=args.vqvgae_pretrain_path)
