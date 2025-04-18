import torch, yaml, os, argparse
from datetime import datetime
from data.dataset import load_qm9_data
from model import build_vqvgae_and_var
import torch.optim as optim
from easydict import EasyDict as edict
from utils.transforms import AddSpectralFeat
from train.var_train import train


def main(vqvgae_pretrain_path, config_path='config.yaml'):
    # Load config
    torch.manual_seed(42)
    config = edict(yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader))

    # Create log dir if does not exist
    os.makedirs(config.log.checkpoint_dir, exist_ok=True)

    # Save config in log file
    checkpoint_name = f"VAR_{datetime.now().strftime('%m_%d_%H_%M')}"
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
        batch_size=config.var.train.batch_size,
        num_workers=2,
        train_val_test_split=config.data.train_val_test_split,
        dataset_size=config.data.dataset_size,
    )

    # Initialise model, optimizer and scheduler
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vqvgae, var = build_vqvgae_and_var(config, device, vqvgae_pretrain_path=vqvgae_pretrain_path, var_pretrain_path=None)
    var_optimizer = optim.Adam(var.parameters(), lr=config.var.train.lr, betas=(config.var.train.beta1, config.var.train.beta2))
    var_scheduler = optim.lr_scheduler.ReduceLROnPlateau(var_optimizer, factor=config.var.train.lr_decay, patience=config.var.train.sch_patience, min_lr=2*1e-5)

    # Train model
    train(
        vqvgae=vqvgae, var=var, var_optimizer=var_optimizer, var_scheduler=var_scheduler, 
        train_loader=train_loader, valid_loader=val_loader, device=device,
        scales=config.vqvgae.quantizer.scales, grad_clip=config.var.train.grad_clip, label_smooth=config.var.train.label_smooth, n_exp_samples=config.data.n_exp_samples, 
        n_epochs=config.var.train.epochs, log_loss_per_n_epoch=config.log.log_loss_per_n_epoch,
        checkpoint_path=config.log.checkpoint_dir, checkpoint_name=checkpoint_name,
    )


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--vqvgae_pretrain_path', '-p', type=str, required=True,
                      help='Path to pretrained VQVGAE model')
    
    args = parser.parse_args()
    main(vqvgae_pretrain_path=args.vqvgae_pretrain_path)
