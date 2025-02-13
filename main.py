from data.dataset import load_qm9_data
from model.vqvae import VQVAE
from easydict import EasyDict as edict
from train.trainer import Trainer
from utils.transforms import AddSpectralFeat
import time
import yaml
import torch


def main():
    # Load in config
    config_path = f'config.yaml'
    config = edict(yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader))
    
    # === Data parameters ===
    data_root = config.data.path
    batch_size = config.train.batch_size
    num_workers = 0
    train_val_test_split = config.data.train_val_test_split
    dataset_size = config.data.dataset_size
    
    # === Training parameters ===
    gamma = config.train.gamma
    n_epochs = config.train.epochs
    
    # === Model parameters ===
    # model_config = config.get("model", {})  # This dictionary can include any model-specific parameters

    # Load the QM9 data using your helper function.
    train_loader, val_loader, test_loader = load_qm9_data(
        transforms=[AddSpectralFeat()],
        root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        train_val_test_split=train_val_test_split,
        dataset_size=dataset_size
    )
    
    # Instantiate the model. (Make sure that your VQVAE __init__ accepts the parameters in model_config.)
    # model = VQVAE(config=config)
    
    # Optionally, print some information.
    print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Training for {n_epochs} epochs with batch size {batch_size}...")
    
    # --- Timing the training ---
    start_time = time.time()
    
    # Run the experiment (training, validation, and testing).
    # (We add a small change: pass model_name to run_experiment so it can be logged per epoch.)
    # best_val_error, test_error, perf_per_epoch = run_experiment(
    #     model, train_loader, val_loader, test_loader, gamma, n_epochs)

    Trainer(dataloaders=(train_loader, val_loader), config=config).train()
    
    total_time = (time.time() - start_time) / 60.0  # total time in minutes

    # Print final results.
    print("=====================================")
    print(f"Training completed in {total_time:.2f} minutes")
    print(f"Best validation error: {best_val_error:.7f}")
    print(f"Test error corresponding to best validation: {test_error:.7f}")
    print("=====================================")


if __name__ == "__main__":
    main()
