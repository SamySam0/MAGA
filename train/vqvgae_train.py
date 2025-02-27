import time, torch
from train.vqvgae_trainer import VQVGAE_Trainer


def train(
    model, optimizer, scheduler, train_loader, valid_loader, 
    device, train_gamma, n_epochs, log_loss_per_n_epoch,
    checkpoint_path, checkpoint_name,
):
    # Setup trainer
    trainer = VQVGAE_Trainer(
        model=model, optimizer=optimizer, scheduler=scheduler, 
        dataloaders=(train_loader, valid_loader), 
        device=device, gamma=train_gamma,
    )

    # Training phase 1 and 2: codebook initialisation
    training_times = []
    if trainer.model.quantizer.init_steps > 0 or trainer.model.quantizer.collect_phase:
        print('Starting Initialisation...')
        start_time = time.time()
        init_loss = round(trainer.init_codebook_training(), 5)
        training_times.append(time.time() - start_time)
        print("Initialisation terminated. Final epoch's partial loss:", init_loss, '\n')
    else:
        init_loss = None
    
    # Training phase 2: VQVAE training
    train_recon_errors, valid_recon_errors, lrs = [], [], []
    valid_s, unique_s, novel_s, valid_w_corr_s = [], [], [], []
    print('Starting Training...\n')
    for epoch in range(1, n_epochs+1):
        lrs.append(trainer.scheduler.optimizer.param_groups[0]['lr'])
        start_time = time.time()
        train_recon_loss = round(trainer.train_ep(), 5)
        training_times.append(time.time() - start_time)
        valid_recon_loss = round(trainer.valid_ep(), 5)
        valid, unique, novel, valid_w_corr = trainer.qm9_exp()
        valid, unique, novel, valid_w_corr = round(valid, 5), round(unique, 5), round(novel, 5), round(valid_w_corr, 5)
        if epoch % log_loss_per_n_epoch == 0 or epoch == n_epochs:
            print(f"Epoch: {epoch} | Train Recon Loss: {train_recon_loss}, Valid Recon Loss: {valid_recon_loss}, LR: {lrs[-1]}", end=' | ')
            print(f"Validity: {valid}, Unique: {unique}, Novel: {novel}, Valid w/ corr: {valid_w_corr}", end='')

        # Save model checkpoint if best validation loss so far
        if len(valid_recon_errors) == 0 or valid_recon_loss < min(valid_recon_errors):
            best_epoch = epoch
            print(' [NEW BEST]', end='')
            torch.save({
                'epoch': epoch,
                'model_state_dict': trainer.model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'scheduler_state_dict': trainer.scheduler.state_dict(),
                'train_recon_loss': train_recon_loss,
                'valid_recon_loss': valid_recon_loss,
                'exp_validity': valid,
                'exp_unique': unique,
                'exp_novel': novel,
                'exp_valid_w_corr': valid_w_corr,
                'cur_learning_rate': lrs[-1],
            }, f"{checkpoint_path}/{checkpoint_name}_best.pt")
        print()

        train_recon_errors.append(train_recon_loss)
        valid_recon_errors.append(valid_recon_loss)
        valid_s.append(valid)
        unique_s.append(unique)
        novel_s.append(novel)
        valid_w_corr_s.append(valid_w_corr)
    
    total_training_time = round(sum(training_times), 1)
    print(f"\nBEST EPOCH: {best_epoch} ==> Train Recon Loss: {train_recon_errors[best_epoch-1]}, Valid Recon Loss: {valid_recon_errors[best_epoch-1]}, LR: {lrs[best_epoch-1]}", end=', ')
    print(f"Validity: {valid_s[best_epoch-1]}, Unique: {unique_s[best_epoch-1]}, Novel: {novel_s[best_epoch-1]}, Valid w/ corr: {valid_w_corr_s[best_epoch-1]}")
    print(f"\nTraining time: {total_training_time} seconds")
    
    # Training logs
    with open(f'{checkpoint_path}/{checkpoint_name}_logs.txt', 'a') as f:
        # Init codebook loss
        f.write('\n\n===== Training logs =====')
        f.write(f"\nCodebook Initialisation Training Loss: {init_loss}\n")

        # Training losses
        for epoch in range(n_epochs):
            train_recon_loss, valid_recon_loss, lr = train_recon_errors[epoch], valid_recon_errors[epoch], lrs[epoch]
            valid, unique, novel, valid_w_corr = valid_s[epoch], unique_s[epoch], novel_s[epoch], valid_w_corr_s[epoch]
            f.write(f"\nEpoch {epoch+1} ==> Training recon error: {train_recon_loss} | Validation recon error: {valid_recon_loss} | LR: {lr} | Validity: {valid} | Unique: {unique} | Novel: {novel} | Valid w/ corr: {valid_w_corr}")

        # Best checkpoint infos
        f.write('\n\n===== Best checkpoint info =====')
        f.write(f"\nEpoch: {best_epoch}") 
        f.write(f"\nTRL: {train_recon_errors[best_epoch-1]}, VRL: {valid_recon_errors[best_epoch-1]}, LR: {lrs[best_epoch-1]} | ")
        f.write(f"Validity: {valid_s[best_epoch-1]}, Unique: {unique_s[best_epoch-1]}, Novel: {novel_s[best_epoch-1]}, Valid w/ corr: {valid_w_corr_s[best_epoch-1]}")
        f.write(f"\ncheckpoint_dir: {checkpoint_path}/{checkpoint_name}_best.pt")

        # Time logs
        f.write('\n\n===== Time logs =====')
        f.write(f"\nDevice: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        f.write(f"\nTraining time: {total_training_time} seconds")
