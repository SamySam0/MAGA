import time, torch
from train.var_trainer import VAR_Trainer


def train(
    vqvgae, var, var_optimizer, var_scheduler, train_loader, valid_loader, device,
    scales, grad_clip, label_smooth, n_epochs, exp_batch_size, log_loss_per_n_epoch, n_exp_samples,
    checkpoint_path, checkpoint_name,
):  
    # Setup trainer
    trainer = VAR_Trainer(
        vqvgae=vqvgae, var=var, var_optimizer=var_optimizer, var_scheduler=var_scheduler,
        dataloaders=(train_loader, valid_loader), device=device,
        L=sum(scales), last_l=scales[-1], grad_clip=grad_clip, label_smooth=label_smooth,
    )

    # Training phase
    training_times = []
    train_errors, lrs = [], []
    valid_s, unique_s, novel_s, valid_w_corr_s = [], [], [], []
    valid_errors = {'L_mean': [], 'L_tail': [], 'acc_mean': [], 'acc_tail': []}

    print('Starting Training...\n')
    for epoch in range(1, n_epochs+1):
        lrs.append(trainer.var_scheduler.optimizer.param_groups[0]['lr'])
        start_time = time.time()
        train_loss = round(trainer.train_ep(), 5)
        training_times.append(time.time() - start_time)
        val_mean_loss, val_tail_loss, val_mean_acc, val_tail_acc = trainer.eval_ep()
        val_mean_loss, val_tail_loss, val_mean_acc, val_tail_acc = round(val_mean_loss, 5), round(val_tail_loss, 5), round(val_mean_acc, 5), round(val_tail_acc, 5)

        valid, unique, novel, valid_w_corr = trainer.qm9_exp(n_samples=n_exp_samples, batch_size=exp_batch_size)
        valid, unique, novel, valid_w_corr = round(valid, 5), round(unique, 5), round(novel, 5), round(valid_w_corr, 5)

        if epoch % log_loss_per_n_epoch == 0 or epoch == n_epochs:
            print(f"Epoch: {epoch} | Train Loss: {train_loss}", end=' | ')
            print(f"Valid Mean Loss: {val_mean_loss}, Valid Tail Loss: {val_tail_loss}, Valid Mean Acc: {val_mean_acc}, Valid Tail Acc: {val_tail_acc}, LR: {lrs[-1]}")
        
        # Save model checkpoint if best validation loss so far
        if len(valid_errors['L_tail']) == 0 or val_tail_loss < min(valid_errors['L_tail']):
            best_epoch = epoch
            print(' [NEW BEST]', end='')
            torch.save({
                'epoch': epoch,
                'model_state_dict': trainer.var.state_dict(),
                'optimizer_state_dict': trainer.var_optimizer.state_dict(),
                'scheduler_state_dict': trainer.var_scheduler.state_dict(),
                'val_mean_loss': val_mean_loss, 
                'val_tail_loss': val_tail_loss,
                'val_mean_acc': val_mean_acc,
                'val_tail_acc': val_tail_acc,
                'train_loss': train_loss,
                'cur_learning_rate': lrs[-1],
            }, f"{checkpoint_path}/{checkpoint_name}_best.pt")
        
        print(f"\nExperiment => Valid: {valid}, Unique: {unique}, Novel: {novel}, Valid_w_corr: {valid_w_corr}")

        train_errors.append(train_loss)
        valid_errors['L_mean'].append(val_mean_loss)
        valid_errors['L_tail'].append(val_tail_loss)
        valid_errors['acc_mean'].append(val_mean_acc)
        valid_errors['acc_tail'].append(val_tail_acc)
        valid_s.append(valid)
        unique_s.append(unique)
        novel_s.append(novel)
        valid_w_corr_s.append(valid_w_corr)

    total_training_time = round(sum(training_times), 1)
    print(f"\nBEST EPOCH: {best_epoch} ==> Train Loss: {train_errors[best_epoch-1]}, Valid Mean Loss: {valid_errors['L_mean'][best_epoch-1]}, Valid Tail Loss: {valid_errors['L_tail'][best_epoch-1]}, Valid Mean Acc: {valid_errors['acc_mean'][best_epoch-1]}, Valid Tail Acc: {valid_errors['acc_tail'][best_epoch-1]}, LR: {lrs[best_epoch-1]}")
    print(f"Validity: {valid_s[best_epoch-1]}, Unique: {unique_s[best_epoch-1]}, Novel: {novel_s[best_epoch-1]}, Valid w/ corr: {valid_w_corr_s[best_epoch-1]}")
    print(f"\nTraining time: {total_training_time} seconds")

    # Training logs
    with open(f'{checkpoint_path}/{checkpoint_name}_logs.txt', 'a') as f:
        # Training losses
        f.write('\n\n===== Training logs =====')
        for epoch in range(n_epochs):
            train_loss, val_mean_loss, val_tail_loss, val_mean_acc, val_tail_acc, lr = train_errors[epoch], valid_errors['L_mean'][epoch], valid_errors['L_tail'][epoch], valid_errors['acc_mean'][epoch], valid_errors['acc_tail'][epoch], lrs[epoch]
            valid, unique, novel, valid_w_corr = valid_s[epoch], unique_s[epoch], novel_s[epoch], valid_w_corr_s[epoch]
            f.write(f"\nEpoch {epoch+1} ==> Train Loss: {train_loss}, Valid Mean Loss: {val_mean_loss}, Valid Tail Loss: {val_tail_loss}, Valid Mean Acc: {val_mean_acc}, Valid Tail Acc: {val_tail_acc}, LR: {lr} | ")
            f.write(f"Validity: {valid} | Unique: {unique} | Novel: {novel} | Valid w/ corr: {valid_w_corr}")

        # Best checkpoint infos
        f.write('\n\n===== Best checkpoint info =====')
        f.write(f"\nEpoch: {best_epoch}") 
        f.write(f"\nTL: {train_errors[best_epoch-1]}, VML: {valid_errors['L_mean'][best_epoch-1]}, VTL: {valid_errors['L_tail'][best_epoch-1]}, VMA: {valid_errors['acc_mean'][best_epoch-1]}, VTA: {valid_errors['acc_tail'][best_epoch-1]}, LR: {lrs[best_epoch-1]}, ")
        f.write(f"Validity: {valid_s[best_epoch-1]}, Unique: {unique_s[best_epoch-1]}, Novel: {novel_s[best_epoch-1]}, Valid w/ corr: {valid_w_corr_s[best_epoch-1]}")
        f.write(f"\ncheckpoint_dir: {checkpoint_path}/{checkpoint_name}_best.pt")

        # Time logs
        f.write('\n\n===== Time logs =====')
        f.write(f"\nDevice: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        f.write(f"\nTraining time: {total_training_time} seconds")
