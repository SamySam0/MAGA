import torch
from torch import nn
import numpy as np


class VAR_Trainer(object):
    def __init__(
        self, vqvgae, var, var_optimizer, var_scheduler, dataloaders, device, L, grad_clip, label_smooth,
    ):
        super(VAR_Trainer, self).__init__()

        self.var, self.vqvgae, self.quantizer_local = var, vqvgae, vqvgae.quantizer
        self.var_optimizer = var_optimizer
        self.var_scheduler = var_scheduler
        self.train_loader, self.valid_loader = dataloaders
        self.device = device
        
        self.train_loss = nn.CrossEntropyLoss(label_smoothing=label_smooth, reduction='none')
        self.loss_weight = torch.ones(1, L, device=device) / L
        self.grad_clip = grad_clip

    def train_step(self, batch, label_B):
        # Zero gradients at the start of each step
        self.var_optimizer.zero_grad()

        # Forward pass
        B, V = label_B.shape[0], self.vqvgae.quantizer.codebook_size

        # Get VAR input (x_BLCv_wo_first_l) and ground truth (gt_BL)
        gt_idx_Bl: List = self.vqvgae.graph_to_idxBl(batch)
        x_BLCv_wo_first_l = self.quantizer_local.idxBl_to_var_input(gt_idx_Bl)
        gt_BL = torch.cat(gt_idx_Bl, dim=1)
        
        logits_BLV = self.var(x_BLCv_wo_first_l, label_B) # B, L, V
        loss = self.train_loss(logits_BLV.view(-1, V), gt_BL.view(-1)).view(B, -1)
        loss = loss.mul(self.loss_weight).sum(dim=-1).mean()
        
        # Backward pass
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.var.parameters(), self.grad_clip)
        self.var_optimizer.step()

        return loss.item(), grad_norm

    def train_ep(self):
        self.var.train()
        ep_train_loss = []
        for it, batch in enumerate(self.train_loader):
            label = batch.batch.bincount()
            it_loss, it_grad_norm = self.train_step(batch=batch.to(self.device), label_B=label.to(self.device))
            ep_train_loss.append(it_loss)
        return np.mean(ep_train_loss)
    
    def eval_step(self, batch, label_B):
        # Forward pass
        B, V = label_B.shape[0], self.vqvgae.quantizer.codebook_size

        # Get VAR input (x_BLCv_wo_first_l) and ground truth (gt_BL)
        gt_idx_Bl: List = self.vqvgae.graph_to_idxBl(batch)
        x_BLCv_wo_first_l = self.quantizer_local.idxBl_to_var_input(gt_idx_Bl)
        gt_BL = torch.cat(gt_idx_Bl, dim=1)
        
        logits_BLV = self.var(x_BLCv_wo_first_l, label_B) # B, L, V
        L_mean = self.valid_loss(logits_BLV.view(-1, V), gt_BL.view(-1))
        L_tail = self.valid_loss(logits_BLV[:, -self.last_l:].reshape(-1, V), gt_BL[:, -self.last_l:].reshape(-1))
        acc_mean = (logits_BLV.argmax(dim=-1) == gt_BL).sum() * (100/gt_BL.shape[1])
        acc_tail = (logits_BLV[:, -self.last_l:].argmax(dim=-1) == gt_BL[:, -self.last_l:]).sum() * (100 / self.last_l)

        return L_mean.item(), L_tail.item(), acc_mean.item()/B, acc_tail.item()/B

    def eval_ep(self):
        self.var.eval()
        ep_eval_mean_loss, ep_eval_tail_loss = [], []
        ep_eval_mean_acc, ep_eval_tail_acc = [], []
        with torch.no_grad():
            for it, batch in enumerate(self.valid_loader):
                label = batch.batch.bincount()
                L_mean, L_tail, acc_mean, acc_tail = self.eval_step(bathc=batch.to(self.device), label_B=label.to(self.device))
                ep_eval_mean_loss.append(L_mean)
                ep_eval_tail_loss.append(L_tail)
                ep_eval_mean_acc.append(acc_mean)
                ep_eval_tail_acc.append(acc_tail)
        
        self.var_scheduler.step(np.mean(ep_eval_tail_loss))
        return np.mean(ep_eval_mean_loss), np.mean(ep_eval_tail_loss), np.mean(ep_eval_mean_acc), np.mean(ep_eval_tail_acc)
