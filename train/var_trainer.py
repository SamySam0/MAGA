import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from collections import Counter
from typing import List

from eval.evaluation import qm9_eval


class VAR_Trainer(object):
    def __init__(
        self, vqvgae, var, var_optimizer, var_scheduler, dataloaders, device, L, last_l, grad_clip, label_smooth,
    ):
        super(VAR_Trainer, self).__init__()

        self.var, self.vqvgae, self.quantizer_local = var, vqvgae, vqvgae.quantizer
        self.var_optimizer = var_optimizer
        self.var_scheduler = var_scheduler
        self.train_loader, self.valid_loader = dataloaders
        self.device = device
        
        self.train_loss = nn.CrossEntropyLoss(label_smoothing=label_smooth, reduction='none')
        self.valid_loss = nn.CrossEntropyLoss(label_smoothing=0.0, reduction='mean')
        self.loss_weight = torch.ones(1, L, device=device) / L
        self.grad_clip = grad_clip
        self.last_l = last_l

        self.pd_graph_size = CategoricalGraphSize(self.train_loader)

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
        for it, batch in tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc='Training', leave=False):
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
            for it, batch in tqdm(enumerate(self.valid_loader), total=len(self.valid_loader), desc='Evaluation (valid)', leave=False):
                label = batch.batch.bincount()
                L_mean, L_tail, acc_mean, acc_tail = self.eval_step(batch=batch.to(self.device), label_B=label.to(self.device))
                ep_eval_mean_loss.append(L_mean)
                ep_eval_tail_loss.append(L_tail)
                ep_eval_mean_acc.append(acc_mean)
                ep_eval_tail_acc.append(acc_tail)
        
        self.var_scheduler.step(np.mean(ep_eval_tail_loss))
        return np.mean(ep_eval_mean_loss), np.mean(ep_eval_tail_loss), np.mean(ep_eval_mean_acc), np.mean(ep_eval_tail_acc)

    def qm9_exp(self, n_samples, batch_size):
        self.var.eval()
        assert n_samples % batch_size == 0, f'n_samples ({n_samples}) must be divisible by the batch_size ({batch_size})!'
        valid_s, unique_s, novel_s, fcd_s, valid_w_corr_s = [], [], [], [], [] 
        with torch.no_grad():
            for batch in tqdm(range(n_samples//batch_size), desc='Experiment: Molecule Generation', leave=False):
                label = self.pd_graph_size.sample(batch_size).to(self.device)
                
                nodes_recon, edges_recon = self.var.autoregressive_infer_cfg(B=batch_size, label_B=label, cfg=1.5, top_k=0.0, top_p=0.0)
                oh_nodes_recon = F.one_hot(nodes_recon[:, :, :5].argmax(dim=-1), num_classes=5)
                oh_edges_recon = F.one_hot(edges_recon.argmax(dim=-1), num_classes=4)
                valid, unique, novel, valid_w_corr = qm9_eval(oh_nodes_recon, oh_edges_recon.permute(0, 3, 1, 2))

                valid_s.append(valid)
                unique_s.append(unique)
                novel_s.append(novel)
                valid_w_corr_s.append(valid_w_corr)
        return sum(valid_s)/n_samples, np.mean(unique_s), np.mean(novel_s), sum(valid_w_corr_s)/n_samples


class CategoricalGraphSize:
    def __init__(self, train_loader):
        # Collect all graph sizes
        graph_sizes = []
        for batch in train_loader:
            sizes = batch.batch.bincount().tolist()
            graph_sizes.extend(sizes)
        
        counts = Counter(graph_sizes)
        total = sum(counts.values())
        density = {item: count / total for item, count in counts.items()}

        self.items = list(density.keys())
        self.probabilities = list(density.values())
    
    def sample(self, n_samples):
        return torch.tensor(np.random.choice(self.items, size=n_samples, p=self.probabilities))
