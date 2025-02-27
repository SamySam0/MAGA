import torch
import numpy as np
from tqdm import tqdm

from model.vgae_helpers import prepare_for_exp
from utils.losses import get_losses, get_edge_masks
from eval.evaluation import qm9_eval


class VQVGAE_Trainer(object):
    def __init__(self, model, optimizer, scheduler, dataloaders, device, gamma):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.train_loader, self.valid_loader = dataloaders
        self.gamma = gamma # config.train.gamma

    def step(self, batch, train: bool, init_phase: bool = False, experimenting: bool = False):
        if train:
            self.optimizer.zero_grad()

        if init_phase:
            nodes_recon, edges_recon, node_masks = self.model.forward_init(batch)
        else:
            commitment_loss, q_latent_loss, nodes_recon, edges_recon, node_masks = self.model(batch)
        
        masks = node_masks.detach(), get_edge_masks(node_masks) 

        recon_loss, rec_losses = get_losses(
            batch=batch, 
            nodes_rec=nodes_recon, 
            edges_rec=edges_recon, 
            n_node_feat=self.model.decoder.out_node_feature_dim,
            node_masks=masks[0].unsqueeze(-1), 
            edge_masks=masks[1],
        )

        if init_phase:
            loss = recon_loss
        else:
            loss = recon_loss + (commitment_loss + q_latent_loss) * self.gamma
        
        if train:
            loss.backward()
            self.optimizer.step()
        
        if experimenting:
            return nodes_recon, edges_recon, masks[0], masks[1]
        return recon_loss.item()
    
    def train_ep(self):
        self.model.train()
        batch_recon_loss = []
        for batch in tqdm(self.train_loader, total=len(self.train_loader), desc='Training', leave=False):
            recon_loss = self.step(batch.to(self.device), train=True)
            batch_recon_loss.append(recon_loss)
        return np.mean(batch_recon_loss)

    def valid_ep(self):
        self.model.eval()
        batch_recon_loss = []
        for batch in tqdm(self.valid_loader, total=len(self.valid_loader), desc='Evaluation (valid)', leave=False):
            with torch.no_grad():
                recon_loss = self.step(batch.to(self.device), train=False)
            batch_recon_loss.append(recon_loss)
        
        val_recon_loss = np.mean(batch_recon_loss)
        self.scheduler.step(val_recon_loss)
        return val_recon_loss
    
    def qm9_exp(self):
        self.model.eval()
        valid_s, unique_s, novel_s, fcd_s, valid_w_corr_s = [], [], [], [], [] 
        for batch in tqdm(self.valid_loader, total=len(self.valid_loader), desc='Experiment: Molecule Validity', leave=False):
            with torch.no_grad():
                annots_recon, adjs_recon, node_masks, edge_masks = self.step(batch.to(self.device), train=False, experimenting=True)
                annots_recon, adjs_recon = prepare_for_exp(annots_recon, adjs_recon, node_masks, edge_masks)
                valid, unique, novel, valid_w_corr = qm9_eval(annots_recon, adjs_recon)

                valid_s.append(valid)
                unique_s.append(unique)
                novel_s.append(novel)
                valid_w_corr_s.append(valid_w_corr)
        
        n_samples = sum(len(batch.batch.bincount()) for batch in self.valid_loader)
        return sum(valid_s)/n_samples, np.mean(unique_s), np.mean(novel_s), sum(valid_w_corr_s)/n_samples
    
    def init_codebook_training(self):
        while self.model.quantizer.init_steps > 0 or self.model.quantizer.collect_phase:
            epoch_loss = []
            for batch in self.train_loader:
                if self.model.quantizer.init_steps == 1: 
                    print('Starting Initialisation Phase 2...')
                train_recon_loss = self.step(batch.to(self.device), train=True, init_phase=True)
                epoch_loss.append(train_recon_loss)
                if self.model.quantizer.init_steps <= 0 and not self.model.quantizer.collect_phase:
                    return np.mean(epoch_loss)
            else:
                print('Epoch training loss:', np.mean(epoch_loss))
