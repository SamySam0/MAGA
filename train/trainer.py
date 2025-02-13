import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml
from model.vqvae import VQVAE
from easydict import EasyDict as edict
from utils.losses import get_losses
from utils.func import discretize, get_edge_target, get_edge_masks

class Trainer:
    def __init__(self, dataloaders, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = VQVAE(config=config).to(self.device)
        self.train_loader, self.valid_loader = dataloaders
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.train.lr, betas=(config.train.beta1, config.train.beta2))
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=config.train.lr_decay, patience=4, min_lr=2*1e-5)
        self.gamma = config.train.gamma
        self.n_epochs = config.train.epochs
        self._log_model_parameters()

    def _log_model_parameters(self):
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Number of trainable parameters in the model: {num_params}")

    def step(self, batch, train: bool, init_phase: bool = False):
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
            node_masks=node_masks.unsqueeze(-1), 
            annotated_nodes=True, 
            annotated_edges=True, 
            max_node_num=9, 
            n_node_feat=11, 
            edge_masks=masks[1],
        )

        if init_phase:
            loss = recon_loss
        else:
            loss = recon_loss + (commitment_loss + q_latent_loss) * self.gamma
        
        if train:
            loss.backward()
            self.optimizer.step()
        
        return recon_loss.item()
    
    def train_step(self):
        self.model.train()
        batch_recon_loss = []
        for batch in self.train_loader:
            recon_loss = self.step(batch.to(self.device), train=True)
            batch_recon_loss.append(recon_loss)
        return np.mean(batch_recon_loss)

    def valid_step(self):
        self.model.eval()
        batch_recon_loss = []
        for batch in self.valid_loader:
            with torch.no_grad():
                recon_loss = self.step(batch.to(self.device), train=False)
            batch_recon_loss.append(recon_loss)
        return np.mean(batch_recon_loss)
    
    def train(self):
        # Phase 1 and 2: codebook initialisation
        print('Starting Initialisation...')
        while self.model.quantizer.init_steps > 0 or self.model.quantizer.collect_phase:
            epoch_loss = []
            for batch in self.train_loader:
                if self.model.quantizer.init_steps == 1: 
                    print('Starting Initialisation Phase 2...')
                train_recon_loss = self.step(batch.to(self.device), train=True, init_phase=True)
                epoch_loss.append(train_recon_loss)
                if self.model.quantizer.init_steps <= 0 and not self.model.quantizer.collect_phase:
                    print("Initialisation terminated. Final epoch's partial loss:", np.mean(epoch_loss))
                    break
            else:
                print('Epoch training loss:', np.mean(epoch_loss))
        
        # Phase 2: VQVAE training
        print('Starting Training...')
        for epoch in range(1, self.n_epochs+1):
            train_recon_loss = self.train_step()
            valid_recon_loss = self.valid_step()
            if epoch % 5 == 0:
                print(f"Epoch: {epoch}, Train Loss: {train_recon_loss}, Valid Loss: {valid_recon_loss}, LR: {self.scheduler.optimizer.param_groups[0]['lr']}")
            self.scheduler.step(valid_recon_loss)
