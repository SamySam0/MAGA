import torch
from torch import nn
from torch.nn import functional as F

from model.basic_vgae import Encoder, Decoder
from model.quantizer import VectorQuantizer as Quantizer
from torch_geometric.utils import to_dense_batch

class VQVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(n_layers=3, hidden_dim=64, emb_dim=32) # Output dim is 16
        self.quantizer = Quantizer(codebook_size=256, embedding_dim=16, commitment_cost=0.25)
        self.decoder = Decoder(n_layers=3, hidden_dim=64, emb_dim=32)
        
    def encode(self, x):
        out_nodes, _ = self.encoder(x)
        quantized, commitment_loss, q_latent_loss, perplexity, encoding_indices = self.quantizer(out_nodes)
        return quantized, commitment_loss, q_latent_loss
    
    def decode(self, x):
        nodes_rec, edges_rec = self.decoder(x, mask=node_masks)
        return nodes_rec, edges_rec
    
    def forward(self, batch):
        quantized, commitment_loss, q_latent_loss = self.encode(batch)
        quantized, node_masks = to_dense_batch(quantized, batch.batch)
        nodes_recon, edges_recon = self.decoder(quantized, mask=node_masks)
        return commitment_loss, q_latent_loss, nodes_recon, edges_recon, node_masks
