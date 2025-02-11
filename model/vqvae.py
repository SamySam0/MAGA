import torch
from torch import nn

from model.basic_vgae import Encoder, Decoder
from model.quantizer import VectorQuantizer as Quantizer
from torch_geometric.utils import to_dense_batch


class VQVAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder(
            n_layers=config.model.encoder.n_layers,
            hidden_dim=config.model.encoder.hidden_dim,
            emb_dim=config.model.encoder.emb_dim,
            in_node_feature_dim=config.data.node_feature_dim,
            in_edge_feature_dim=config.data.edge_feature_dim,
            out_node_feature_dim=config.model.quantizer.emb_dim,
        )

        self.quantizer = Quantizer(
            codebook_size=config.model.quantizer.codebook_size, 
            embedding_dim=config.model.quantizer.emb_dim, 
            commitment_cost=config.model.quantizer.commitment_cost,
        )

        self.decoder = Decoder(
            n_layers=config.model.decoder.n_layers, 
            hidden_dim=config.model.decoder.hidden_dim, 
            emb_dim=config.model.decoder.emb_dim,
            in_node_feature_dim=config.model.quantizer.emb_dim,
            out_node_feature_dim=config.data.node_feature_dim,
            out_edge_feature_dim=config.data.edge_feature_dim,
        )
        
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

    def forward_test(self, batch):
        commitment_loss, q_latent_loss, nodes_recon, edges_recon, node_masks = self.forward(batch)
        return nodes_recon, edges_recon
