import torch
from torch import nn

from model.vgae_basics import Encoder, Decoder, DownPooling
from model.vgae_quantizer import VectorQuantizer as Quantizer


class VQVAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder(
            n_layers=config.vqvgae.encoder.n_layers,
            hidden_dim=config.vqvgae.encoder.hidden_dim,
            emb_dim=config.vqvgae.encoder.emb_dim,
            in_node_feature_dim=config.dataset.node_feature_dim + config.dataset.additional_node_features,
            in_edge_feature_dim=config.dataset.edge_feature_dim,
            out_node_feature_dim=config.vqvgae.quantizer.emb_dim,
        )

        self.scales = config.vqvgae.quantizer.scales
        self.down_pooling = DownPooling(
            node_dim=config.vqvgae.quantizer.emb_dim,
            pooling_to_size=self.scales[-1],
        )

        self.quantizer = Quantizer(
            codebook_size=config.vqvgae.quantizer.codebook_size, 
            embedding_dim=config.vqvgae.quantizer.emb_dim, 
            commitment_cost=config.vqvgae.quantizer.commitment_cost,
            init_steps=config.vqvgae.quantizer.init_steps,
            collect_desired_size=config.vqvgae.quantizer.collect_desired_size,
            scales=self.scales,
        )

        self.decoder = Decoder(
            # Latent Attention Projection
            latent_proj_n_layers=config.vqvgae.latent_attn_proj.n_layers, 
            latent_proj_n_heads=config.vqvgae.latent_attn_proj.n_heads, 
            latent_proj_max_gsize=config.vqvgae.latent_attn_proj.max_gsize, 
            # GNN Decoding
            n_layers=config.vqvgae.decoder.n_layers, 
            hidden_dim=config.vqvgae.decoder.hidden_dim, 
            emb_dim=config.vqvgae.decoder.emb_dim,
            in_node_feature_dim=config.vqvgae.quantizer.emb_dim,
            out_node_feature_dim=config.dataset.node_feature_dim,
            out_edge_feature_dim=1+config.dataset.edge_feature_dim,        # +1 for "no edge" prediction
        )
        
    def forward(self, batch):
        # Encoder
        init_graph_sizes = batch.batch.bincount()
        node_feat, _ = self.encoder(batch)
        # Down Pooling
        node_feat = self.down_pooling(node_feat, edge_index=batch.edge_index, batch=batch.batch)
        node_feat = node_feat.view(len(init_graph_sizes), self.scales[-1], self.quantizer.embedding_dim) # B, max_scale, 16
        # Quantizer
        quantized, commitment_loss, q_latent_loss = self.quantizer(node_feat) # B, max_scale, C
        # Decoder
        nodes_recon, edges_recon, node_masks = self.decoder(quantized, init_graph_sizes=init_graph_sizes)
        return commitment_loss, q_latent_loss, nodes_recon, edges_recon, node_masks
    
    def forward_init(self, batch):
        # Encoder
        init_graph_sizes = batch.batch.bincount()
        node_feat, _ = self.encoder(batch)
        # Down Pooling
        node_feat = self.down_pooling(node_feat, edge_index=batch.edge_index, batch=batch.batch)
        node_feat = node_feat.view(len(init_graph_sizes), self.scales[-1], self.quantizer.embedding_dim)

        # First stage: VAE-only latent training, no quantization
        if self.quantizer.init_steps > 0:
            self.quantizer.init_steps -= 1
        
        # Secons stage: collect latent to initialise codebook words with k++ means, no quantization
        elif self.quantizer.collect_phase:
            self.quantizer.collect_samples(node_feat.detach())

        # Decoder
        nodes_recon, edges_recon, node_masks = self.decoder(node_feat, init_graph_sizes=init_graph_sizes)
        return nodes_recon, edges_recon, node_masks
    
    def graph_to_idxBl(self, batch):
        # Encoder
        init_graph_sizes = batch.batch.bincount()
        node_feat, _ = self.encoder(batch)
        # Down Pooling
        node_feat = self.down_pooling(node_feat, edge_index=batch.edge_index, batch=batch.batch)
        node_feat = node_feat.view(len(init_graph_sizes), self.scales[-1], self.quantizer.embedding_dim)
        return self.quantizer.f_to_idxBl(node_feat)
    
    def fhat_to_graph(self, f_hat, init_graph_sizes):
        # Decoder
        nodes_recon, edges_recon, node_masks = self.decoder(f_hat.permute(0, 2, 1), init_graph_sizes=init_graph_sizes)
        return nodes_recon, edges_recon, node_masks
