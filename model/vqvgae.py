import torch
from torch import nn
from typing import List

from model.vgae_basics import Encoder, Decoder, DownPooling
from model.vgae_quantizer import VectorQuantizer as Quantizer
from model.vgae_helpers import interpolate_batch, sizes_to_mask
from torch_geometric.utils import to_dense_batch


class VQVAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder(
            n_layers=config.vqvgae.encoder.n_layers,
            hidden_dim=config.vqvgae.encoder.hidden_dim,
            emb_dim=config.vqvgae.encoder.emb_dim,
            in_node_feature_dim=config.data.node_feature_dim + config.data.additional_node_features,
            in_edge_feature_dim=config.data.edge_feature_dim,
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
            n_layers=config.vqvgae.decoder.n_layers, 
            hidden_dim=config.vqvgae.decoder.hidden_dim, 
            emb_dim=config.vqvgae.decoder.emb_dim,
            in_node_feature_dim=config.vqvgae.quantizer.emb_dim,
            out_node_feature_dim=config.data.qm9_node_feature_dim,
            out_edge_feature_dim=config.data.edge_feature_dim,
        )
        
    def forward(self, batch):
        # Encoder
        original_graph_sizes = batch.batch.bincount()
        node_feat, _ = self.encoder(batch)

        # Down Pooling
        node_feat, batch_idx = self.down_pooling(node_feat, edge_index=batch.edge_index, batch=batch.batch)
        node_feat = node_feat.view(len(original_graph_sizes), self.scales[-1], self.quantizer.embedding_dim)
        
        # Quantizer
        quantized, commitment_loss, q_latent_loss = self.quantizer(node_feat) # B, max_scale, C

        # Interpolate to original graph sizes and pad
        graphs = torch.split(quantized, 1)
        quantized, node_masks = interpolate_batch(
            graphs=graphs, to_sizes=original_graph_sizes, padding_size=max(original_graph_sizes),
        ) # B, max_nodes, C

        # Decoder
        nodes_recon, edges_recon = self.decoder(quantized, mask=node_masks)
        return commitment_loss, q_latent_loss, nodes_recon, edges_recon, node_masks
    
    def forward_init(self, batch):
        original_graph_sizes = batch.batch.bincount()

        node_feat, _ = self.encoder(batch)
        node_feat, batch_idx = self.down_pooling(node_feat, edge_index=batch.edge_index, batch=batch.batch)
        node_feat = node_feat.view(len(original_graph_sizes), self.scales[-1], self.quantizer.embedding_dim)

        # First stage: VAE-only latent training, no quantization
        if self.quantizer.init_steps > 0:
            self.quantizer.init_steps -= 1
        
        # Secons stage: collect latent to initialise codebook words with k++ means, no quantization
        elif self.quantizer.collect_phase:
            self.quantizer.collect_samples(node_feat.detach())
        
        # Interpolate to original graph sizes and pad
        graphs = torch.split(node_feat, 1)
        quantized, node_masks = interpolate_batch(
            graphs=graphs, to_sizes=original_graph_sizes, padding_size=max(original_graph_sizes),
        ) # B, max_nodes, C

        nodes_recon, edges_recon = self.decoder(quantized, mask=node_masks)
        return nodes_recon, edges_recon, node_masks
    
    def graph_to_idxBl(self, batch):
        original_graph_sizes = batch.batch.bincount()

        node_feat, _ = self.encoder(batch)
        node_feat, batch_idx = self.down_pooling(node_feat, edge_index=batch.edge_index, batch=batch.batch)
        node_feat = node_feat.view(len(original_graph_sizes), self.scales[-1], self.quantizer.embedding_dim)

        return self.quantizer.f_to_idxBl(node_feat)
    
    def fhat_to_graph(self, f_hat, original_sizes):
        node_masks = sizes_to_mask(original_sizes, max_size=self.scales[-1], device=f_hat.device)
        graphs = torch.split(f_hat.permute(0, 2, 1), 1)
        interpolated_nodes = interpolate_batch(
            graphs=graphs, to_sizes=original_sizes, padding_size=node_masks.size(1),
        ) # B, max_nodes, C

        # Decoder
        nodes_recon, edges_recon = self.decoder(interpolated_nodes, mask=node_masks)
        return nodes_recon, edges_recon, node_masks
