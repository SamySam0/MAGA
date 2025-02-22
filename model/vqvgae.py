import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Tuple

from model.vgae_basics import Encoder, Decoder
from model.vgae_quantizer import VectorQuantizer as Quantizer
from torch_geometric.utils import to_dense_batch


class VQVAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder(
            n_layers=config.model.encoder.n_layers,
            hidden_dim=config.model.encoder.hidden_dim,
            emb_dim=config.model.encoder.emb_dim,
            in_node_feature_dim=config.data.node_feature_dim + config.data.additional_node_features,
            in_edge_feature_dim=config.data.edge_feature_dim,
            out_node_feature_dim=config.model.quantizer.emb_dim,
        )

        self.scales = config.model.quantizer.scales
        self.quantizer = Quantizer(
            codebook_size=config.model.quantizer.codebook_size, 
            embedding_dim=config.model.quantizer.emb_dim, 
            commitment_cost=config.model.quantizer.commitment_cost,
            init_steps=config.model.quantizer.init_steps,
            collect_desired_size=config.model.quantizer.collect_desired_size,
            scales=self.scales,
        )

        self.decoder = Decoder(
            n_layers=config.model.decoder.n_layers, 
            hidden_dim=config.model.decoder.hidden_dim, 
            emb_dim=config.model.decoder.emb_dim,
            in_node_feature_dim=config.model.quantizer.emb_dim,
            out_node_feature_dim=config.data.node_feature_dim,
            out_edge_feature_dim=config.data.edge_feature_dim,
        )
        
    def forward(self, batch):
        # Encoder
        node_feat, _ = self.encoder(batch)

        # Save mask for decoding
        _, node_masks = to_dense_batch(node_feat, batch.batch) # mask: B, max_nodes

        # Interpolate all graphs to max scale's size
        graphs = torch.split(node_feat, batch.batch.bincount().tolist())
        interpolated_nodes = interpolate_batch(
            graphs=graphs, 
            to_sizes=list(self.scales[-1] for _ in range(len(graphs))), 
            padding_size=None
        ) # B, max_scale, C

        # Quantizer
        quantized, commitment_loss, q_latent_loss = self.quantizer(interpolated_nodes) # B, max_scale, C

        # Interpolate back to original sizes and pad
        graphs = torch.split(quantized, 1)
        original_sizes = node_masks.sum(1)
        quantized = interpolate_batch(
            graphs=graphs, to_sizes=original_sizes, padding_size=node_masks.size(1),
        ) # B, max_nodes, C

        # Decoder
        nodes_recon, edges_recon = self.decoder(quantized, mask=node_masks)
        return commitment_loss, q_latent_loss, nodes_recon, edges_recon, node_masks
    
    def forward_init(self, batch):
        node_feat, _ = self.encoder(batch) 
        quantized, node_masks = to_dense_batch(node_feat, batch.batch) # B, max_nodes, C and B, max_nodes

        # First stage: VAE-only latent training, no quantization
        if self.quantizer.init_steps > 0:
            self.quantizer.init_steps -= 1
        
        # Secons stage: collect latent to initialise codebook words with k++ means, no quantization
        elif self.quantizer.collect_phase:
            # Interpolate all graphs to max scale's size
            graphs = torch.split(node_feat, batch.batch.bincount().tolist())
            interpolated_nodes = interpolate_batch(
                graphs=graphs, 
                to_sizes=list(self.scales[-1] for _ in range(len(graphs))), 
                padding_size=None
            ) # B, max_scale, C

            self.quantizer.collect_samples(interpolated_nodes.detach())
        
        nodes_recon, edges_recon = self.decoder(quantized, mask=node_masks)
        return nodes_recon, edges_recon, node_masks
    
    def img_to_idxBl(self, batch):
        node_feat, _ = self.encoder(batch)
        graphs = torch.split(node_feat, batch.batch.bincount().tolist())

        # Interpolate all graphs to max scale's size
        interpolated_nodes = interpolate_batch(
            graphs=graphs, 
            to_sizes=list(self.scales[-1] for _ in range(len(graphs))), 
            padding_size=None,
        ) # B, max_scale, C

        return self.quantizer.f_to_idxBl(interpolated_nodes)


def interpolate_batch(graphs: Tuple[torch.Tensor], to_sizes: List[int], padding_size: int = None):
    '''
    Interpolate all graphs to desired sizes.
    If sizes are different, add padding.
    '''
    # Interpolate all graphs to desired sizes
    interpolated_nodes = []
    for graph, size in zip(graphs, to_sizes):
        if len(graph.shape) < 3: graph = graph.unsqueeze(0)
        graph = graph.transpose(1, 2)
        graph = F.interpolate(graph, size=(size if isinstance(size, int) else size.item()), mode='linear')

        # Add padding if necessary
        if padding_size is not None:
            padded = torch.zeros(1, graph.size(1), padding_size, device=graph.device)
            padded[:, :, :size] = graph
            graph = padded

        interpolated_nodes.append(graph.transpose(1, 2).squeeze(0))
    
    return torch.stack(interpolated_nodes)
