import torch
from torch import nn
from torch.nn import functional as F

from model.basic_vae import Encoder, Decoder
from model.quantizer import VectorQuantizer

class VQVAE(nn.Module):
    def __init__(self, 
        channel_in, num_hiddens, num_res_layers, num_res_hiddens,
        codebook_size, embedding_dim, commitment_cost=0.25,
    ):
        super().__init__()

        # Encoder
        self.encoder = Encoder(
            in_channels=channel_in, 
            num_hiddens=num_hiddens, 
            num_res_layers=num_res_layers, 
            num_res_hiddens=num_res_hiddens,
        ) # In: [H, W, channel_in] -> Out: [H/(2^2), W/(2^2), num_hidden]

        # Quantization
        self.pre_vq_conv = nn.Conv2d(
            in_channels=num_hiddens,
            out_channels=embedding_dim,
            kernel_size=1, stride=1,
        ) # Out: [H/(2^2), W/(2^2), embedding_dim]

        self.vq = VectorQuantizer(
            codebook_size=codebook_size, 
            embedding_dim=embedding_dim, 
            commitment_cost=commitment_cost,
        ) # Out: [H/(2^2), W/(2^2), embedding_dim]

        # Decoder
        self.decoder = Decoder(
            in_channels = embedding_dim,
            out_channels = channel_in,
            num_hiddens = num_hiddens, 
            num_res_layers = num_res_layers, 
            num_res_hiddens = num_res_hiddens,
        ) # Out: [H, W, channel_in]
        
    def encode(self, x):
        encoding = self.encoder(x)
        encoding = self.pre_vq_conv(encoding)
        vq_loss, quantized, _ = self.vq(encoding)
        return vq_loss, quantized
    
    def decode(self, x):
        return self.decoder(x)
    
    def forward(self, x):
        vq_loss, quantized = self.encode(x)
        recon = self.decode(quantized)
        return vq_loss, recon
