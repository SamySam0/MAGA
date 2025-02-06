import torch
from torch import nn
from torch.nn import functional as F


class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size, embedding_dim, commitment_cost):
        super().__init__()

        self.codebook_size = codebook_size      # N
        self.embedding_dim = embedding_dim      # K = C
        self.commitment_cost = commitment_cost  # beta/lambda

        self.embedding = nn.Embedding(codebook_size, embedding_dim)
        self.embedding.weight.data.uniform_(-1/codebook_size, 1/codebook_size)

    def forward(self, x):
        inputs = x.permute(0, 2, 3, 1).contiguous() # [C, H, W] -> [H, W, C]
        input_shape = inputs.shape

        flat_input = inputs.view(-1, self.embedding_dim) # [H, W, C] -> [H*W, C]

        # Calculate the distance between each embedding and each codebook vector
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.t())) # [H*W, N]

        # Find the closest codebook vector (one-hot encodding)
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.codebook_size, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1) # [H*W, N]

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape) # [H, W, K=C]

        # Create loss that pulls encoder embeddings and codebook vector selected
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Reconstructions quantized representation using the encoder embeddings
        # to allow for backpropagation of gradients into decoder
        # (In forward equals quantized, in backward calculates gradients on inputs)
        quantized = inputs + (quantized - inputs).detach()
        
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), encodings
