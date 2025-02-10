import torch
from torch import nn
from torch.nn import functional as F


class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size, embedding_dim, commitment_cost):
        super().__init__()

        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(codebook_size, embedding_dim)
        self.embedding.weight.data.uniform_(-1/codebook_size, 1/codebook_size)

    def forward(self, x, mask=None):
        # Calculate the distance between each embedding and each codebook vector
        distances = (torch.sum(x**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(x, self.embedding.weight.t()))

        # Find the closest codebook vector (one-hot encodding)
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.codebook_size, device=x.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize
        quantized = torch.matmul(encodings, self.embedding.weight)
        
        # Apply masking 
        if mask is not None:
            quantized = quantized * mask.unsqueeze(-1)

        # Create loss that pulls encoder embeddings and codebook vector selected
        e_latent_loss = F.mse_loss(quantized.detach(), x)
        q_latent_loss = F.mse_loss(quantized, x.detach())
        commitment_loss = self.commitment_cost * e_latent_loss

        # Reconstructions quantized representation using the encoder embeddings
        # to allow for backpropagation of gradients into decoder
        # (In forward equals quantized, in backward calculates gradients on x)
        quantized = x + (quantized - x).detach()

        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-8)))
        
        return quantized, commitment_loss, q_latent_loss, perplexity, encoding_indices
