import torch
from torch import nn
from torch.nn import functional as F
from scipy.cluster.vq import kmeans2


class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size, embedding_dim, commitment_cost, init_steps, collect_desired_size):
        super().__init__()

        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(codebook_size, embedding_dim)
        self.embedding.weight.data.uniform_(-1/codebook_size, 1/codebook_size)

        self.init_steps = init_steps
        self.collect_phase = init_steps > 0
        collected_samples = torch.Tensor(0, self.embedding_dim)
        self.collect_desired_size = collect_desired_size
        self.register_buffer("collected_samples", collected_samples)
        
    def forward(self, f_BChw, mask=None):
        f_BChw = f_BChw.permute(0, 2, 1)
        B, C, N = f_BChw.shape

        v_patch_nums = list(range(1, N+1))
        
        f_no_grad = f_BChw.detach()
        f_rest = f_no_grad.clone()
        f_hat  = torch.zeros_like(f_rest)

        with torch.cuda.amp.autocast(enabled=False):
            mean_q_latent_loss: torch.Tensor = 0.0
            mean_commitment_loss: torch.Tensor = 0.0
            SN = len(v_patch_nums)
            for si, pn in enumerate(v_patch_nums):
                if si != SN-1:
                    rest_NC = F.interpolate(f_rest, size=(pn), mode='area').permute(0, 2, 1).reshape(-1, C)
                else:
                    rest_NC = f_rest.permute(0, 2, 1).reshape(-1, C)
                if self.collect_phase:
                    self.collected_samples = torch.cat((self.collected_samples, rest_NC), dim=0)
                
                d_no_grad = torch.sum(rest_NC.square(), dim=1, keepdim=True) + torch.sum(self.embedding.weight.data.square(), dim=1, keepdim=False)
                d_no_grad.addmm_(rest_NC, self.embedding.weight.data.T, alpha=-2, beta=1)
                idx_N = torch.argmin(d_no_grad, dim=1)

                idx_Bhw = idx_N.view(B, pn)
                if si != SN-1:
                    h_BChw = F.interpolate(self.embedding(idx_Bhw).permute(0, 2, 1), size=(N), mode='linear').contiguous()
                else:
                    h_BChw = self.embedding(idx_Bhw).permute(0, 2, 1).contiguous()
                # h_BChw = quant_resi[si/(SN-1)](h_BChw)
                f_hat = f_hat + h_BChw
                f_rest -= h_BChw

                mean_commitment_loss += F.mse_loss(f_hat.data, f_BChw).mul_(0.25)
                mean_q_latent_loss += F.mse_loss(f_hat, f_no_grad)
            
            mean_commitment_loss *= 1. / SN
            mean_q_latent_loss *= 1. / SN

            f_hat = (f_hat.data - f_no_grad).add_(f_BChw)
            f_hat = f_hat.permute(0, 2, 1)
            
            return f_hat, mean_commitment_loss, mean_q_latent_loss

    def collect_samples(self, zq):
        # Collect samples
        self.forward(zq)
        # If enough samples collected, initialise codebook with k++ means
        if self.collected_samples.shape[0] >= self.collect_desired_size:
            self.collected_samples = self.collected_samples[-self.collect_desired_size:]
            self.collect_phase = False
            self.kmeans_init()
            self.collected_samples = torch.Tensor(0, self.embedding_dim)
    
    def kmeans_init(self):
        print('K++ means Codebook initialisation starting...')
        device = self.collected_samples.device
        collected_samples = self.collected_samples.cpu().detach().numpy()
        # Perform k-means clustering on the entire embedding space
        k = kmeans2(collected_samples, self.codebook_size, minit='++')[0]
        # Update embedding weights with k-means centroids
        self.embedding.weight.data = torch.from_numpy(k).to(device)
        print('K++ Success!')
