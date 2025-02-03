import torch
import torch.nn as nn

class MPL_VAE(nn.Module):
    '''
    Normal VAE.
    '''
    def __init__(self, hidden_size, embedding_size, output_size):
        super().__init__()
        self.encoder_mus = nn.Linear(hidden_size, embedding_size) # mus
        self.encoder_sgm = nn.Linear(hidden_size, embedding_size) # lsgms

        self.decoder_1 = nn.Linear(embedding_size, embedding_size)
        self.decoder_2 = nn.Linear(embedding_size, output_size)
        self.relu = nn.RelU()

    def forward(self, hn):
        # Encoder
        z_mus = self.encoder_mus(hn)
        z_sgm = self.encoder_sgm(hn)

        # Reparameterize
        z_sgm2 = z_sgm.mul(0.5).exp_()
        eps = Variable(torch.randn(z_sgm2.size())).cuda()
        z = (eps * z_sgm2) + z_mu

        # Decoder
        y = self.decode_2(self.relu(self.decode_1(z)))
        return y, z_mu, z_sgm


class GraphConv(nn.Module):
    ''' GCN basic operation. '''
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.weight = nn.Parameter(torch.FloatTensor(input_size, output_size).cuda())
        
    def forward(self, x, adj):
        y = torch.matmul(adj, x)
        y = torch.matmul(y, self.weight)
        return y



class GraphVAE(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, max_num_nodes, pool='sum'):
        super().__init__()
        self.max_num_nodes = max_num_nodes
        self.pool = pool

        self.conv1 = model.GraphConv(input_size=input_size, output_size=hidden_size)
        self.bn1 = nn.BatchNorm1d(input_dim)
        self.conv2 = model.GraphConv(input_size=hidden_size, output_size=hidden_size)
        self.bn2 = nn.BatchNorm1d(input_dim)
        self.relu = nn.ReLU()

        output_dim = max_num_nodes * (max_num_nodes + 1) // 2
        self.vae = MLP_VAE(
            hidden_size=input_size**2, 
            embedding_size=latent_size, 
            output_size=output_size,
        )

        


