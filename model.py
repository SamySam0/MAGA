import torch
import numpy as np

class GraphRNN:
    def __init__(self, device, input_size=16, max_nodes=10):

        self.MAX_NODES = max_nodes

        # Define tokens
        self.SOS_TOKEN_GL = torch.tensor(
            [3]*input_size, dtype=torch.float32,
        ).unsqueeze(dim=0)

        self.SOS_TOKEN_EL = torch.tensor(
            [3], dtype=torch.float32
        ).unsqueeze(dim=0)

        # Graph-level RNN
        self.graph_level_rnn = torch.nn.GRU(
            input_size=input_size,
            hidden_size=128,
            num_layers=4,
            dtype=torch.float32,
            device=device,
        )

        # Edge-level RNN
        self.edge_input_dim_reduc = torch.nn.Linear(128, 16)

        self.edge_level_rnn = torch.nn.GRU(
            input_size=1,
            hidden_size=16,
            num_layers=4,
            dtype=torch.float32,
            device=device,
        )

        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(16, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 1),
            torch.nn.Sigmoid(),
        )
    

    def forward(self):
        edges = np.empty((self.MAX_NODES, self.MAX_NODES))

        hn_gl, x_gl = None, self.SOS_TOKEN_GL
        for n in range(1, self.MAX_NODES):

            # Graph-level RNN
            _, hn_gl = self.graph_level_rnn(x_gl, hn_gl)
            hn_el = self.edge_input_dim_reduc(hn_gl)
            x_el = self.SOS_TOKEN_EL

            # Edge-level RNN
            for e in range(n):
                out_el, hn_el = self.edge_level_rnn(x_el, hn_el)
                out_el = self.edge_mlp(out_el)
                print(out_el.shape)

                # Flip a coin given sigmoid probability
                x_el = torch.bernoulli(out_el).item()
                edges[n,e] = x_el
                x_el = torch.tensor(
                    [x_el], dtype=torch.float32
                ).unsqueeze(dim=0)
            
            # Input to next cell of graph-level RNN is the hidden 
            # state of the last edge-level RNN cell
            x_gl = hn_el

            # Check for EOS (isolated node, no edges)
            if sum(edges[n,:]) == 0:
                break
        
        return edges

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GraphRNN(device=device)
    edges = model.forward()
    print(edges)