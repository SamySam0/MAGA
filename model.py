''' 
GraphRNN model: A hierarchical RNN, where the first (i.e., the graph-level) 
RNN generates the nodes and maintains the state of the graph, while the second 
(i.e., the edge-level) RNN generates the edges of a given node. 
Implementation following details from https://arxiv.org/pdf/1802.08773 .
'''

import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class GraphLevelRNN(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.input_size = input_size
        self.num_layers = num_layers

        self.linear_in = nn.Linear(
            in_features=input_size,
            out_features=embedding_size,
            dtype=torch.float32,
        )
        self.relu = nn.ReLU()

        self.gru = nn.GRU(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dtype=torch.float32,
        )

        self.linear_out = nn.Linear(
            in_features=hidden_size, 
            out_features=output_size,
            dtype=torch.float32,
        )

        self.hidden = None # defaults to zeros tensor

    def forward(self, x, x_lens):
        x = torch.flatten(x, 2, 3) # TODO: this fixes a bug that shouldn't exist
        x = self.relu(self.linear_in(x)) # [batch, seq_len, embedding_dim]
        x = pack_padded_sequence(x, x_lens, batch_first=True, enforce_sorted=False)
        x, self.hidden = self.gru(x, self.hidden) # [batch, seq_len, hidden_size]
        # Unpack (reintroduce padding)
        x, _ = pad_packed_sequence(x, batch_first=True)
        x = self.linear_out(x) # [batch, seq_len, output_size]
        return x
    
    def reset_hidden(self):
        self.hidden = None


class EdgeLevelRNN(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_layers):
        super().__init__()
        self.num_layers = num_layers

        self.linear_in = nn.Linear(
            in_features=1,
            out_features=embedding_size,
            dtype=torch.float32,
        )
        self.relu = nn.ReLU()

        self.gru = nn.GRU(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dtype=torch.float32,
        )

        self.mlp_out = nn.Sequential(
            nn.Linear(hidden_size, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, 1),
            nn.Sigmoid(),
        )

        self.hidden = None # defaults to zeros tensor
    
    def set_first_layer_hidden(self, hn):
        '''
        Sets the hidden state of the first GRU layer to the output of the
        graph-level RNN. The hidden state of all other layers are reset to 0.
        '''
        zeros = torch.zeros([self.num_layers-1, hn.shape[-2], hn.shape[-1]], device=hn.device)
        if len(hn.shape) == 2:
            hn = hn.unsqueeze(dim=0)
        self.hidden = torch.cat([hn, zeros], dim=0) # [num_layers, batch_size, hidden_size]

    def forward(self, x, x_lens):
        x = self.relu(self.linear_in(x)) # [batch, seq_len, embedding_dim]
        x = pack_padded_sequence(x, x_lens, batch_first=True, enforce_sorted=False)
        x, self.hidden = self.gru(x, self.hidden) # [batch, seq_len, hidden_size]
        # Unpack (reintroduce padding)
        x, _ = pad_packed_sequence(x, batch_first=True)
        x = self.mlp_out(x) # [batch, seq_len, 1 (sigmoid)]
        return x
