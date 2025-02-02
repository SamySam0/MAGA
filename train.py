import torch, time, yaml
import numpy as np
from model import GraphLevelRNN, EdgeLevelRNN
from data import GraphDataSet
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


def train_rnn_step(
    graph_rnn, edge_rnn, data, 
    criterion, optimizer_graph_rnn, optimizer_edge_rnn, 
    sheduler_graph_rnn, sheduler_edge_rnn, device
):
    '''
    Train GraphRNN with RNN edge model. 
    Inspired from https://github.com/mark-koch/graph-rnn/blob/master/train.py.
    '''
    graph_rnn.zero_grad()
    edge_rnn.zero_grad()

    seq, lens = data['x'].float().to(device), data['len'].cpu()

    # If sequence doesn't have edge features, just add a dummy dimension to the end
    if len(seq.shape) == 3:
        seq = seq.unsqueeze(3)

    # Add SOS token to the node-level RNN input to prevent it from looking into the future
    one_frame = torch.ones([seq.shape[0], 1, seq.shape[2], seq.shape[3]], device=device)
    x_node_rnn = torch.cat((one_frame, seq[:, :-1, :]), dim=1)

    # Compute hidden graph-level representation
    graph_rnn.reset_hidden()
    hidden = graph_rnn(x_node_rnn, lens)

    # Pack data to be parallelized
    seq_packed = pack_padded_sequence(seq, lens, batch_first=True, enforce_sorted=False).data

    # Compute the seqence lengths of seq_packed
    seq_packed_len = []
    m = graph_rnn.input_size
    for l in lens:
        for i in range(1, l+1):
            seq_packed_len.append(min(i, m))
    seq_packed_len.sort()

    # Add SOS token to the edge-level RNN input to prevent it from looking into the future.
    one_frame = torch.ones([seq_packed.shape[0], 1, seq_packed.shape[2]], device=device)
    x_edge_rnn = torch.cat((one_frame, seq_packed[:, :-1, :]), dim=1)
    y_edge_rnn = seq_packed

    # Set the hidden state of the first EdgeRNN layer to the previously computed hidden
    # representation. Sice we feed node-packed data to the EdgeRNN, we also need to pack
    # the hidden representation.
    hidden_packed = pack_padded_sequence(hidden, lens, batch_first=True, enforce_sorted=False).data
    edge_rnn.set_first_layer_hidden(hidden_packed)

    # Compute edge probabilities
    y_edge_rnn_pred = edge_rnn(x_edge_rnn, seq_packed_len)

    y_edge_rnn = pack_padded_sequence(y_edge_rnn, seq_packed_len, batch_first=True, enforce_sorted=False)
    y_edge_rnn, _ = pad_packed_sequence(y_edge_rnn, batch_first=True)

    # Calculate loss
    loss = criterion(y_edge_rnn_pred, y_edge_rnn)
    loss.backward()
    optimizer_graph_rnn.step()
    optimizer_edge_rnn.step()
    sheduler_graph_rnn.step()
    sheduler_edge_rnn.step()

    return loss.item()


def valid_rnn_step(num_nodes, node_model, edge_model, input_size):
    # Initialize adjacency vector arbitrarily to all ones
    adj_vec = torch.ones([1, 1, input_size, node_model.edge_feature_len], device=device)
    # Data structure for storing final adjacency matrix
    list_adj_vecs = []

    node_model.reset_hidden()

    def edge_gen_function(edge_rnn, h, num_edges, adj_vec_size):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        adj_vec = torch.zeros([1, 1, adj_vec_size, edge_rnn.edge_feature_len], device=device)

        edge_rnn.set_first_layer_hidden(h)

        # SOS token
        x = torch.ones([1, 1, edge_rnn.edge_feature_len], device=device)

        for i in range(num_edges):
            # calculate probability of this edge existing
            prob = edge_rnn(x)
            # sample from this probability and assign value to adjacency vector
            # assign the value of this edge into the input of the next iteration
            x[0, 0, :] = prob[0, 0, :].detach()
            adj_vec[0, 0, i, :] = x[0, 0, :]

        return adj_vec
    
    def m_seq_to_adj_mat(m_seq, m):
        n = m_seq.shape[0] + 1
        adj_mat = np.zeros((n, n))
        for i, prev_nodes in enumerate(m_seq):
            adj_mat[i+1, max(i+1-m, 0) : i+1] = list(reversed(prev_nodes[:i+1 - max(i+1-m, 0)]))
        return adj_mat

    for i in range(1, num_nodes):
        # Initialize graph state vector by running model on values from previous iteration
        # (or on the ones vector for first iteration)
        h = node_model(adj_vec)

        # Run model to generate edges and save output
        adj_vec = edge_gen_function(edge_model, h, num_edges=min(i, input_size), adj_vec_size=input_size)
        list_adj_vecs.append(adj_vec[0, 0, :min(num_nodes, input_size), 0].cpu().detach().int().numpy())

        # EOS
        if np.array(list_adj_vecs[-1] == 0).all():
            break

    # Turn into full adjacency matrix
    adj = m_seq_to_adj_mat(np.array(list_adj_vecs), m=input_size)

    adj = adj + adj.T
    adj = adj[~np.all(adj == 0, axis=1)]
    adj = adj[:, ~np.all(adj == 0, axis=0)]

    adj = np.tril(adj)
    adj = adj + adj.T

    # Remove isolated nodes as done in the GraphRNN paper.
    adj = adj[~np.all(adj == 0, axis=1)]
    adj = adj[:, ~np.all(adj == 0, axis=0)]

    return adj




if __name__ == "__main__":
    # Load in model and training configs
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    node_model = GraphLevelRNN(
        input_size=config['data']['m'],
        embedding_size=config['model']['GraphRNN']['embedding_size'],
        hidden_size=config['model']['GraphRNN']['hidden_size'],
        num_layers=config['model']['GraphRNN']['num_layers'],
        output_size=config['model']['EdgeRNN']['hidden_size'], # EdgeRNN's hidden size
    ).to(device)

    edge_model = EdgeLevelRNN(
        embedding_size=config['model']['EdgeRNN']['embedding_size'],
        hidden_size=config['model']['EdgeRNN']['hidden_size'],
        num_layers=config['model']['EdgeRNN']['num_layers'],
    ).to(device)

    criterion = torch.nn.BCELoss().to(device)

    optimizer_node = torch.optim.Adam(list(node_model.parameters()), lr=config['train']['lr'])
    optimizer_edge = torch.optim.Adam(list(edge_model.parameters()), lr=config['train']['lr'])

    scheduler_node = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_node, milestones=config['train']['lr_schedule_milestones'], gamma=config['train']['lr_schedule_gamma'])
    scheduler_edge = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_edge, milestones=config['train']['lr_schedule_milestones'], gamma=config['train']['lr_schedule_gamma'])

    # Load Training Dataset
    train_dataset = GraphDataSet(
        dataset=config['data']['dataset'], m=config['data']['m'], use_bfs=True, 
        train_split=config['data']['train_split'], val_split=config['data']['valid_split'], split="train",
    )
    train_data_loader = DataLoader(train_dataset, batch_size=config['train']['batch_size'])

    # Load Validation Dataset
    # valid_dataset = GraphDataSet(
    #     dataset=config['data']['dataset'], m=config['data']['m'], use_bfs=True, 
    #     train_split=config['data']['train_split'], val_split=config['data']['valid_split'], split="val",
    # )
    # valid_data_loader = DataLoader(valid_dataset, batch_size=1)

    # Training + Validation
    start_time = time.time()
    
    for epoch in range(config['train']['epochs']):

        node_model.train()
        edge_model.train()
        train_epoch_loss = 0
        for batch in train_data_loader:
            loss = train_rnn_step(
                node_model, edge_model, batch,
                criterion, optimizer_node, optimizer_edge,
                scheduler_node, scheduler_edge, device
            )
            train_epoch_loss += loss
        
        # node_model.eval()
        # edge_model.eval()
        # valid_epoch_loss = 0
        # for batch in valid_data_loader:
        #     loss = 0
        #     valid_rnn_step(num_nodes=5, node_model=node_model, edge_model=edge_model, input_size=config['data']['m'])
        #     valid_epoch_loss += loss
        
        # Save every X epochs
        if epoch % config['train']['checkpoint_epochs'] == 0:
            checkpoint_date = time.strftime("%d_%m_%Hh_%Mm")
            state = {
                'date': checkpoint_date,
                'epoch': epoch,
                'node_model_state_dict': node_model.state_dict(),
                'edge_model_state_dict': edge_model.state_dict(),
                'optimizer_node_state_dict': optimizer_node.state_dict(),
                'optimizer_edge_state_dict': optimizer_edge.state_dict(),
                'scheduler_node_state_dict': scheduler_node.state_dict(),
                'scheduler_edge_state_dict': scheduler_edge.state_dict(),
                'criterion': criterion.state_dict(),
            }
            torch.save(state, f"{config['train']['checkpoint_dir']}/checkpoint-date={checkpoint_date}-epoch={epoch}-tloss={round(train_epoch_loss, 2)}.pth")

        print(f"Epoch {epoch+1} | Training Loss: {train_epoch_loss} | Valid Loss: {None} | Total time {time.time() - start_time}.")
