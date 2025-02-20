import torch
import os
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, dense_to_sparse, to_dense_batch
import torch.optim as optim
# from model import Encoder, Decoder, Quantizer, Transformer2d


def init_model(config, data_info, device):
    encoder = Encoder(config, data_info).to(device)
    decoder = Decoder(config, data_info).to(device)
    quantizer = Quantizer(config).to(device)
    transformer = None
    if config.work_type == 'train_autoencoder':
        params = list(encoder.parameters()) + list(decoder.parameters()) + list(quantizer.parameters())
    elif config.work_type == 'train_prior':
        transformer = Transformer2d(config).to(device)
        params = transformer.parameters()
    elif config.work_type == 'sample':
        transformer = Transformer2d(config).to(device)
        opt = None
        scheduler = None

    if config.work_type != 'sample':
        opt = optim.Adam(params, lr=config.training.learning_rate, betas=config.training.betas)
        scheduler = optim.lr_scheduler.ExponentialLR(opt, config.training.lr_decay)

    if config.train_prior or config.sample:
        path = os.path.join('./wandb', config.autoencoder_path, 'files/best_run_val_rec_loss.pt')
        saved_model = torch.load(path, map_location=torch.device(device))
        encoder.load_state_dict(saved_model['encoder'])
        quantizer.load_state_dict(saved_model['quantizer'])
        decoder.load_state_dict(saved_model['decoder'])

        if config.work_type == 'sample':
            if config.dataset == 'zinc' or config.dataset == 'qm9':
                path = os.path.join('./wandb', config.model_folder, 'files/best_run_epoch_nspdk.pt')
            else:
                path = os.path.join('./wandb', config.model_folder, 'files/best_run_epoch_avg.pt')
            saved_model = torch.load(path, map_location=torch.device(device))
            transformer.load_state_dict(saved_model['transformer'])

    else:
        params = list(encoder.parameters()) \
                 + list(decoder.parameters()) \
                 + list(quantizer.parameters())
        opt = optim.Adam(params, lr=config.training.learning_rate, betas=config.training.betas)
    return encoder, decoder, quantizer, transformer, opt, scheduler

def get_features(adjs, moment = 3):
    n = adjs.shape[-1]
    device = adjs.device
    degrees = adjs.sum(-1)
    mask = (degrees.squeeze() != 0).float()
    node_feat = degrees.permute(0,2,1)
    edge_feat = adjs.permute(0,2,3,1)
    if moment > 1:
        Degrees = degrees.diag_embed(dim1=-2, dim2=-1)
        Degrees_minus1 = Degrees - torch.eye(n).to(device)
        adjs2 = adjs @ adjs
        adjs3 = adjs @ adjs2
        path3 = (adjs3 - adjs@Degrees - Degrees_minus1@adjs) * (1-torch.eye(n).to(device))
        path2 = adjs2 * (torch.ones(n, n).to(device)-torch.eye(n).to(device))
        degrees2 = path2.sum(-1)
        degrees3 = path3.sum(-1)

        node_feat = torch.cat((degrees, degrees2, degrees3), dim=1).permute(0,2,1)
        edge_feat = torch.cat((adjs, path2, path3), dim=1).permute(0,2,3,1)
    return node_feat, edge_feat, mask


def get_plt_fig(adjs, max_plot=9, title=None):
    n_plot = min(adjs.shape[0], max_plot)
    adjs_numpy = adjs.detach().cpu().numpy()
    fig, axes = plt.subplots(n_plot // 4, 4)
    ax = axes.flat
    n_plot = n_plot - (n_plot % 4)
    for i in range(n_plot):
        g = nx.from_numpy_array(adjs_numpy[i])
        g.remove_nodes_from(list(nx.isolates(g)))
        largest_component = max(nx.connected_components(g), key=len)
        g = g.subgraph(largest_component).copy()
        nx.draw(g, ax=ax[i], node_size=24)
    for a in ax:
        a.margins(0.10)
    fig.suptitle(title)
    return fig

def get_plt_fig_list(adjs, max_plot=16, title=None):
    n_plot = min(len(adjs), max_plot)
    fig, axes = plt.subplots(n_plot // 4, 4)
    ax = axes.flat
    for i, adj in enumerate(adjs):
        adj = adj[:,:,1:].sum(-1)
        g = nx.from_numpy_array(adj.detach().cpu().numpy())
        g.remove_nodes_from(list(nx.isolates(g)))
        nx.draw(g, ax=ax[i], node_size=4)
    for a in ax:
        a.margins(0.10)
    fig.suptitle(title)
    return fig

def plot_graphs(batch, max_plot=16, wandb=None, title=None):
    if isinstance(batch, Data):
        batch = to_dense_adj(batch.edge_index, batch.batch)

    fig = get_plt_fig(batch, max_plot=max_plot, title=title)
    fig.savefig("qm9_train.svg")
    fig.show()
    plt.show()


def plot_reconstructed(edge_rec, batch, graph_reconstructed):
    a = []
    r = []
    adjs = to_dense_adj(batch.edge_index, batch.batch)

    for rec, true, g_rec in zip(edge_rec, adjs, graph_reconstructed):
        if not g_rec:
            a.append(adjs)
            r.append(rec)
    fig = get_plt_fig_list(r, max_plot=32, title='Rec')
    fig.show()
    plt.show()
    fig = get_plt_fig_list(a, max_plot=32, title='True')
    fig.show()
    plt.show()


def dense_to_sparse_batch(nodes_rec, edges_rec, masks):
    batch_list = []
    for node, adj, mask in zip(nodes_rec, edges_rec, masks):
        adj = adj.argmax(-1)
        adj[(1 - mask).squeeze().bool()] = 0
        edge_index, edge_attr = dense_to_sparse(adj)
        data = Data(x=node, edge_index=edge_index, edge_attr=edge_attr)
        batch_list.append(data)
    return batch_list

def get_edge_masks(node_masks):
    device = node_masks.device
    n = node_masks.shape[1]
    batch_size = node_masks.shape[0]
    mask_reversed = (1 - node_masks.float())
    mask_reversed = mask_reversed.reshape(batch_size, -1, 1) + mask_reversed.reshape(batch_size, 1, -1)
    mask_reversed = mask_reversed + torch.eye(n).to(device)
    mask_reversed = (mask_reversed>0).float()
    masks = 1-mask_reversed
    return masks.unsqueeze(-1)

def discretize(score, masks=None):
    '''
    Args:
        score (tensor: batch size x # nodes x ... x #node types): A tensor containing the (log)
        probabilities (normalized or not) of each type on the last dim.
        Masks (tensor: bool, size as score), with True where there is a node/edge and False if
        the is no nodes/edges as this position.
        '''
    argmax = score.argmax(-1)
    device = score.device
    rec = torch.eye(score.shape[-1]).to(device)[argmax]
    if masks is not None:
        rec = rec * masks
    return rec

def get_sparse_graphs(annots, adjs):
    '''
    Args:
        Annots (Tensor: batch size x max_num_nodes x F nodes features)
        Adjs (Tensor: batch size x max_num_nodes x max_num_nodes x F edges features
    Return:
        List of graphs Data (torch_geometrc)
    '''
    graphs = []
    for annot, adj in zip(annots, adjs):
        if len(adj.shape)==3:
            adj_ = adj.sum(-1)!=0
        edge_index, _ = dense_to_sparse(adj_)
        edge_attr = adj[edge_index[0], edge_index[1]]
        data = Data(x=annot, edge_index=edge_index, edge_attr=edge_attr)
        graphs.append(data)
    return graphs

def dense_zq(zq, indices, quantizer, batch, max_node_num, sort):
        # Prepare indices and node masks
        device = zq.device
        indices, node_masks = to_dense_batch(indices, batch.batch, max_num_nodes=max_node_num)
        cb_size = quantizer.n_embeddings
        indices[~node_masks] = cb_size

        # Sort indices if needed
        if sort:
            indices = sort_indices(indices)

        # Convert indices to zq
        zq = quantizer.indices_to_zq(indices, padded=True)
        sz = indices.shape[1]
        masks = torch.triu(torch.full((sz, sz), float('-inf'), device=device), diagonal=1)
        return zq, masks, indices


def atom_number_to_one_hot(x, dataset):
    x = x[x > 0]
    if dataset == 'zinc':
        zinc250k_atomic_index = torch.tensor([0, 0, 0, 0, 0, 0, 1, 2, 3, 4,
                                              0, 0, 0, 0, 0, 5, 6, 7, 0, 0,
                                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                              0, 0, 0, 0, 0, 8, 0, 0, 0, 0,
                                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                              0, 0, 0, 9])  # 0, 6, 7, 8, 9, 15, 16, 17, 35, 53
        x = zinc250k_atomic_index[x] - 1  # 6, 7, 8, 9, 15, 16, 17, 35, 53 -> 0, 1, 2, 3, 4, 5, 6, 7, 8
        x = torch.eye(9)[x]
    else:
        x = torch.eye(4)[x - 6]
    return x


def from_dense_numpy_to_sparse(adj):
    adj = torch.from_numpy(adj)
    no_edge = 1 - adj.sum(0, keepdim=True)
    adj = torch.cat((no_edge, adj), dim=0)
    adj = adj.argmax(0)
    edge_index, edge_attr = dense_to_sparse(adj)
    edge_attr = torch.eye(3)[edge_attr - 1]
    return edge_index, edge_attr


def from_sparse_to_dense(batch, config):
    '''
    Agrs:
        - Batch of torch_geometric Data
    Return:
        - Tensor (n x n x p), n is # of nodes, p is size of edge attribute vector
    '''
    adjs = to_dense_adj(batch.edge_index, batch=batch.batch,
                        edge_attr=batch.edge_attr, max_num_nodes=config.data.max_node_num)
    no_edge = torch.zeros(*adjs.shape[:-1], 1)
    condition = (adjs.sum(-1) == 0).unsqueeze(-1)
    no_edge[condition] = 1
    adjs = torch.cat((no_edge, adjs), -1)
    return adjs


def sort_indices(indices):
    bs, n, nlv = indices.shape
    indices0 = torch.arange(bs).unsqueeze(1).repeat(1, indices.shape[1]).flatten()
    for i in reversed(range(nlv)):
        indices = indices[indices0, indices[:, :, i].sort(dim=1, stable=True)[1].flatten()]
        indices = indices.reshape(bs, n, nlv)
    return indices

def get_edge_target(batch):
    dense_edge_attr = to_dense_adj(batch.edge_index, batch=batch.batch, edge_attr=batch.edge_attr,
                                   max_num_nodes=batch.max_num_nodes[0])
    if len(dense_edge_attr.shape) == 3:
        return dense_edge_attr
    else:
        no_edge = 1 - dense_edge_attr.sum(-1, keepdim=True)
        dense_edge_attr = torch.cat((no_edge, dense_edge_attr), dim=-1)
        return dense_edge_attr.argmax(-1)

