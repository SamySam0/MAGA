import os, torch
import numpy as np
from torch_geometric.data import Data, InMemoryDataset, download_url
from torch_geometric.utils import dense_to_sparse


class KekulizedMolDataset(InMemoryDataset):
    def __init__(self, root, dataset=None, transform=None, pre_transform=None, pre_filter=None):
        self.dataset = dataset
        super().__init__(root, transform, pre_transform, pre_filter)
        torch.load(self.processed_paths[0], weights_only=False)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        if self.dataset == 'zinc':
            return ['zinc250k_kekulized.npz']
        elif self.dataset == 'qm9':
            return ['qm9_kekulized.npz']
        else:
            raise NotImplementedError()

    @property
    def processed_file_names(self):
        if self.dataset == 'zinc':
            return ['zinc_data.pt']
        elif self.dataset == 'qm9':
            return ['data_qm9.pt']

    def download(self):
        # Download to `self.raw_dir`.
        if self.dataset == 'zinc':
            download_url('https://drive.switch.ch/index.php/s/D8ilMxpcXNHtVUb/download', self.raw_dir,
                         filename='zinc250k_kekulized.npz')
        elif self.dataset == 'qm9':
            download_url('https://drive.switch.ch/index.php/s/SESlx1ylQAopXsi/download', self.raw_dir,
                         filename='qm9_kekulized.npz')

    def process(self):
        if self.dataset == 'zinc':
            filepath = os.path.join(self.raw_dir, 'zinc250k_kekulized.npz')
            max_num_nodes = 38
        elif self.dataset == 'qm9':
            filepath = os.path.join(self.raw_dir, 'qm9_kekulized.npz')
            max_num_nodes = 9

        load_data = np.load(filepath)
        xs, adjs = load_data['arr_0'], load_data['arr_1']
        
        load_data = 0
        data_list = []
        for x, adj in zip(xs, adjs):
            x = atom_number_to_one_hot(x, self.dataset)
            edge_index, edge_attr = from_dense_numpy_to_sparse(adj)
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, max_num_nodes=max_num_nodes)
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)

        print(f'{len(data_list)} graphs processed')
        data, slices = self.collate(data_list)
        del data_list
        print('Data collated!')
        torch.save((data, slices), self.processed_paths[0])


def from_dense_numpy_to_sparse(adj):
    adj = torch.from_numpy(adj)
    no_edge = 1 - adj.sum(0, keepdim=True)
    adj = torch.cat((no_edge, adj), dim=0)
    adj = adj.argmax(0)
    edge_index, edge_attr = dense_to_sparse(adj)
    edge_attr = torch.eye(3)[edge_attr - 1]
    return edge_index, edge_attr

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
