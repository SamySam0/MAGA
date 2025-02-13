import torch
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_dense_adj, get_laplacian


class AddSpectralFeat(BaseTransform):
    ''' 
    Add edge_index_ext and edge_attr_ext as object attribute and fill it with edge_index and edge_attr.
    '''
    def __init__(self):
        super().__init__()

    def __call__(self, data):
        lap,_ = get_laplacian(data.edge_index)
        lap = to_dense_adj(lap).squeeze()
        eigvals, eigvectors = torch.linalg.eigh(lap)
        K = 5
        eigfeat = eigvectors[..., :K]
        if eigfeat.shape[-1] < K:
            missing = K - eigfeat.shape[-1]
            if data.x.shape[0] == 1:
                eigfeat = torch.zeros(1, 5)
            else:
                eigfeat = torch.cat((eigfeat, torch.zeros(data.x.shape[0], missing)), dim=-1)

        if data.x is not None:
            data.x = torch.cat((data.x, eigfeat), dim=-1)
        else:
            data.x = eigfeat
        return data
