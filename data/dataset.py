import os, json, torch
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from data.loader import KekulizedMolDataset


def get_dataset(root_dir, dataset_name, debug, batch_size, transforms):
    # Create the dataset_name with the Spectral Feat transforms
    data = KekulizedMolDataset(root_dir, pre_transform=Compose(transforms), dataset=dataset_name)

    # Load the test indices from the corresponding file
    train_idx, test_idx = get_indices(root_dir, dataset_name, len(data), debug)

    # Create DataLoaders for training and test sets
    train_loader = DataLoader(
        data[train_idx],
        batch_size=batch_size,
        shuffle=True, drop_last=True,
    )
    test_loader = DataLoader(
        data[test_idx], 
        batch_size=1024, 
        shuffle=True, drop_last=True, 
    )

    return train_loader, test_loader


def get_indices(root_dir, dataset_name, n_instances, debug):
    with open(os.path.join(root_dir, f'valid_idx_{dataset_name}.json')) as f:
        test_idx = json.load(f)
        if dataset_name == 'qm9':
            test_idx = test_idx['valid_idxs']
        test_idx = [int(i) for i in test_idx]

    # Create a boolean mask for the training indices
    train_idx = torch.ones(n_instances).bool()
    train_idx[test_idx] = False
    train_idx = train_idx[train_idx]

    if debug:
        train_idx = train_idx[:30_000]
        test_idx = test_idx[:1024*10]

    return train_idx, test_idx
