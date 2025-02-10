from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader

def get_dataloaders(config):
    # Load the QM9 dataset with some transforms
    dataset = QM9(config.data.path, transform=None)

    # Split dataset
    train_split = config.data.train_split
    val_split   = train_split + config.data.valid_split
    test_split  = val_split + config.data.test_split
    train_pyg_dataset = dataset[:int(len(dataset)*train_split)]
    val_pyg_dataset   = dataset[int(len(dataset)*train_split):int(len(dataset)*val_split)]
    test_pyg_dataset  = dataset[int(len(dataset)*val_split):int(len(dataset)*test_split)]

    # Create dataloaders with batch size
    train_loader = DataLoader(train_pyg_dataset, batch_size=config.train.batch_size, shuffle=True)
    val_loader   = DataLoader(val_pyg_dataset, batch_size=config.train.batch_size, shuffle=False)
    test_loader  = DataLoader(test_pyg_dataset, batch_size=config.train.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
