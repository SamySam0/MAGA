import torch
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
import math


def load_qm9_data(root, batch_size, transforms, num_workers=0,
                  train_val_test_split=[0.8, 0.1, 0.1],
                  dataset_size=None):
   # Load the full dataset
   assert isinstance(transforms, list), 'transforms should be a list'
   dataset = QM9(root=root, transform=Compose(transforms))
  
   # Shuffle the dataset
   dataset = dataset.shuffle()
  
   # Optionally restrict to a given number of observations.
   if dataset_size is not None:
       # In case dataset_size exceeds the available number of samples.
       dataset = dataset[:min(dataset_size, len(dataset))]

   n_total = len(dataset)
   n_train = math.floor(n_total * train_val_test_split[0])
   n_val   = math.floor(n_total * train_val_test_split[1])
   n_test  = n_total - n_train - n_val

   train_dataset = dataset[:n_train]
   val_dataset   = dataset[n_train:n_train+n_val]
   test_dataset  = dataset[n_train+n_val:]

   train_loader = DataLoader(train_dataset, batch_size=batch_size,
                             shuffle=True, num_workers=num_workers)
   val_loader   = DataLoader(val_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)
   test_loader  = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)

   return train_loader, val_loader, test_loader
