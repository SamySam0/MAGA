import torch
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch


def permute_graph(data, perm):
    # Permute the node attribute ordering
    data.x = data.x[perm]
    data.z = data.z[perm]
    data.batch = data.batch[perm]

    # Permute the edge index
    adj = to_dense_adj(data.edge_index)
    adj = adj[:, perm, :]
    adj = adj[:, :, perm]
    data.edge_index = dense_to_sparse(adj)[0]
    return data


class PermutationEvaluation:
    def __init__(self, model, dataset):
        self.train_loader = DataLoader(dataset[:256], batch_size=1)
        self.model = model
    
    def is_encoder_permutation_equivariant(self):
        # Sample next batch from train loader
        data = next(iter(self.train_loader))

        # Forward pass on original example
        out_1, _ = self.model.encoder(data)

        # Permute the input data
        perm = torch.randperm(data.x.shape[0])
        data = permute_graph(data, perm)

        # Forward pass on permuted example
        out_2, _ = self.model.encoder(data)

        # Check whether output varies after applying transformations
        return torch.allclose(out_1[perm], out_2, atol=1e-03)
    
    def is_decoder_permutation_equivariant(self):
        # Sample next batch from train loader
        data = next(iter(self.train_loader))

        # Align data.x to embedding dim
        data.x = torch.cat([data.x, torch.rand(*data.x.shape[:-1], 5)], dim=-1)

        # Forward pass on original example
        quantized, node_masks = to_dense_batch(data.x, data.batch)
        nodes_recon_1, edges_recon_1 = self.model.decoder(quantized, mask=node_masks)

        # Create random permutation for each graph in the batch
        batch_size, max_nodes = quantized.size(0), quantized.size(1)
        batch_perm = torch.stack([torch.randperm(max_nodes) for _ in range(batch_size)])

        # Permute the quantized input
        quantized_permuted = torch.stack([quantized[b][batch_perm[b]] for b in range(batch_size)])
        node_masks_permuted = torch.stack([node_masks[b][batch_perm[b]] for b in range(batch_size)])

        # Forward pass on permuted example
        nodes_recon_2, edges_recon_2 = self.model.decoder(quantized_permuted, mask=node_masks_permuted)

        # Check node features equivariance
        is_node_equivariant = torch.allclose(
            torch.stack([nodes_recon_1[b][batch_perm[b]] for b in range(batch_size)]),
            nodes_recon_2,
            atol=1e-03,
        )

        # Check edge features equivariance
        is_edge_equivariant = torch.allclose(
            torch.stack([edges_recon_1[b][batch_perm[b]][:, batch_perm[b]] for b in range(batch_size)]),
            edges_recon_2,
            atol=1e-03,
        )

        return is_node_equivariant, is_edge_equivariant
    
    def is_model_permutation_equivariant(self):
        # Sample next batch
        data = next(iter(self.train_loader))
        
        # Forward pass on original example
        _, _, nodes_recon_1, edges_recon_1, _ = self.model(data)
        
        # Permute the input data
        perm = torch.randperm(data.x.shape[0])
        data = permute_graph(data, perm)
        
        # Forward pass on permuted example
        _, _, nodes_recon_2, edges_recon_2, _ = self.model(data)
        
        # Check node features equivariance
        is_node_equivariant = torch.allclose(
            nodes_recon_1[0][perm], 
            nodes_recon_2[0], 
            atol=1e-03
        )
        
        # Check edge features equivariance
        is_edge_equivariant = torch.allclose(
            edges_recon_1[0][perm][:, perm],
            edges_recon_2[0],
            atol=1e-03
        )
        
        return is_node_equivariant, is_edge_equivariant
    
    def evaluate(self):
        is_node_equivariant, is_edge_equivariant = self.is_model_permutation_equivariant()
        print(f'Is Model permutation-equivariant on node-features: {is_node_equivariant}')
        print(f'Is Model permutation-equivariant on edge-features: {is_edge_equivariant}')
