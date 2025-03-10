import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from model import build_vqvgae
from data.dataset import get_dataset
import warnings

warnings.filterwarnings("ignore")


def load_vqvgae_model(checkpoint_path, config, device="cuda"):
    model = build_vqvgae(config, device, vqvgae_pretrain_path=checkpoint_path)
    model.eval()
    return model.to(device)


def get_latent_embeddings(vqvgae, dataloader, device="cuda"):
    vqvgae.eval()
    all_embeddings = []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            scales = config.vqvgae.quantizer.scales

            # Encoding
            init_graph_sizes = batch.batch.bincount()
            node_feat, _ = vqvgae.encoder(batch)

            # Downpooling
            node_feat = vqvgae.down_pooling(node_feat, edge_index=batch.edge_index, batch=batch.batch)
            node_feat = node_feat.view(len(init_graph_sizes), scales[-1], vqvgae.quantizer.embedding_dim)

            # Quantizing
            quantized_latents, _, _ = vqvgae.quantizer(node_feat)

            all_embeddings.append(quantized_latents.cpu())

    return torch.cat(all_embeddings, dim=0).numpy()



def visualize_latent_space(embeddings, method="tsne", save_path=None):
    num_graphs, scale_size, feature_dim = embeddings.shape
    embeddings_reshaped = embeddings.reshape(num_graphs * scale_size, feature_dim)

    if method == "pca":
        reducer = PCA(n_components=2)
        title = "PCA of Latent Embeddings"
    else:
        reducer = TSNE(n_components=2, perplexity=30, n_iter=300)
        title = "t-SNE of Latent Embeddings"

    reduced_embeds = reducer.fit_transform(embeddings_reshaped)

    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_embeds[:, 0], reduced_embeds[:, 1], alpha=0.7, s=5)
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    import yaml
    from easydict import EasyDict as edict

    config_path = "config.yaml"
    config = edict(yaml.load(open(config_path, "r"), Loader=yaml.FullLoader))

    if config.dataset.name == 'qm9':
        config.dataset.update(config.qm9)
    else:
        config.dataset.update(config.zinc)


    checkpoint_path = "checkpoints/VQ-VGAE_03_10_14_41_best.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vqvgae = load_vqvgae_model(checkpoint_path, config, device)

    train_loader, _ = get_dataset(
        root_dir=config.dataset.path, 
        dataset_name=config.dataset.name, 
        debug=config.dataset.debug, 
        batch_size=64,
        transforms=[]
    )

    embeddings = get_latent_embeddings(vqvgae, train_loader, device)
    # visualize_latent_space(embeddings, method="tsne", save_path="latent_space.png")
    visualize_latent_space(embeddings, method="tsne")
