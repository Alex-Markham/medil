import random
from collections import defaultdict
import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Hyperparameters
latent_dims = 20
hidden_dims = 400
batch_size = 128
learning_rate = 1e-3
epochs = 100
append_path = ""


class IvnDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        with open(self.data_path, "r") as f:
            self.raw_lines = f.readlines()

    def __len__(self):
        return len(self.raw_lines)

    def __getitem__(self, idx):
        string = self.raw_lines[idx][:-1]
        nump = np.fromstring(string, sep=",")
        image = torch.tensor(nump[:-1].reshape(28, 28), dtype=torch.float32)
        label = int(nump[-1])
        return image, label


# dataset = IvnDataset("test_ivn.csv")
dataset = IvnDataset("mnist_images_concat.csv")

# train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
# check why `train_loader.dataset[400000]` appears to be all 0s!!!

# Custom Sampler for grouping by label
class SameLabelBatchSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

        # Group indices by label
        self.label_to_indices = defaultdict(list)
        for idx in range(len(dataset)):
            _, label = dataset[idx]
            self.label_to_indices[label].append(idx)

        # Create batches for each label
        self.batches = []
        for label, indices in self.label_to_indices.items():
            random.shuffle(indices)  # Shuffle indices for randomness
            # Split indices into batches of size `batch_size`
            for i in range(0, len(indices), batch_size):
                self.batches.append(indices[i : i + batch_size])

        random.shuffle(self.batches)  # Shuffle the order of batches

    def __iter__(self):
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)


sampler = SameLabelBatchSampler(dataset, batch_size)
trainloader = DataLoader(dataset, batch_sampler=sampler)

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1),  # 14x14
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 7x7
            nn.ReLU(),
            nn.Conv2d(64, 128, 7),  # 1x1
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128, hidden_dims),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(hidden_dims, latent_dims)
        self.fc_var = nn.Linear(hidden_dims, latent_dims)

        self.causal_layer = nn.Linear(latent_dims, latent_dims) # use self.batch_label here; define a causal_layer class
        # Decoder
        self.decoder_linear = nn.Sequential(
            self.causal_layer,
            nn.Linear(latent_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, 128),
            nn.ReLU(),
        )

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 7),  # 7x7
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),  # 28x28
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_linear(z)
        h = h.view(-1, 128, 1, 1)
        return self.decoder_conv(h)

    def forward(self, x, label):
        self.batch_label = label
        print(batch_label)
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var


def loss_function(recon_x, x, mu, logvar, causal_weights, pen):
    BCE = F.binary_cross_entropy(recon_x, x, reduction="sum")

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    causal_weights = torch.abs(causal_weights)
    diag = torch.diag(causal_weights)

    sparse_reg = causal_weights.pow(2).mean()

    s = torch.tensor([5])
    d = latent_dims
    dag_reg = -torch.logdet(
        s * torch.eye(d) - torch.square(causal_weights - diag)
    ) + d * torch.log(s)

    return BCE + KLD + 1000 * sparse_reg


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def train():
    model.train()
    train_loss = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        data = data.to(device)
        labels = labels[0]         # fix
        optimizer.zero_grad()
        recon_batch, mu, log_var = model(data,label)
        causal_weights_batch = model.causal_layer.weight
        loss = loss_function(recon_batch, data, mu, log_var, causal_weights_batch)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    return train_loss / len(train_loader.dataset)


# Train the model
def train_model():
    losses = []
    pbar = tqdm(range(epochs), desc="Training...", unit="epoch")

    for epoch in pbar:
        loss = train()
        losses.append(loss)
        pbar.set_postfix({"loss": f"{loss:.4f}"})

        # Save checkpoint after each epoch
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
                "losses": losses,
            },
            f"conv-vae_mnist_checkpoint_epoch_{epoch}{append_path}.pth",
        )

    # Save model and training losses
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "losses": losses,
        },
        f"conv-vae_mnist{append_path}.pth",
    )


train_model()

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE().to(device)
checkpoint = torch.load(f"conv-vae_mnist{append_path}.pth", weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
losses = checkpoint["losses"]


# def plot_latent_traversal():
#     model.eval()

#     selected_dims = [i for i in range(20)]
#     # Create figure with 8 rows and 10 columns
#     fig, axs = plt.subplots(20, 10, figsize=(20, 16))

#     # Create base latent vector
#     z_base = torch.zeros(1, latent_dims).to(device)

#     # Values to traverse for each dimension
#     traverse_values = torch.linspace(-3, 3, 10)

#     for i, row in enumerate(selected_dims):
#         for col in range(10):
#             z = z_base.clone()
#             z[0, row] = traverse_values[col]

#             with torch.no_grad():
#                 img = model.decode(z).cpu().view(28, 28)

#             axs[i, col].imshow(img, cmap="gray")
#             axs[i, col].axis("off")

#             # Add labels only for first and last column
#             # if col == 0:
#             #     axs[i, col].set_title(f"-3")
#             # elif col == 9:
#             #     axs[i, col].set_title(f"3")

#     plt.tight_layout()
#     plt.savefig(f"conv-latent_traversal{append_path}.png", dpi=150, bbox_inches="tight")
#     plt.close()


# def plot_reconstructions():
#     model.eval()
#     with torch.no_grad():
#         data = next(iter(train_loader))[0][:8].to(device)
#         recon, _, _ = model(data)

#     fig, axes = plt.subplots(2, 8, figsize=(15, 4))
#     for i in range(8):
#         axes[0, i].imshow(data[i].cpu().squeeze(), cmap="gray")
#         axes[0, i].axis("off")
#         axes[1, i].imshow(recon[i].cpu().reshape(28, 28), cmap="gray")
#         axes[1, i].axis("off")

#     axes[0, 0].set_title("Original")
#     axes[1, 0].set_title("Reconstructed")
#     plt.tight_layout()
#     plt.savefig(f"conv-reconstructions{append_path}.png")
#     plt.close()


# def plot_training_loss():
#     plt.figure()
#     plt.plot(losses)
#     plt.title("Training Loss")
#     plt.xlabel("Epoch")
#     plt.ylabel("Loss")
#     plt.savefig(f"conv-training_loss{append_path}.png")
#     plt.close()


# def plot_random_samples(num_samples=8):
#     model.eval()
#     with torch.no_grad():
#         # Sample from standard normal distribution
#         z = torch.randn(num_samples, latent_dims).to(device)
#         # Decode latent vectors
#         samples = model.decode(z)

#     # Plot samples
#     fig, axes = plt.subplots(2, 4, figsize=(8, 4))
#     for i, ax in enumerate(axes.flat):
#         ax.imshow(samples[i].cpu().reshape(28, 28), cmap="gray")
#         ax.axis("off")

#     plt.tight_layout()
#     plt.savefig(f"conv-random_samples{append_path}.png")
#     plt.close()


# def plot_causal():
#     to_plot = model.causal_layer.weight.detach().numpy()

#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

#     # Plot the heatmap
#     sns.heatmap(np.abs(to_plot), ax=ax1, cmap="viridis")
#     ax1.set_title("DAG adjacency")

#     # Plot the histogram
#     ax2.hist(to_plot.flatten())
#     ax2.set_title("Weight distribution")

#     plt.tight_layout()
#     plt.savefig(f"conv-causal{append_path}.png")
#     plt.close()


# # Generate all visualizations
# plot_training_loss()
# plot_reconstructions()
# plot_random_samples()
# plot_latent_traversal()
# plot_causal()


# def find_dag():
#     w = model.causal_layer.weight.detach().numpy()
#     for thresh in np.linspace(0, np.abs(w).max(), 10):
#         adj = np.abs(w) > thresh
#         dag = nx.DiGraph(adj)
#         if nx.is_directed_acyclic_graph(dag):
#             return dag, adj, thresh
#     print(":(")


# dag, adj, thresh = find_dag()
# s = torch.tensor([5])
# d = latent_dims
# causal_weights = torch.abs(model.causal_layer.weight.detach())
# diag = torch.diag(causal_weights)
# dag_reg = -torch.logdet(
#     s * torch.eye(d) - torch.square(causal_weights - diag)
# ) + d * torch.log(s)

# print(f"dagness: {dag_reg}")
# print(adj.sum())
