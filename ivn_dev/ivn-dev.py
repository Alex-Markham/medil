import os
import random
from collections import defaultdict
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Hyperparameters
latent_dims = (
    20  # actual number of latents in VAE (also number of epsilon/L in this case)
)
context_dims = (
    7  # number of interventions (needed for constructing the intervenable layer)
)
hidden_dims = 128  # same as hidden_dims in vanilla arch
batch_size = 512
learning_rate = 1e-3
epochs = 100
append_path = "-dev"


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
        image = torch.tensor(nump[:-1].reshape(1, 28, 28), dtype=torch.float32)
        label = int(nump[-1])
        return image, label


dataset = IvnDataset("mnist_images_concat5000.csv")

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
train_loader = DataLoader(dataset, batch_sampler=sampler)


class Intervenable(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        width=1,
        mask=None,
        bias=True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.width = width
        self.mask = mask
        self.weight = Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )

        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.weight)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / torch.sqrt(torch.tensor(fan_in)) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input, obs_weight, interv_idx):
        if interv_idx != -1:
            min_weight = torch.minimum(self.weight, obs_weight)
            num_vars = len(self.weight) // self.width
            interv_mask = torch.ones(num_vars, num_vars)
            interv_mask[interv_idx] = 0
            interv_mask[interv_idx, interv_idx] = 1
            interv_mask = interv_mask.kron(torch.ones(self.width, self.width))
            self.weight.data = min_weight * interv_mask
        if self.mask is None:
            return nn.functional.linear(input, self.weight, self.bias)
        else:
            return nn.functional.linear(input, self.weight * self.mask, self.bias)

    def extra_repr(self):
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1),  # 14x14
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 7x7
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 7),  # 1x1
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128, hidden_dims),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(hidden_dims, latent_dims)
        self.fc_var = nn.Linear(hidden_dims, latent_dims)

        self.causal_layer = nn.ModuleDict(
            {
                str(interv_idx): Intervenable(
                    in_features=latent_dims, out_features=latent_dims, device=device
                )
                for interv_idx in chain((-1,), range(1, context_dims))
            }
        )
        # Decoder
        self.decoder_linear = nn.Sequential(
            nn.Linear(latent_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, 128),
            nn.ReLU(),
        )

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 7),  # 7x7
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 14x14
            nn.BatchNorm2d(32),
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
        obs_weight = self.causal_layer[str(-1)].weight
        h = self.causal_layer[str(self.batch_label)](z, obs_weight, self.batch_label)
        h = self.decoder_linear(h)
        h = h.view(-1, 128, 1, 1)
        return self.decoder_conv(h)

    def forward(self, x, label):
        self.batch_label = label
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var


def loss_function(recon_x, x, mu, logvar, causal_weights):
    BCE = F.binary_cross_entropy(recon_x, x, reduction="sum")

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    diag = torch.diag(causal_weights)

    sparse_reg = (causal_weights - diag).pow(2).mean()

    return BCE + KLD + 1000 * sparse_reg


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


##### Plot code
def plot_latent_traversal():
    model.eval()

    selected_dims = [i for i in range(20)]
    # Create figure with 8 rows and 10 columns
    fig, axs = plt.subplots(20, 10, figsize=(20, 16))

    # Create base latent vector
    z_base = torch.zeros(1, latent_dims).to(device)

    # Values to traverse for each dimension
    traverse_values = torch.linspace(-3, 3, 10)

    for i, row in enumerate(selected_dims):
        for col in range(10):
            z = z_base.clone()
            z[0, row] = traverse_values[col]

            with torch.no_grad():
                img = model.decode(z).cpu().view(28, 28)

            axs[i, col].imshow(img, cmap="gray")
            axs[i, col].axis("off")

            # Add labels only for first and last column
            # if col == 0:
            #     axs[i, col].set_title(f"-3")
            # elif col == 9:
            #     axs[i, col].set_title(f"3")

    plt.tight_layout()
    plt.savefig(f"ivn-latent_traversal{append_path}.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_reconstructions():
    model.eval()
    with torch.no_grad():
        data, labels = next(iter(train_loader))[:8]
        data = data.to(device)
        label = torch.unique(labels).to(int)
        assert len(label) == 1
        label = int(label)
        recon, _, _ = model(data, label)

    fig, axes = plt.subplots(2, 8, figsize=(15, 4))
    for i in range(8):
        axes[0, i].imshow(data[i].cpu().squeeze(), cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(recon[i].cpu().reshape(28, 28), cmap="gray")
        axes[1, i].axis("off")

    axes[0, 0].set_title("Original")
    axes[1, 0].set_title("Reconstructed")
    plt.tight_layout()
    plt.savefig(f"ivn-reconstructions{append_path}.png")
    plt.close()


def plot_training_loss():
    plt.figure()
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(f"ivn-training_loss{append_path}.png")
    plt.close()


def plot_random_samples(num_samples=8):
    model.eval()
    with torch.no_grad():
        # Sample from standard normal distribution
        z = torch.randn(num_samples, latent_dims).to(device)
        # Decode latent vectors
        samples = model.decode(z)

    # Plot samples
    fig, axes = plt.subplots(2, 4, figsize=(8, 4))
    for i, ax in enumerate(axes.flat):
        ax.imshow(samples[i].cpu().reshape(28, 28), cmap="gray")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(f"ivn-random_samples{append_path}.png")
    plt.close()


def plot_causal():
    to_plot = model.causal_layer.weight.detach().cpu().numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot the heatmap
    sns.heatmap(np.abs(to_plot), ax=ax1, cmap="viridis")
    ax1.set_title("DAG adjacency")

    # Plot the histogram
    ax2.hist(to_plot.flatten())
    ax2.set_title("Weight distribution")

    plt.tight_layout()
    plt.savefig(f"ivn-causal{append_path}.png")
    plt.close()


#### Training loop
def train():
    model.train()
    train_loss = 0
    pbar = tqdm(enumerate(train_loader), desc="training epoch...", unit="batch")
    for batch_idx, (data, labels) in pbar:
        data = data.to(device)
        label = torch.unique(labels).to(int)
        assert len(label) == 1
        label = int(label)
        optimizer.zero_grad()
        recon_batch, mu, log_var = model(data, label)
        causal_weights_batch = model.causal_layer[str(label)].weight
        loss = loss_function(recon_batch, data, mu, log_var, causal_weights_batch)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        pbar.set_postfix({"loss": f"{train_loss / (batch_idx + 1):.4f}"})
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
        dir_path = f"ivn-vae_mnist_checkpoints{append_path}"
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
                "losses": losses,
            },
            f"{dir_path}/epoch_{epoch}.pth",
        )
        plot_reconstructions()
        plot_random_samples()
        plot_latent_traversal()
        # plot_causal()

    # Save model and training losses
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "losses": losses,
        },
        f"ivn-vae_mnist{append_path}.pth",
    )


train_model()

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE().to(device)
checkpoint = torch.load(f"ivn-vae_mnist{append_path}.pth", weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
losses = checkpoint["losses"]


# Generate all visualizations
plot_training_loss()
plot_reconstructions()
plot_random_samples()
plot_latent_traversal()
plot_causal()
