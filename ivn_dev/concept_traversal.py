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

assert torch.cuda.is_available(), "This script requires a GPU to run!"
torch.set_default_tensor_type(
    "torch.cuda.FloatTensor"
)  # Set default tensor type to CUDA
torch.set_default_dtype(
    torch.float32
)  # If you also need to handle other tensor types, you can set their defaults too

# Hyperparameters
context_dims = (
    7  # number of interventions + obs (needed for constructing the intervenable layer)
)
width = 4  # >=1; width/degrees of freedom/num neurons per context in the block weight matrix
depth = 1  # >=0; w>1 requires d>0; w=1 & d=0 implies Z ≡ ε; number of hidden layers and activations between Z and ε
latent_dims = (
    context_dims
    * width  # actual number of latents in VAE (also number of epsilon/L in this case)
)
hidden_dims = 128  # same as hidden_dims in vanilla arch
batch_size = 512
learning_rate = 1e-3
epochs = 200
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


class BlockLinear(nn.Module):
    def __init__(
        self,
        context_dims,
        width,
        bias=True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.block_mask = torch.eye(context_dims).kron(torch.ones(width, width))
        num_features = context_dims * width
        self.weight = Parameter(
            torch.empty((num_features, num_features), **factory_kwargs)
        )
        if bias:
            self.bias = Parameter(torch.empty(num_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        # nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.orthogonal_(self.weight)
        # nn.init.sparse_(self.weight, 2 / 3)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / torch.sqrt(torch.tensor(fan_in)) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        # masked linear layer
        return nn.functional.linear(input, self.weight * self.block_mask, self.bias)

    def extra_repr(self):
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )


class Intervenable(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        mask=None,
        bias=True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
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
            num_vars = len(self.weight)
            interv_mask = torch.ones(num_vars, num_vars)
            interv_mask[interv_idx] = 0
            interv_mask[interv_idx, interv_idx] = 1
            self.weight.data = min_weight * interv_mask.to(device)
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

        # Our module
        unchained = BlockLinear(context_dims, width), nn.GELU()
        deeply_expressive = chain(*(unchained for _ in range(depth)))
        self.expressive_layer = nn.Sequential(*deeply_expressive, nn.AvgPool1d(width))
        self.causal_layer = nn.ModuleDict(
            {
                str(interv_idx): Intervenable(
                    in_features=context_dims, out_features=context_dims, device=device
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
        epsilon = self.expressive_layer(z)
        obs_weight = self.causal_layer[str(-1)].weight
        l = self.causal_layer[str(self.batch_label)](
            epsilon, obs_weight, self.batch_label
        )
        h = l.kron(torch.ones(width))
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
    sparse_reg = causal_weights.pow(2).mean()
    return BCE + KLD + 1000 * sparse_reg


##### Plot code
###############


def _decode_eps(model, eps_i, ivn):
    # take a random sample of eps_i as input, fix all other eps_j
    fixed_eps = torch.zeros(concept_dims)  # concept_dims globally defined at top
    fixed_eps[ivn] = eps_i

    # pass through causal layer ivn and blackbox decoder to get reconstruction
    obs_weight = model.causal_layer[str(-1)].weight
    l = model.causal_layer[str(ivn)](fixed_eps, obs_weight, ivn)
    h = l.kron(torch.ones(width))
    h = model.decoder_linear(h)
    h = h.view(-1, 128, 1, 1)
    return model.decoder_conv(h)


def plot_concept_traversal(model):
    # un/comment dict items below to in/exclude from plot
    concept_dict = {
        -1: "raw",
        0: "free",
        1: "scaled",
        2: "shear",
        3: "shift",
        4: "swel",
        5: "thic",
        6: "thin",
    }

    model.eval()

    traversed_dims = len(concept_dict)
    # Create figure with 1 row per concept and 10 columns
    fig, axs = plt.subplots(traversed_dims, 10, figsize=(20, 16))

    # Values to traverse for each concept
    traverse_values = torch.linspace(-3, 3, 10)

    for i, row in enumerate(concept_dict.keys()):
        for col in range(10):
            eps_i = traverse_values[col]

            with torch.no_grad():
                img = _decode_eps(model, eps_i, row).cpu().view(28, 28)

            axs[i, col].imshow(img, cmap="gray")
            axs[i, col].axis("off")

            # Add labels only for first and last column
            if col == 0:
                axs[i, col].set_title(f"-3")
            elif col == 9:
                axs[i, col].set_title(f"3")
            elif col == 4:
                axs[i, col].set_title(f"{concept_dict[row]}")
    plt.tight_layout()
    plt.savefig(f"concept_traversal.png", dpi=150, bbox_inches="tight")
    plt.close()


# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE().to(device)
checkpoint = torch.load(f"ivn-vae_mnist{append_path}.pth", weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
losses = checkpoint["losses"]


# Generate plots
plot_concept_traversal(model)
