"""For simulating and learning functional MeDIL Causal Models"""
import torch
import torch.nn as nn
import torch.distributions as dist
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader
import numpy as np
from collections import OrderedDict


def gaussian_mixture_sampler(
    num_latent, num_mixtures=4, weights=None, means=None, cov=None
):
    """

    :param num_latent:
    :param num_mixtures:
    :param weights:
    :param means:
    :param cov:
    :return:
    """

    if weights is None:
        weights = torch.randn(num_latent, num_mixtures).softmax(dim=1)

    if means is None:
        means = torch.randn(num_latent, num_mixtures, 1) * 2

    if cov is None:
        cov = torch.randn(num_latent, num_mixtures, 1)

    mix = dist.Categorical(weights)
    comp = dist.Independent(dist.Normal(means, cov), 1)

    gmm = dist.MixtureSameFamily(mix, comp)

    return lambda n: gmm.sample((n,)).squeeze()


def uniform_sampler(num_latent, low=-1, high=1):
    """

    :param num_latent:
    :param min:
    :param max:
    :return:
    """

    return lambda n: torch.empty((n, num_latent)).uniform_(low, high)


class NNMechanism(nn.Module):
    def __init__(
        self,
        num_causes,
        num_hidden_layers=1,
        num_hidden_units=20,
        non_linearity="tanh",
        output=None,
        noise_function=None,
        noise_coeff=1.0,
    ):
        """

        :param num_causes:
        :param num_hidden_layers:
        :param num_hidden_units:
        :param activation:
        :param output:
        :param noise_function:
        :param noise_apply:
        """

        super().__init__()
        self.num_causes = num_causes

        # Init Layer list.
        layers = []

        # Input Layer
        layers.append(nn.Linear(self.num_causes + 1, num_hidden_units))

        # Select Activation function
        activation = nn.Identity

        if non_linearity is "tanh":
            activation = nn.Tanh
        elif non_linearity is "relu":
            activation = nn.ReLU
        elif non_linearity is "sigmoid":
            activation = nn.Sigmoid
        elif non_linearity is "gelu":
            activation = nn.GELU

        # Initialise Layers
        for idx in range(num_hidden_layers):
            layers.append(activation())
            layers.append(nn.Linear(num_hidden_units, num_hidden_units))

        # Output Layer
        final = nn.Identity
        if output is "tanh":
            final = nn.Tanh
        elif output is "sigmoid":
            final = nn.Sigmoid

        layers.append(activation())
        layers.append(nn.Linear(num_hidden_units, 1))
        layers.append(final())

        # Assign Layers to Causal Function
        self.causal_function = nn.Sequential(*layers)

        # Horrible shameful hack
        # final_layer_var = 5.0
        # nn.init.uniform_(layers[-2].weight, -final_layer_var, final_layer_var)
        # nn.init.uniform_(layers[-2].bias, -final_layer_var, final_layer_var)
        # print('final', layers[-2].weight, 'pre-final',layers[-4].weight)

        # Exogenous Noise function
        self.noise_function = lambda n: dist.Normal(0, 0.1).sample((n, 1))

        if noise_function is not None:
            self.noise_function = noise_function

        # Exogenous Noise Coefficient
        self.noise_coeff = noise_coeff

    def forward(self, x):
        """

        :param x:
        :return:
        """
        # Sample Exogenous noise corresponding to variable.
        noise = self.noise_coeff * self.noise_function(x.shape[0])

        # Concatenate with input
        causes = torch.cat((noise, x), axis=1)

        # Compute Output
        output = self.causal_function(causes)

        return output


class MeDILCausalModel(nn.Module):
    def __init__(
        self,
        biadj_mat,
        num_hidden_layers=1,
        num_hidden_units=20,
        non_linearity="tanh",
        output="none",
        noise_function=None,
        noise_coeff=1.0,
        debug=False,
    ):

        super().__init__()

        self.biadj_mat = biadj_mat
        self.debug = debug

        self.hparams = {
            "num_hidden_layers": num_hidden_layers,
            "num_hidden_units": num_hidden_units,
            "non_linearity": non_linearity,
            "output": output,
            "noise_coeff": noise_coeff,
        }

        self.num_latent = biadj_mat.shape[0]
        self.num_obs = biadj_mat.shape[1]

        self.subset_mask = [
            np.argwhere(self.biadj_mat[:, idx]).squeeze(axis=-1)
            for idx in range(self.num_obs)
        ]

        print(
            "Generator initialised. #latent: {}, #observed: {}".format(
                self.num_latent, self.num_obs
            )
        )

        forward_dict = {}

        for idx in range(self.num_obs):

            num_latent_causes = np.sum(self.biadj_mat[:, idx].astype(int))

            mechanism = NNMechanism(
                num_latent_causes,
                num_hidden_layers,
                num_hidden_units,
                non_linearity,
                output,
                noise_function,
                noise_coeff,
            )

            forward_dict[str(idx)] = mechanism

        if self.debug:
            print("blah")
        self.observed = nn.ModuleDict(forward_dict)

        if self.debug:
            print("blah")

    def forward(self, x):

        output = []

        if self.debug:
            print("input", x.shape)
        for idx in range(self.num_obs):

            if self.debug:
                print("obs" + str(idx))
            input_subset_mask = np.argwhere(self.biadj_mat[:, idx])

            if self.debug:
                print(input_subset_mask.squeeze())
                print("pre", self.subset_mask[idx])
            input_subset = x[:, self.subset_mask[idx]]

            if self.debug:
                print("input_subset", input_subset.shape)
                print(list(self.observed[str(idx)].modules()))
            obs = self.observed[str(idx)](input_subset)

            if self.debug:
                print("obs", obs.shape)
            output.append(obs)

        output = torch.cat(output, axis=1)

        return output

    def sample(self, sampler, num_samples=100):
        """

        :param sampler:
        :param num_samples:
        :return:
        """

        input_sample = sampler(num_samples)

        self.eval()
        with torch.no_grad():
            output_sample = self.forward(input_sample)

        return input_sample, output_sample

    def set_noise_coeff(self, value):
        """

        :param value:
        :return:
        """

        for module in self.observed:
            # print('before',self.observed[module].noise_coeff)
            self.observed[module].noise_coeff = value
            # print('after',self.observed[module].noise_coeff)

    def interpolate(self, num_samples=100):
        """

        :param num_samples:
        :return:
        """

        self.eval()
        self.set_noise_coeff(0.0)

        output_sample_list = list()

        for latent_idx in range(self.num_latent):

            latent_sample = torch.zeros((num_samples, self.num_latent))
            latent_sample[:, latent_idx] = torch.arange(-10.0, 10.0, 20.0 / num_samples)

            output_sample = self.forward(latent_sample)

            output_sample_list.append(output_sample)

        self.set_noise_coeff(1.0)

        return output_sample_list


class GAN(LightningModule):
    def __init__(self, data_filepath, decoder, latent_sampler=None, batch_size=256):
        """

        :param data_filepath:
        :param decoder:
        :param latent_sampler:
        :param batch_size:
        """

        super().__init__()

        self.hparams = decoder.hparams
        self.hparams["batch_size"] = batch_size
        print(self.hparams)

        self.data_filepath = data_filepath
        self.decoder = decoder
        self.latent_sampler = latent_sampler
        self.batch_size = batch_size

        self.num_latent = self.decoder.num_latent
        self.num_obs = self.decoder.num_obs

    def forward(self, z):

        return self.decoder(z)

    def mmd_loss(self, x_hat_batch, x_batch, alpha=1):

        return MMDLoss(x_batch.shape[0])(x_hat_batch, x_batch)

    def train_dataloader(self):

        dataset = np.load(self.data_filepath).astype(np.float32)

        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    # Train Functions
    def training_step(self, batch, batch_idx):

        sample = batch

        z = self.latent_sampler(batch.shape[0])

        x_hat = self.forward(z)

        loss = self.mmd_loss(x_hat, batch)

        tqdm_dict = {"loss": loss}
        output = OrderedDict(
            {
                "loss": loss,
                # 'progress_bar': tqdm_dict,
                "log": tqdm_dict,
            }
        )

        self.logger.log_metrics({"mmd_loss": loss.item()}, batch_idx)

        return output


class MMDLoss(nn.Module):
    def __init__(self, input_size, bandwidths=None):
        """Init the model."""
        super(MMDLoss, self).__init__()
        if bandwidths is None:
            bandwidths = torch.Tensor([0.01, 0.1, 1, 10, 100])
        else:
            bandwidths = bandwidths
        s = torch.cat(
            [
                torch.ones([input_size, 1]) / input_size,
                torch.ones([input_size, 1]) / -input_size,
            ],
            0,
        )

        self.register_buffer("bandwidths", bandwidths.unsqueeze(0).unsqueeze(0))
        self.register_buffer("S", (s @ s.t()))

    def forward(self, x, y):

        X = torch.cat([x, y], 0)

        XX = X @ X.t()
        X2 = (X * X).sum(dim=1).unsqueeze(0)

        exponent = -2 * XX + X2.expand_as(XX) + X2.t().expand_as(XX)

        b = (
            exponent.unsqueeze(2).expand(-1, -1, self.bandwidths.shape[2])
            * -self.bandwidths
        )
        lossMMD = torch.sum(self.S.unsqueeze(2) * b.exp()).sqrt()

        return lossMMD

