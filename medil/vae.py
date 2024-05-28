import math

import torch
from torch import nn
from torch.nn.parameter import Parameter


class VariationalAutoencoder(nn.Module):
    def __init__(self, m, n, mask):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = Encoder(m, n)
        self.decoder = Decoder(m, n, mask)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        latent = self.latent_sample(mu, logvar)
        x_recon, logcov = self.decoder(latent)

        return x_recon, logcov, mu, logvar

    def latent_sample(self, mu, logvar):
        # the re-parameterization trick
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu


class Block(nn.Module):
    def __init__(self, m, n):
        super(Block, self).__init__()
        self.input_dim = n
        self.latent_dim = m
        self.output_dim = n


class Encoder(Block):
    def __init__(self, m, n):
        super(Encoder, self).__init__(m, n)

        # first encoder layer
        self.inter_dim = self.input_dim
        self.enc1 = nn.Linear(in_features=self.input_dim, out_features=self.inter_dim)

        # second encoder layer
        self.enc2 = nn.Linear(in_features=self.inter_dim, out_features=self.inter_dim)

        # map to mu and variance
        self.fc_mu = nn.Linear(in_features=self.inter_dim, out_features=self.latent_dim)
        self.fc_logvar = nn.Linear(
            in_features=self.inter_dim, out_features=self.latent_dim
        )

    def forward(self, x):
        # encoder layers
        inter = torch.relu(self.enc1(x))
        inter = torch.relu(self.enc2(inter))

        # calculate mu & logvar
        mu = self.fc_mu(inter)
        logvar = self.fc_logvar(inter)

        return mu, logvar


class Decoder(Block):
    def __init__(self, m, n, mask):
        super(Decoder, self).__init__(m, n)

        # decoder layer -- estimate mean
        self.dec_mean = SparseLinear(
            in_features=self.latent_dim, out_features=self.output_dim, mask=mask
        )

        # decoder layer -- estimate log-covariance
        self.fc_logcov = SparseLinear(
            in_features=self.latent_dim, out_features=self.output_dim, mask=mask
        )

    def forward(self, z):
        # linear layer
        mean = self.dec_mean(z)
        logcov = self.fc_logcov(z)

        return mean, logcov


class SparseLinear(nn.Module):
    def __init__(
        self, in_features, out_features, mask, bias=True, device=None, dtype=None
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super(SparseLinear, self).__init__()
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
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        # masked linear layer
        return nn.functional.linear(input, self.weight * self.mask, self.bias)

    def extra_repr(self):
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )
