"""Variational autoencoder components for NeuroCausalFactorAnalysis."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class VariationalAutoencoder(nn.Module):
    def __init__(
        self,
        num_latent,
        num_meas,
        num_hidden_layers,
        latent_width,
        meas_width,
        biadj=None,
        encoder_hidden_dim=None,
        num_classes=1,
    ):
        super().__init__()

        if encoder_hidden_dim is None:
            encoder_hidden_dim = max(num_meas, 64)

        self.encoder = Encoder(
            num_latent=num_latent * latent_width,
            num_meas=num_meas,
            hidden_dim=encoder_hidden_dim,
        )

        self.decoder = Decoder(
            num_latent=num_latent,
            num_meas=num_meas,
            num_hidden_layers=num_hidden_layers,
            latent_width=latent_width,
            meas_width=meas_width,
            biadj=biadj,
            num_classes=num_classes,
        )

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.latent_sample(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

    def latent_sample(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu


class Encoder(nn.Module):
    def __init__(self, num_latent, num_meas, hidden_dim=64):
        super().__init__()
        self.enc1 = nn.Linear(num_meas, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.enc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.activation = nn.GELU()
        self.fc_mu = nn.Linear(hidden_dim, num_latent)
        self.fc_logvar = nn.Linear(hidden_dim, num_latent)

    def forward(self, x):
        h = self.activation(self.bn1(self.enc1(x)))
        h = self.activation(self.bn2(self.enc2(h)))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(
        self,
        num_latent,
        num_meas,
        num_hidden_layers,
        latent_width,
        meas_width,
        biadj=None,
        num_classes=1,
    ):
        super().__init__()

        self.num_latent = num_latent
        self.num_meas = num_meas
        self.num_classes = num_classes
        self.latent_width = latent_width
        self.meas_width = meas_width

        self.latent_dim = num_latent * latent_width
        self.hidden_dim = num_meas * meas_width

        if biadj is None:
            biadj = torch.ones(num_meas, num_latent)
        else:
            biadj = torch.as_tensor(biadj, dtype=torch.float32)

        first_mask = self._expand_biadj(biadj, meas_width, latent_width)
        hidden_mask = self._make_hidden_block_mask(num_meas, meas_width)
        output_mask = self._make_output_mask(num_meas, meas_width, num_classes)

        self.linear_in = SparseLinear(
            in_features=self.latent_dim,
            out_features=self.hidden_dim,
            mask=first_mask,
        )

        self.bn_in = nn.BatchNorm1d(self.hidden_dim)

        self.hidden_layers = nn.ModuleList(
            [
                SparseLinear(
                    in_features=self.hidden_dim,
                    out_features=self.hidden_dim,
                    mask=hidden_mask,
                )
                for _ in range(num_hidden_layers)
            ]
        )

        self.hidden_bns = nn.ModuleList(
            [nn.BatchNorm1d(self.hidden_dim) for _ in range(num_hidden_layers)]
        )

        self.linear_out = SparseLinear(
            in_features=self.hidden_dim,
            out_features=self.num_meas * num_classes,
            mask=output_mask,
        )

        self.activation = nn.GELU()

    @staticmethod
    def _expand_biadj(biadj, meas_width, latent_width):
        return biadj.repeat_interleave(meas_width, dim=0).repeat_interleave(
            latent_width, dim=1
        )

    @staticmethod
    def _make_hidden_block_mask(num_meas, width_per_meas):
        block = torch.ones(width_per_meas, width_per_meas)
        blocks = [block for _ in range(num_meas)]
        return torch.block_diag(*blocks)

    @staticmethod
    def _make_output_mask(num_meas, width_per_meas, num_classes=1):
        block = torch.ones(num_classes, width_per_meas)
        blocks = [block for _ in range(num_meas)]
        return torch.block_diag(*blocks)

    def forward(self, z):
        h = self.activation(self.bn_in(self.linear_in(z)))

        for layer, bn in zip(self.hidden_layers, self.hidden_bns):
            h = self.activation(bn(layer(h)))

        x_recon = self.linear_out(h)
        return x_recon


class SparseLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        mask=None,
        bias=True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self.in_features = in_features
        self.out_features = out_features

        if mask is None:
            mask = torch.ones(out_features, in_features)
        else:
            if mask.shape != (out_features, in_features):
                raise ValueError(
                    f"mask must have shape {(out_features, in_features)}, "
                    f"got {tuple(mask.shape)}"
                )

        self.register_buffer("mask", mask.float())

        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.weight)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return F.linear(x, self.weight * self.mask, self.bias)
