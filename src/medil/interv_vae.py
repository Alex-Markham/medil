import math
import warnings

import torch
from torch import nn
from torch.nn.parameter import Parameter


class VariationalAutoencoder(nn.Module):
    def __init__(
        self, num_meas, meas_width, meas_depth, num_latent, latent_width, latent_depth
    ):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = Encoder(num_meas, num_latent, latent_width)
        self.decoder = Decoder(
            num_latent, latent_width, latent_depth, num_meas, meas_width, meas_depth
        )

    def forward(self, x, label):
        mu, logvar = self.encoder(x)
        latent = self.latent_sample(mu, logvar)
        x_recon, logcov = self.decoder(latent, label)

        return x_recon, logcov, mu, logvar

    def latent_sample(self, mu, logvar):
        # the re-parameterization trick
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu


class Encoder(nn.Module):
    def __init__(self, num_meas, num_latent, latent_width):
        super(Encoder, self).__init__()

        # first encoder layer
        self.enc1 = nn.Linear(in_features=num_meas, out_features=num_meas)

        # second encoder layer
        self.enc2 = nn.Linear(in_features=num_meas, out_features=num_meas)

        # map to mu and variance
        num_vae_latent = num_latent * latent_width
        self.fc_mu = nn.Linear(in_features=num_meas, out_features=num_vae_latent)
        self.fc_logvar = nn.Linear(in_features=num_meas, out_features=num_vae_latent)

    def forward(self, x):
        activation = torch.nn.GELU()
        # encoder layers
        x = activation(self.enc1(x))
        x = activation(self.enc2(x))

        # calculate mu & logvar
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar


class Decoder(nn.Module):
    def __init__(
        self, num_latent, latent_width, latent_depth, num_meas, meas_width, meas_depth
    ):
        super(Decoder, self).__init__()
        num_vae_latent = num_latent * latent_width
        num_vae_meas = num_meas * meas_width
        if meas_depth == 0 and meas_width > 1:
            num_vae_meas = num_meas
            warnings.warn(
                f"Reduced architecture complexity: `meas_width` set to 1 rather than {meas_width} since `meas_depth`={meas_depth}."
            )

        # hidden latent layers
        hidden_block = torch.ones(latent_width, latent_width)
        hidden_blocks = [hidden_block for _ in range(num_latent)]
        hidden_mask = torch.block_diag(*hidden_blocks)
        self.mean_hidden_latent = {
            layer_idx: SparseLinear(
                in_features=num_vae_latent,
                out_features=num_vae_latent,
                mask=hidden_mask,
            )
            for layer_idx in range(latent_depth)
        }
        self.logcov_hidden_latent = {
            layer_idx: SparseLinear(
                in_features=num_vae_latent,
                out_features=num_vae_latent,
                mask=hidden_mask,
            )
            for layer_idx in range(latent_depth)
        }

        # causal layer
        self.mean_causal = SparseLinear(
            in_features=num_vae_latent, out_features=num_vae_latent
        )
        self.logcov_causal = SparseLinear(
            in_features=num_vae_latent, out_features=num_vae_latent
        )

        # mixture layer
        self.mean_mix = SparseLinear(
            in_features=num_vae_latent, out_features=num_vae_meas
        )
        self.logcov_mix = SparseLinear(
            in_features=num_vae_latent, out_features=num_vae_meas
        )

        # additional mixture layers
        self.mean_hidden_mix = {
            layer_idx: SparseLinear(
                in_features=num_vae_meas,
                out_features=num_vae_meas,
            )
            for layer_idx in range(meas_depth - 1)
        }
        self.mean_hidden_mix[meas_depth] = SparseLinear(
            in_features=num_vae_meas,
            out_features=num_meas,
        )
        self.logcov_hidden_mix = {
            layer_idx: SparseLinear(
                in_features=num_vae_latent,
                out_features=num_vae_latent,
            )
            for layer_idx in range(meas_depth - 1)
        }
        self.logcov_hidden_mix[meas_depth] = SparseLinear(
            in_features=num_vae_meas,
            out_features=num_meas,
        )

        self.activation = torch.nn.GELU()

    def forward(self, z, label):
        # hidden layers for latent exogenous variables
        mean = z.copy()
        logcov = z.copy()
        for hidden_layer in self.mean_hidden_latent.values():
            mean = hidden_layer(mean)
            mean = self.activation(mean)
        for hidden_layer in self.logcov_hidden_latent.values():
            logcov = hidden_layer(logcov)
            logcov = self.activation(logcov)

        # connect exogenous variables to latent causal DAG
        mean = self.interv_mask(label, mean)
        mean = self.mean_causal(mean)
        mean = self.activation(mean)
        logcov = self.interv_mask(label, logcov)
        logcov = self.logcov_causal(logcov)
        logcov = self.activation(logcov)

        # mix latent causal vars into measurements
        mean = self.mean_mix(mean)
        logcov = self.logcov_mix(logcov)

        # hidden layers for mixture
        for hidden_layer in self.mean_hidden_mix.values():
            mean = self.activation(mean)
            mean = hidden_layer(mean)
        for hidden_layer in self.logcov_hidden_mix.values():
            logcov = self.activation(logcov)
            logcov = hidden_layer(logcov)

        return mean, logcov

    def interv_mask(self, label, noise):
        print(f"noise has shape {noise.shape}")
        return noise


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
        # nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.orthogonal_(self.weight)
        # nn.init.sparse_(self.weight, 2 / 3)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        # masked linear layer
        if self.mask is None:
            return nn.functional.linear(input, self.weight, self.bias)
        else:
            return nn.functional.linear(input, self.weight * self.mask, self.bias)

    def extra_repr(self):
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )
