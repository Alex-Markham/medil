import math

import torch
from torch import nn
from torch.nn.parameter import Parameter


class VariationalAutoencoder(nn.Module):
    def __init__(self, num_vae_latent, num_meas, num_hidden_layers, width_per_meas):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = Encoder(num_vae_latent, num_meas)
        self.decoder = Decoder(
            num_vae_latent, num_meas, num_hidden_layers, width_per_meas
        )

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
    def __init__(self, num_vae_latent, num_meas, width_per_meas=1):
        super(Block, self).__init__()
        self.input_dim = num_meas
        self.latent_dim = num_vae_latent
        self.hidden_dim = num_meas * width_per_meas
        self.output_dim = num_meas


class Encoder(Block):
    def __init__(self, num_vae_latent, num_meas):
        super(Encoder, self).__init__(num_vae_latent, num_meas)

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
    def __init__(self, num_vae_latent, num_meas, num_hidden_layers, width_per_meas):
        super(Decoder, self).__init__(num_vae_latent, num_meas, width_per_meas)

        # # decoder layer -- estimate mean
        # self.dec_mean = SparseLinear(
        #     in_features=self.latent_dim, out_features=self.output_dim
        # )

        # # decoder layer -- estimate log-covariance
        # self.fc_logcov = SparseLinear(
        #     in_features=self.latent_dim, out_features=self.output_dim
        # )

        # new arch
        self.mean_linear_fulcon = SparseLinear(
            in_features=self.latent_dim, out_features=self.hidden_dim
        )
        self.cov_linear_fulcon = SparseLinear(
            in_features=self.latent_dim, out_features=self.hidden_dim
        )

        hidden_block = torch.ones(width_per_meas, width_per_meas)
        hidden_blocks = [hidden_block for _ in range(num_meas)]
        hidden_mask = torch.block_diag(*hidden_blocks)

        self.mean_linear_hidden = {
            layer_idx: SparseLinear(
                in_features=self.hidden_dim,
                out_features=self.hidden_dim,
                mask=hidden_mask,
            )
            for layer_idx in range(num_hidden_layers)
        }
        self.cov_linear_hidden = {
            layer_idx: SparseLinear(
                in_features=self.hidden_dim,
                out_features=self.hidden_dim,
                mask=hidden_mask,
            )
            for layer_idx in range(num_hidden_layers)
        }

        output_block = torch.ones(1, width_per_meas)
        output_blocks = [output_block for _ in range(num_meas)]
        output_mask = torch.block_diag(*output_blocks)
        self.mean_linear_output = SparseLinear(
            in_features=self.hidden_dim, out_features=self.output_dim, mask=output_mask
        )
        self.cov_linear_output = SparseLinear(
            in_features=self.hidden_dim, out_features=self.output_dim, mask=output_mask
        )

        self.activation = torch.nn.Sigmoid()

    def forward(self, z):
        # linear layer
        # mean = self.dec_mean(z)
        # logcov = self.fc_logcov(z)

        # new arch
        mean = self.mean_linear_fulcon(z)
        mean = self.activation(mean)
        for hidden_layer in self.mean_linear_hidden.values():
            mean = hidden_layer(mean)
            mean = self.activation(mean)
        mean = self.mean_linear_output(mean)

        logcov = self.cov_linear_fulcon(z)
        logcov = self.activation(logcov)
        for hidden_layer in self.cov_linear_hidden.values():
            logcov = hidden_layer(logcov)
            logcov = self.activation(logcov)
        logcov = self.cov_linear_output(logcov)

        return mean, logcov


class SparseLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        mask=torch.ones(1),
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
