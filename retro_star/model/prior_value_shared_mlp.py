import torch
import torch.nn as nn
import torch.nn.functional as F
import logging


class SharedMLP(nn.Module):
    def __init__(self, fp_dim, latent_dim, dropout_rate, device):
        super(ValueMLP, self).__init__()
        self.dropout_rate = dropout_rate
        self.device = device

        logging.info('Initializing value model: latent_dim=%d' % self.latent_dim)

        # TODO
        self.representation_layers = self._generate_blk(2, fp_dim, latent_dim, latent_dim)
        self.value_layers = self._generate_blk(2, latent_dim, 1, latent_dim)
        self.prior_layers = self._generate_blk(2, latent_dim, 1, latent_dim)

    def _generate_blk(self, n_layers, in_dim, out_dim, latent_dim):
        layers = []
        layers.append(nn.Linear(in_dim, latent_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(self.dropout_rate))
        for _ in range(self.n_layers - 1):
            layers.append(nn.Linear(latent_dim, latent_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout_rate))
        layers.append(nn.Linear(latent_dim, out_dim))

        self.layers = nn.Sequential(*layers)

    # TODO
    def forward(self, fps):
        x = fps
        x = self.layers(x)
        # x = torch.log(1 + torch.exp(x))

        return x

    # TODO
    def get_value(self, fps):
        pass

    # TODO
    def get_prior(self, fps):
        pass