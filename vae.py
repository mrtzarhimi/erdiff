import torch
import torch.nn as nn
import torch.nn.functional as F


class SequentialVAEEncoder(nn.Module):
    def __init__(
        self,
        n_neurons: int,
        hidden_dim: int = 32,
        latent_dim: int = 8,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.read_in = nn.Linear(n_neurons, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.act = nn.ELU()
        self.dropout = nn.Dropout(p=dropout)

        self.rnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)

        # Fix Smoking Gun 2: Initialize logvar bias to -3.0 (initial std ~ 0.22)
        # Prevents unit-variance white noise from destroying trajectories at epoch 1
        nn.init.constant_(self.fc_logvar.bias, -3.0)
        nn.init.xavier_uniform_(self.fc_logvar.weight, gain=0.1)

    def forward(self, x: torch.Tensor):
        h = self.read_in(x)
        h = self.norm(h)
        h = self.dropout(self.act(h))

        rnn_out, _ = self.rnn(h)
        rnn_out = self.dropout(rnn_out)

        mu = self.fc_mu(rnn_out)
        logvar = torch.clamp(self.fc_logvar(rnn_out), min=-5.0, max=0.0)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class SequentialVAEDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = 8,
        hidden_dim: int = 32,
        n_neurons: int = 96,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.fc_in = nn.Linear(latent_dim, hidden_dim)
        self.act = nn.ELU()
        self.dropout = nn.Dropout(p=dropout)

        self.rnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.read_out = nn.Linear(hidden_dim, n_neurons)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.dropout(self.act(self.fc_in(z)))
        rnn_out, _ = self.rnn(h)
        log_rates = self.read_out(self.dropout(rnn_out))
        rates = F.softplus(log_rates) + 1e-6
        return rates


class RidgeBehaviorRegressor(nn.Module):
    def __init__(self, latent_dim: int = 8, behavior_dim: int = 2, l2_alpha: float = 0.1):
        super().__init__()
        self.linear = nn.Linear(latent_dim, behavior_dim)
        self.l2_alpha = l2_alpha

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.linear(z)

    def ridge_penalty(self) -> torch.Tensor:
        return self.l2_alpha * torch.sum(self.linear.weight ** 2)


class SequentialVAE(nn.Module):
    def __init__(
        self,
        n_neurons: int,
        latent_dim: int = 8,
        hidden_dim: int = 32,
        behavior_dim: int = 2,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.encoder = SequentialVAEEncoder(n_neurons, hidden_dim, latent_dim, dropout)
        self.decoder = SequentialVAEDecoder(latent_dim, hidden_dim, n_neurons, dropout=0.3)
        self.behavior_decoder = RidgeBehaviorRegressor(latent_dim, behavior_dim, l2_alpha=0.1)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encoder(x)
        # During training, pass reparameterized z to BOTH decoder and behavior_decoder
        # so the encoder cannot overfit deterministic spikes into mu.
        # During eval, reparameterize() automatically returns deterministic mu.
        z = self.encoder.reparameterize(mu, logvar)
        rates = self.decoder(z)
        pred_vel = self.behavior_decoder(z)
        return rates, pred_vel, mu, logvar, z

    def freeze_for_alignment(self):
        for param in self.parameters():
            param.requires_grad = False
        for param in self.encoder.read_in.parameters():
            param.requires_grad = True


def poisson_nll_loss(rate: torch.Tensor, target_spikes: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    loss = rate - target_spikes * torch.log(rate + eps)
    return loss.mean()


def kl_divergence_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
    return kld.mean()


def trajectory_smoothness_loss(mu: torch.Tensor) -> torch.Tensor:
    """
    Enforces continuous temporal evolution across the latent trajectory Z:
    penalizes sharp, discrete jumps between adjacent time bins t and t+1.
    """
    diff = mu[:, 1:, :] - mu[:, :-1, :]
    return torch.mean(diff ** 2)
