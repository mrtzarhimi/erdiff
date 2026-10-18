import torch
import torch.nn as nn
import torch.nn.functional as F


class SequentialVAEEncoder(nn.Module):
    def __init__(
        self,
        n_neurons: int,
        hidden_dim: int = 64,
        latent_dim: int = 8,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.read_in = nn.Linear(n_neurons, hidden_dim)
        self.act = nn.ELU()
        self.dropout = nn.Dropout(p=dropout)

        self.rnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor):
        h = self.dropout(self.act(self.read_in(x)))
        rnn_out, _ = self.rnn(h)
        rnn_out = self.dropout(rnn_out)

        mu = self.fc_mu(rnn_out)
        logvar = self.fc_logvar(rnn_out)
        # Clamp logvar to prevent numerical collapse
        logvar = torch.clamp(logvar, min=-5.0, max=2.0)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class SequentialVAEDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = 8,
        hidden_dim: int = 64,
        n_neurons: int = 96,
        dropout: float = 0.2,
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
        rates = torch.exp(torch.clamp(log_rates, min=-10.0, max=8.0))
        return rates


class ContinuousBehaviorRegressor(nn.Module):
    """
    Continuous temporal lag decoder (causal 1D convolution over 5 time bins ~100ms).
    Prevents discrete point-memorization by modeling continuous temporal flow.
    """
    def __init__(self, latent_dim: int = 8, behavior_dim: int = 2, kernel_size: int = 5):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            in_channels=latent_dim,
            out_channels=behavior_dim,
            kernel_size=kernel_size,
            padding=kernel_size - 1,  # causal padding
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (batch, seq_len, latent_dim) -> (batch, latent_dim, seq_len)
        z_t = z.permute(0, 2, 1)
        out = self.conv(z_t)
        # Slice off right-side padding to keep it strictly causal
        out = out[:, :, :-(self.kernel_size - 1)]
        # Return contiguous tensor: (batch, seq_len, behavior_dim)
        return out.permute(0, 2, 1).contiguous()


class SequentialVAE(nn.Module):
    def __init__(
        self,
        n_neurons: int,
        latent_dim: int = 8,
        hidden_dim: int = 64,
        behavior_dim: int = 2,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.encoder = SequentialVAEEncoder(n_neurons, hidden_dim, latent_dim, dropout)
        self.decoder = SequentialVAEDecoder(latent_dim, hidden_dim, n_neurons, dropout)
        self.behavior_decoder = ContinuousBehaviorRegressor(latent_dim, behavior_dim)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encoder(x)
        z = self.encoder.reparameterize(mu, logvar)
        rates = self.decoder(z)
        pred_vel = self.behavior_decoder(mu)  # Train behavior on clean continuous trajectory
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
