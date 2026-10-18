"""

Variational Autoencoder implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SequentialVAEEncoder(nn.Module):
    """
    Probabilistic Encoder q(Z | X; phi)
    Takes raw neural population spike trains X in R^(batch, seq_len, n_neurons)
    and maps them to mean and log-variance of latent trajectories Z in R^(batch, seq_len, latent_dim).
    """
    def __init__(
        self,
        n_neurons: int,
        hidden_dim: int = 64,
        latent_dim: int = 8,
        rnn_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Read-in layer: linearly projects variable neuron count n to hidden_dim.
        # During target alignment (Phase 2), ONLY this layer is fine-tuned (linear probing).
        self.read_in = nn.Linear(n_neurons, hidden_dim)

        # Non-linear activation & dropout
        self.act = nn.ELU()
        self.dropout = nn.Dropout(p=dropout)

        # Sequential recurrent encoder
        self.rnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=False,
        )

        # Projection heads for variational posterior parameters
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor):
        # x: (batch_size, seq_len, n_neurons)
        h = self.read_in(x)
        h = self.act(h)
        h = self.dropout(h)

        # RNN encodes temporal dependencies across trial bins
        rnn_out, _ = self.rnn(h)  # (batch_size, seq_len, hidden_dim)
        rnn_out = self.dropout(rnn_out)

        mu = self.fc_mu(rnn_out)          # (batch_size, seq_len, latent_dim)
        logvar = self.fc_logvar(rnn_out)  # (batch_size, seq_len, latent_dim)

        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class SequentialVAEDecoder(nn.Module):
    """
    Probabilistic Decoder p(X | Z; psi)
    Reconstructs Poisson firing rates lambda from latent trajectory Z.
    """
    def __init__(
        self,
        latent_dim: int = 8,
        hidden_dim: int = 64,
        n_neurons: int = 200,
        rnn_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Latent read-in projection
        self.fc_in = nn.Linear(latent_dim, hidden_dim)
        self.act = nn.ELU()
        self.dropout = nn.Dropout(p=dropout)

        # Sequential recurrent decoder
        self.rnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=rnn_layers,
            batch_first=True,
        )

        # Read-out layer back to observation space (log firing rates)
        self.read_out = nn.Linear(hidden_dim, n_neurons)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (batch_size, seq_len, latent_dim)
        h = self.fc_in(z)
        h = self.act(h)
        h = self.dropout(h)

        rnn_out, _ = self.rnn(h)  # (batch_size, seq_len, hidden_dim)
        rnn_out = self.dropout(rnn_out)

        # Firing rates must be strictly positive (parameter of Poisson distribution)
        # Using exp(read_out) or softplus
        log_rates = self.read_out(rnn_out)
        rates = torch.exp(torch.clamp(log_rates, min=-10.0, max=10.0))
        return rates


class BehaviorRegressor(nn.Module):
    """
    Kinematics decoder mapping neural latent trajectories to behavioral output (velocity/position).
    Cooperatively trained in Phase 1 and frozen during alignment/testing.
    """
    def __init__(self, latent_dim: int = 8, behavior_dim: int = 2):
        super().__init__()
        self.linear = nn.Linear(latent_dim, behavior_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (batch_size, seq_len, latent_dim) -> (batch_size, seq_len, behavior_dim)
        return self.linear(z)


class SequentialVAE(nn.Module):
    """
    Full Sequential VAE model combining Encoder, Neural Decoder, and Behavior Decoder.
    """
    def __init__(
        self,
        n_neurons: int,
        latent_dim: int = 8,
        hidden_dim: int = 64,
        behavior_dim: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = SequentialVAEEncoder(
            n_neurons=n_neurons,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            dropout=dropout,
        )
        self.decoder = SequentialVAEDecoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            n_neurons=n_neurons,
            dropout=dropout,
        )
        self.behavior_decoder = BehaviorRegressor(
            latent_dim=latent_dim,
            behavior_dim=behavior_dim,
        )

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encoder(x)
        z = self.encoder.reparameterize(mu, logvar)
        x_recon_rate = self.decoder(z)
        v_pred = self.behavior_decoder(z)
        return x_recon_rate, v_pred, mu, logvar, z

    def freeze_for_alignment(self):
        """
        Freezes the entire model EXCEPT the encoder read-in layer for Phase 2 target alignment.
        """
        for param in self.parameters():
            param.requires_grad = False
        
        # Only linear-probe the encoder read-in layer
        for param in self.encoder.read_in.parameters():
            param.requires_grad = True


# ==========================================
# Loss Functions for Source Domain Training
# ==========================================

def poisson_nll_loss(rate: torch.Tensor, target_spikes: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Negative log-likelihood under Poisson assumption:
    NLL = sum( rate - target * log(rate + eps) )
    """
    loss = rate - target_spikes * torch.log(rate + eps)
    return loss.sum(dim=-1).mean()


def kl_divergence_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Analytic KL divergence between q(z|x) = N(mu, diag(exp(logvar))) and p(z) = N(0, I):
    KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    """
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
    return kld.mean()


def source_domain_loss(
    recon_rate: torch.Tensor,
    spikes: torch.Tensor,
    pred_vel: torch.Tensor,
    true_vel: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    kl_weight: float = 1.0,
    behavior_weight: float = 1.0,
) -> torch.Tensor:
    """
    Phase 1 joint loss: Poisson NLL + KL Divergence + Kinematics MSE
    """
    rec_loss = poisson_nll_loss(recon_rate, spikes)
    kl_loss = kl_divergence_loss(mu, logvar)
    mse_loss = F.mse_loss(pred_vel, true_vel)
    
    total_loss = rec_loss + kl_weight * kl_loss + behavior_weight * mse_loss
    return total_loss