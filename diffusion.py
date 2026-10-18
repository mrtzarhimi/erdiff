"""

Diffusion model implementation
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# 1. Sinusoidal Timestep Embeddings
# =====================================================================

class SinusoidalTimeEmbedding(nn.Module):
    """
    Standard sinusoidal positional encoding to inject continuous timestep t
    into the Transformer blocks.
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        # timesteps: (batch_size,)
        half_dim = self.embed_dim // 2
        exponent = -math.log(10000) * torch.arange(
            half_dim, dtype=torch.float32, device=timesteps.device
        ) / half_dim
        emb = torch.exp(exponent)
        emb = timesteps.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.embed_dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


# =====================================================================
# 2. Spatio-Temporal Transformer Block (STBlock)
# =====================================================================

class STBlock(nn.Module):
    """
    Each STBlock consists of:
      1. Spatio Transformer Layer: Multi-Head Self-Attention over latent dimensions at each time bin.
      2. Temporal Transformer Layer: Multi-Head Self-Attention over time bins for each latent dimension.
    """
    def __init__(self, seq_len: int, latent_dim: int, hidden_dim: int = 64, n_heads: int = 4):
        super().__init__()
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Timestep projection
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 1. Spatio Transformer Layer (attending across latent dimensions d)
        self.spatio_norm1 = nn.LayerNorm(hidden_dim)
        self.spatio_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=n_heads, batch_first=True
        )
        self.spatio_norm2 = nn.LayerNorm(hidden_dim)
        self.spatio_ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        # 2. Temporal Transformer Layer (attending across time bins l)
        self.temporal_norm1 = nn.LayerNorm(hidden_dim)
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=n_heads, batch_first=True
        )
        self.temporal_norm2 = nn.LayerNorm(hidden_dim)
        self.temporal_ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, latent_dim, hidden_dim)
        b, l, d, h = x.shape

        # Inject timestep embedding
        t_proj = self.time_mlp(t_emb).view(b, 1, 1, h)
        x = x + t_proj

        # --- Spatio Attention (along latent dimensions for each time bin) ---
        # Reshape to (b * l, d, h)
        x_spatio = x.view(b * l, d, h)
        res_s = x_spatio
        x_spatio = self.spatio_norm1(x_spatio)
        attn_out_s, _ = self.spatio_attn(x_spatio, x_spatio, x_spatio)
        x_spatio = res_s + attn_out_s
        x_spatio = x_spatio + self.spatio_ff(self.spatio_norm2(x_spatio))
        x = x_spatio.view(b, l, d, h)

        # --- Temporal Attention (along time bins for each latent dimension) ---
        # Reshape to (b * d, l, h)
        x_temp = x.permute(0, 2, 1, 3).contiguous().view(b * d, l, h)
        res_t = x_temp
        x_temp = self.temporal_norm1(x_temp)
        attn_out_t, _ = self.temporal_attn(x_temp, x_temp, x_temp)
        x_temp = res_t + attn_out_t
        x_temp = x_temp + self.temporal_ff(self.temporal_norm2(x_temp))

        # Restore original tensor shape: (b, l, d, h)
        x = x_temp.view(b, d, l, h).permute(0, 2, 1, 3).contiguous()
        return x


# =====================================================================
# 3. Diffusion Transformer Denoiser: \epsilon_\theta(Z_t, t)
# =====================================================================

class LatentDiffusionModel(nn.Module):
    """
    Score estimator / Denoiser for neural latent trajectories Z \in R^(batch, seq_len, latent_dim).
    Uses stacked Spatio-Temporal Transformer Blocks.
    """
    def __init__(
        self,
        seq_len: int = 39,
        latent_dim: int = 8,
        hidden_dim: int = 64,
        num_blocks: int = 4,
        n_heads: int = 4,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Input & Output projections for scalar elements at (time_bin, latent_dim)
        self.in_proj = nn.Linear(1, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, 1)

        # Sinusoidal embedding for diffusion time t \in [0, T]
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # Stack of Spatio-Temporal Transformer Blocks
        self.blocks = nn.ModuleList([
            STBlock(seq_len=seq_len, latent_dim=latent_dim, hidden_dim=hidden_dim, n_heads=n_heads)
            for _ in range(num_blocks)
        ])

    def forward(self, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_t: Noisy latent trajectories of shape (batch, seq_len, latent_dim)
            t: Diffusion continuous timesteps of shape (batch,)
        Returns:
            eps_pred: Estimated noise tensor of shape (batch, seq_len, latent_dim)
        """
        # Expand last dim: (batch, seq_len, latent_dim, 1) -> (b, l, d, hidden_dim)
        x = self.in_proj(z_t.unsqueeze(-1))
        t_emb = self.time_embed(t)

        for block in self.blocks:
            x = block(x, t_emb)

        eps_pred = self.out_proj(x).squeeze(-1)  # (batch, seq_len, latent_dim)
        return eps_pred


# =====================================================================
# 4. VP-SDE Diffusion Noise Schedule
# =====================================================================

class VPSDEScheduler:
    """
    Variance-Preserving SDE (VP-SDE) forward marginals:
      p_{0t}(Z_t | Z_0) = N(mu_t * Z_0, Sigma_t)
      mu_t = exp(-0.5 * int_0^t beta(s) ds)
      Sigma_t = (1 - mu_t^2) * I
    """
    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0, T: float = 1.0):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.T = T

    def marginal_params(self, t: torch.Tensor):
        # Integral of beta(s) from 0 to t
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        alpha_t = torch.exp(log_mean_coeff)
        std_t = torch.sqrt(1.0 - torch.exp(2.0 * log_mean_coeff))
        return alpha_t, std_t

    def add_noise(self, z_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
        if noise is None:
            noise = torch.randn_like(z_0)
        alpha_t, std_t = self.marginal_params(t)
        # Reshape for broadcasting over (batch, seq_len, latent_dim)
        alpha_t = alpha_t.view(-1, 1, 1)
        std_t = std_t.view(-1, 1, 1)
        z_t = alpha_t * z_0 + std_t * noise
        return z_t, noise


# =====================================================================
# 5. Losses: DSM (Phase 1), MLA (Phase 2), and Sinkhorn Divergence
# =====================================================================

def dsm_loss(
    diffusion_model: LatentDiffusionModel,
    scheduler: VPSDEScheduler,
    z_0: torch.Tensor,
) -> torch.Tensor:
    """
    Phase 1: Denoising Score Matching (DSM) Loss for training the Diffusion Model.
    """
    batch_size = z_0.size(0)
    # Uniformly sample continuous timesteps t in [0.001, T]
    t = torch.rand(batch_size, device=z_0.device) * (scheduler.T - 0.001) + 0.001
    eps = torch.randn_like(z_0)

    z_t, _ = scheduler.add_noise(z_0, t, eps)
    eps_pred = diffusion_model(z_t, t)

    # Standard DSM weighted squared error
    loss = F.mse_loss(eps_pred, eps)
    return loss


def mla_loss(
    diffusion_model: LatentDiffusionModel,
    scheduler: VPSDEScheduler,
    z_0_target: torch.Tensor,
) -> torch.Tensor:
    """
    Phase 2: Maximum Likelihood Alignment (MLA) Loss (Equation 12 & Equation 26).
    The diffusion model is FROZEN. The gradient flows back through z_t into z_0_target (and the encoder).
    """
    batch_size = z_0_target.size(0)
    t = torch.rand(batch_size, device=z_0_target.device) * (scheduler.T - 0.001) + 0.001
    eps = torch.randn_like(z_0_target)

    # Differentiable forward diffusion of target latents
    z_t, _ = scheduler.add_noise(z_0_target, t, eps)

    # Denoiser evaluates whether the spatio-temporal structure fits source dynamics
    eps_pred = diffusion_model(z_t, t)

    # Weighted noise residual: || eps - eps_theta(z_t, t) ||^2
    # The error drives encoder updates
    residual_loss = torch.mean((eps - eps_pred) ** 2)
    return residual_loss


def sinkhorn_loss(
    z_source: torch.Tensor,
    z_target: torch.Tensor,
    epsilon: float = 0.1,
    num_iters: int = 30,
) -> torch.Tensor:
    """
    Entropy-regularized Optimal Transport / Sinkhorn Divergence (Equation 27).
    Normalized by trajectory dimensionality (seq_len * latent_dim) so its scale
    matches the element-wise mean in mla_loss.
    """
    b_s, l, d = z_source.shape
    b_t = z_target.size(0)
    x = z_source.reshape(b_s, -1)
    y = z_target.reshape(b_t, -1)

    # Mean squared Euclidean cost per time-latent element
    cost = (torch.cdist(x, y, p=2.0) ** 2) / (l * d)

    mu = torch.full((b_s,), 1.0 / b_s, device=z_source.device)
    nu = torch.full((b_t,), 1.0 / b_t, device=z_target.device)

    # Stabilized kernel matrix
    K = torch.exp(-cost / epsilon)

    u = torch.ones_like(mu)
    v = torch.ones_like(nu)

    for _ in range(num_iters):
        u = mu / (torch.matmul(K, v) + 1e-8)
        v = nu / (torch.matmul(K.t(), u) + 1e-8)

    gamma = u.unsqueeze(1) * K * v.unsqueeze(0)
    loss = torch.sum(gamma * cost)
    return loss


def total_alignment_loss(
    diffusion_model: LatentDiffusionModel,
    scheduler: VPSDEScheduler,
    z_target: torch.Tensor,
    z_source: torch.Tensor,
    alpha: float = 0.5,
) -> torch.Tensor:
    """
    Total Phase 2 Alignment Loss:
      L = (1 - alpha) * L_MLA + alpha * L_SHD
    """
    loss_mla = mla_loss(diffusion_model, scheduler, z_target)
    loss_shd = sinkhorn_loss(z_source, z_target)
    return (1.0 - alpha) * loss_mla + alpha * loss_shd