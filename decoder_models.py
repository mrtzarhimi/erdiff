import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import Ridge


# =====================================================================
# 1. Behavior Regressor (Kinematics Decoder)
# =====================================================================

class KinematicsRegressor(nn.Module):
    """
    Kinematics decoder using a small temporal receptive field (causal 1D convolution)
    to account for the neural-to-kinematic delay (~80-100ms).
    """
    def __init__(self, latent_dim: int = 8, behavior_dim: int = 2, kernel_size: int = 5):
        super().__init__()
        # Causal padding = kernel_size - 1
        self.conv = nn.Conv1d(
            in_channels=latent_dim,
            out_channels=behavior_dim,
            kernel_size=kernel_size,
            padding=kernel_size - 1
        )
        self.kernel_size = kernel_size

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (batch, seq_len, latent_dim) -> (batch, latent_dim, seq_len)
        z_trans = z.permute(0, 2, 1)
        out = self.conv(z_trans)
        # Slice off extra padding to maintain causal length
        out = out[:, :, :-(self.kernel_size - 1)]
        # Return as (batch, seq_len, behavior_dim)
        return out.permute(0, 2, 1)

    def regularization_loss(self) -> torch.Tensor:
        """Computes L2 penalty on weights corresponding to ridge penalty."""
        return self.l2_reg * torch.sum(self.linear.weight ** 2)


# =====================================================================
# 2. Target Direction Classifier
# =====================================================================

class DirectionClassifier(nn.Module):
    """
    Categorical target classifier mapping latent dynamics Z to one of the 8
    reaching directions in the center-out reaching task.
    """
    def __init__(self, latent_dim: int = 8, num_classes: int = 8, pool_type: str = "mean"):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.pool_type = pool_type

        # Linear classifier head over pooled latent representations
        self.classifier = nn.Linear(latent_dim, num_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Latent trajectories of shape (batch_size, seq_len, latent_dim)
        Returns:
            logits: Class logits of shape (batch_size, num_classes)
        """
        if self.pool_type == "mean":
            z_pooled = torch.mean(z, dim=1)  # (batch_size, latent_dim)
        elif self.pool_type == "max":
            z_pooled, _ = torch.max(z, dim=1)
        else:
            z_pooled = z[:, -1, :]  # Final time-bin representation

        logits = self.classifier(z_pooled)
        return logits


# =====================================================================
# 3. Scikit-Learn Offline Ridge Decoder
# =====================================================================

class OfflineRidgeDecoder:
    """
    Standard Scikit-Learn Ridge Regression wrapper for computing velocity decoding
    benchmarks offline on extracted NumPy latent dynamics.
    """
    def __init__(self, alpha: float = 1.0):
        self.model = Ridge(alpha=alpha)

    def fit(self, z_source, v_source):
        """
        Args:
            z_source: Array of shape (n_trials, seq_len, latent_dim)
            v_source: Array of shape (n_trials, seq_len, behavior_dim)
        """
        n_trials, seq_len, d = z_source.shape
        behavior_dim = v_source.shape[-1]

        z_flat = z_source.reshape(-1, d)
        v_flat = v_source.reshape(-1, behavior_dim)

        self.model.fit(z_flat, v_flat)

    def predict(self, z_target):
        """
        Args:
            z_target: Array of shape (n_trials, seq_len, latent_dim)
        Returns:
            v_pred: Array of shape (n_trials, seq_len, behavior_dim)
        """
        n_trials, seq_len, d = z_target.shape
        z_flat = z_target.reshape(-1, d)
        v_pred_flat = self.model.predict(z_flat)
        return v_pred_flat.reshape(n_trials, seq_len, -1)


# =====================================================================
# 4. Evaluation Metrics (R-squared, RMSE, Classification Accuracy)
# =====================================================================

def compute_r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Computes the coefficient of determination (R^2 in %):
      R^2 = (1 - SS_res / SS_tot) * 100
    Averaged across behavioral dimensions (vx, vy).
    """
    # Use .reshape instead of .view to handle non-contiguous memory layouts
    y_true_flat = y_true.reshape(-1, y_true.size(-1))
    y_pred_flat = y_pred.reshape(-1, y_pred.size(-1))

    ss_res = torch.sum((y_true_flat - y_pred_flat) ** 2, dim=0)
    ss_tot = torch.sum((y_true_flat - torch.mean(y_true_flat, dim=0)) ** 2, dim=0)

    r2_per_dim = 1.0 - (ss_res / (ss_tot + 1e-8))
    return (torch.mean(r2_per_dim) * 100.0).item()


def compute_rmse(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Computes Root-Mean-Square Error (RMSE) between decoded kinematics and ground truth.
    """
    return torch.sqrt(F.mse_loss(y_pred, y_true)).item()


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Computes multi-class classification accuracy (in %) for reach target directions.
    """
    preds = torch.argmax(logits, dim=-1)
    acc = (preds == labels).float().mean() * 100.0
    return acc.item()
