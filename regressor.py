"""

Regressor model implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import Ridge


# =====================================================================
# 1. Kinematics Regressor (PyTorch Implementation)
# =====================================================================

class KinematicsRegressor(nn.Module):
    """
    Continuous kinematics decoder mapping latent dynamics Z to behavior velocities.
    Used for trial-by-trial velocity decoding (e.g., vx, vy across time bins).
    
    Trained during Phase 1 on source data and FROZEN during Phase 2 & 3.
    """
    def __init__(self, latent_dim: int = 8, behavior_dim: int = 2, l2_reg: float = 1e-4):
        super().__init__()
        self.latent_dim = latent_dim
        self.behavior_dim = behavior_dim
        self.l2_reg = l2_reg
        
        # Linear/Ridge mapping from latent space to velocity
        self.linear = nn.Linear(latent_dim, behavior_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Latent trajectories of shape (batch_size, seq_len, latent_dim)
        Returns:
            v_pred: Predicted velocities of shape (batch_size, seq_len, behavior_dim)
        """
        return self.linear(z)

    def get_l2_loss(self) -> torch.Tensor:
        """Weight decay penalty matching ridge regression regularization."""
        return self.l2_reg * torch.sum(self.linear.weight ** 2)


# =====================================================================
# 2. Target Direction Classifier (PyTorch Implementation)
# =====================================================================

class DirectionClassifier(nn.Module):
    """
    Discrete target classifier mapping latent dynamics Z to reaching targets
    (e.g., 8 center-out reaching directions).
    """
    def __init__(self, latent_dim: int = 8, num_classes: int = 8, pool_type: str = "mean"):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.pool_type = pool_type

        # Linear classifier head
        self.classifier = nn.Linear(latent_dim, num_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Latent trajectories of shape (batch_size, seq_len, latent_dim)
        Returns:
            logits: Target class logits of shape (batch_size, num_classes)
        """
        # Pool latent representation across the time dimension
        if self.pool_type == "mean":
            z_pooled = torch.mean(z, dim=1)  # (batch_size, latent_dim)
        elif self.pool_type == "max":
            z_pooled, _ = torch.max(z, dim=1)
        else:
            z_pooled = z[:, -1, :]  # Final time bin

        logits = self.classifier(z_pooled)
        return logits


# =====================================================================
# 3. Scikit-Learn Offline Ridge Regressor (Convenience Wrapper)
# =====================================================================

class SklearnRidgeVelocityDecoder:
    """
    Standard Ridge Regression decoder matching the paper's baseline evaluation setup.
    Operates on NumPy arrays flattened across time bins.
    """
    def __init__(self, alpha: float = 1.0):
        self.model = Ridge(alpha=alpha)

    def fit(self, z_source, v_source):
        """
        Args:
            z_source: (n_trials, seq_len, latent_dim)
            v_source: (n_trials, seq_len, behavior_dim)
        """
        n_trials, seq_len, d = z_source.shape
        behavior_dim = v_source.shape[-1]

        z_flat = z_source.reshape(-1, d)
        v_flat = v_source.reshape(-1, behavior_dim)

        self.model.fit(z_flat, v_flat)

    def predict(self, z_target):
        """
        Args:
            z_target: (n_trials, seq_len, latent_dim)
        Returns:
            v_pred: (n_trials, seq_len, behavior_dim)
        """
        n_trials, seq_len, d = z_target.shape
        z_flat = z_target.reshape(-1, d)
        v_pred_flat = self.model.predict(z_flat)
        return v_pred_flat.reshape(n_trials, seq_len, -1)


# =====================================================================
# 4. Evaluation Metrics (R-squared, RMSE, Accuracy)
# =====================================================================

def compute_r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Computes coefficient of determination (R^2 in %):
      R^2 = (1 - SS_res / SS_tot) * 100
    """
    # Flatten over batches and time bins
    y_true_flat = y_true.view(-1, y_true.size(-1))
    y_pred_flat = y_pred.view(-1, y_pred.size(-1))

    ss_res = torch.sum((y_true_flat - y_pred_flat) ** 2, dim=0)
    ss_tot = torch.sum((y_true_flat - torch.mean(y_true_flat, dim=0)) ** 2, dim=0)

    # Average R^2 across behavioral coordinates (vx, vy)
    r2_per_dim = 1.0 - (ss_res / (ss_tot + 1e-8))
    return (torch.mean(r2_per_dim) * 100.0).item()


def compute_rmse(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Root-Mean-Square Error: sqrt(mean((y_true - y_pred)^2))
    """
    return torch.sqrt(F.mse_loss(y_pred, y_true)).item()


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Classification accuracy for target direction prediction:
      Acc = sum(pred == target) / N * 100
    """
    preds = torch.argmax(logits, dim=-1)
    acc = (preds == labels).float().mean() * 100.0
    return acc.item()