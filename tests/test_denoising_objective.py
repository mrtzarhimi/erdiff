import torch

from diffusion import VPSDEScheduler, dsm_loss


class ConstantDenoiser(torch.nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.prediction = torch.nn.Parameter(torch.zeros(shape, dtype=torch.float64))

    def forward(self, latent, time):
        return self.prediction.expand_as(latent)


def test_denoising_l1_uses_element_mean_value_and_gradient():
    model = ConstantDenoiser((1, 2, 2))
    latent = torch.zeros(1, 2, 2, dtype=torch.float64)
    noise = torch.tensor([[[2.0, -0.5], [3.0, 0.0]]], dtype=torch.float64)
    loss = dsm_loss(model, VPSDEScheduler(), latent, noise=noise,
                    timesteps=torch.tensor([0.25], dtype=torch.float64))
    gradient = torch.autograd.grad(loss, model.prediction)[0]
    torch.testing.assert_close(loss, torch.tensor(1.375, dtype=torch.float64))
    torch.testing.assert_close(gradient, torch.tensor([[[-0.25, 0.25], [-0.25, 0.0]]], dtype=torch.float64))


def test_denoising_mean_is_invariant_to_duplicated_trials():
    model = ConstantDenoiser((1, 2, 2))
    latent = torch.zeros(3, 2, 2, dtype=torch.float64)
    noise = torch.tensor([[[2.0, -0.5], [3.0, 0.0]]], dtype=torch.float64).expand_as(latent)
    loss = dsm_loss(model, VPSDEScheduler(), latent, noise=noise,
                    timesteps=torch.full((3,), 0.25, dtype=torch.float64))
    torch.testing.assert_close(loss, torch.tensor(1.375, dtype=torch.float64))


def test_legacy_denoising_mse_remains_explicitly_available():
    model = ConstantDenoiser((1, 2, 2))
    latent = torch.zeros(1, 2, 2, dtype=torch.float64)
    noise = torch.tensor([[[2.0, -0.5], [3.0, 0.0]]], dtype=torch.float64)
    loss = dsm_loss(model, VPSDEScheduler(), latent, loss_type="mse", noise=noise,
                    timesteps=torch.tensor([0.25], dtype=torch.float64))
    torch.testing.assert_close(loss, torch.tensor(3.3125, dtype=torch.float64))
