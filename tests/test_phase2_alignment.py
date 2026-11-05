import torch
import math
from torch.utils.data import DataLoader

from dataloader import NHPDataset
from diffusion import VPSDEScheduler, sinkhorn_loss
from train_phase2 import clean_training_view
from vae import sample_latent_posterior


def test_clean_training_view_visits_each_source_trial_once_without_noise():
    spikes = torch.arange(15, dtype=torch.float32).reshape(5, 3, 1)
    dataset = NHPDataset(
        spikes, torch.zeros(5, 3, 2), torch.zeros(5, dtype=torch.long),
        is_train=True, augment_multiplier=3,
    )
    clean = clean_training_view(DataLoader(dataset, batch_size=4, drop_last=True), batch_size=4)
    observed = torch.cat([batch["spikes"] for batch in clean])

    assert torch.equal(observed, spikes)
    assert len(clean.dataset) == 5


def test_alignment_samples_posterior_while_encoder_is_in_eval_mode():
    mu = torch.tensor([[[2.0]]])
    logvar = torch.zeros_like(mu)
    noise = torch.tensor([[[3.0]]])

    assert torch.equal(sample_latent_posterior(mu, logvar, noise), torch.tensor([[[5.0]]]))


def test_vp_forward_noise_matches_constant_beta_marginal():
    scheduler = VPSDEScheduler(beta_min=2.0, beta_max=2.0)
    z0 = torch.tensor([[[2.0]]])
    noise = torch.tensor([[[3.0]]])
    noisy, returned_noise = scheduler.add_noise(z0, torch.tensor([0.5]), noise)
    expected = 2.0 * math.exp(-0.5) + 3.0 * math.sqrt(1.0 - math.exp(-1.0))

    assert torch.allclose(noisy, torch.tensor([[[expected]]]))
    assert torch.equal(returned_noise, noise)


def test_transport_uses_full_squared_trajectory_cost_and_gradient():
    source = torch.tensor([[[0.0, 0.0]]])
    target = torch.tensor([[[2.0, 0.0]]], requires_grad=True)
    loss = sinkhorn_loss(source, target, epsilon=0.1)
    gradient = torch.autograd.grad(loss, target)[0]

    assert torch.allclose(loss, torch.tensor(4.0))
    assert torch.allclose(gradient, torch.tensor([[[4.0, 0.0]]]))


def test_transport_value_includes_entropy_term():
    source = torch.zeros(2, 1, 1)
    target = torch.zeros(2, 1, 1)

    loss = sinkhorn_loss(source, target, epsilon=2.0)

    assert torch.allclose(loss, torch.tensor(-2.0 * math.log(4)), atol=1e-6)
