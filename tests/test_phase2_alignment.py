import torch
import math
from torch.utils.data import DataLoader

from dataloader import NHPDataset
from diffusion import sinkhorn_loss
from train_phase2 import clean_training_view, sample_latent_posterior


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
