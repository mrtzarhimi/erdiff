import torch
from torch.utils.data import DataLoader

from dataloader import NHPDataset
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
