import pytest
import torch

from vae import (SequentialVAE, SequentialVAEEncoder, kl_divergence_loss,
                 posterior_configuration, posterior_logvar_mode_from_checkpoint,
                 sample_latent_posterior)


def test_default_encoder_learns_log_variance_beyond_previous_bounds():
    encoder = SequentialVAEEncoder(3, hidden_dim=4, latent_dim=2, dropout=0).double()
    with torch.no_grad():
        encoder.fc_logvar.weight.zero_()
        encoder.fc_logvar.bias.copy_(torch.tensor([-7.0, 2.0], dtype=torch.float64))
    mu, logvar = encoder(torch.ones(2, 3, 3, dtype=torch.float64))

    torch.testing.assert_close(logvar, torch.tensor([-7.0, 2.0], dtype=torch.float64).expand(2, 3, 2))
    kl = kl_divergence_loss(mu, logvar)
    gradient = torch.autograd.grad(kl, encoder.fc_logvar.bias)[0]
    torch.testing.assert_close(gradient, 0.5 * (torch.exp(logvar[0, 0]) - 1))
    assert gradient[0] < 0 < gradient[1]


def test_posterior_sampling_uses_unclamped_learned_scale():
    model = SequentialVAE(3, latent_dim=2, hidden_dim=4, dropout=0).double().eval()
    with torch.no_grad():
        model.encoder.fc_logvar.weight.zero_()
        model.encoder.fc_logvar.bias.copy_(torch.tensor([-7.0, 2.0], dtype=torch.float64))
    mu, logvar = model.encoder(torch.ones(1, 3, 3, dtype=torch.float64))
    noise = torch.ones_like(mu)
    sample = sample_latent_posterior(mu, logvar, noise)
    torch.testing.assert_close(sample - mu, torch.exp(torch.tensor([-3.5, 1.0], dtype=torch.float64)).expand_as(mu))


def test_legacy_checkpoint_retains_clamped_outputs_after_loading():
    model = SequentialVAE(3, latent_dim=2, hidden_dim=4, dropout=0).eval()
    with torch.no_grad():
        model.encoder.fc_logvar.weight.zero_()
        model.encoder.fc_logvar.bias.copy_(torch.tensor([-7.0, 2.0]))
    checkpoint = {"vae_state_dict": model.state_dict()}
    mode = posterior_logvar_mode_from_checkpoint(checkpoint)
    restored = SequentialVAE(3, latent_dim=2, hidden_dim=4, dropout=0,
                             posterior_logvar_mode=mode).eval()
    restored.load_state_dict(checkpoint["vae_state_dict"])
    _, logvar = restored.encoder(torch.ones(1, 3, 3))
    torch.testing.assert_close(logvar, torch.tensor([-5.0, 0.0]).expand_as(logvar))


def test_saved_unclamped_configuration_survives_checkpoint_reload():
    source = SequentialVAE(3, latent_dim=2, hidden_dim=4, dropout=0).eval()
    with torch.no_grad():
        source.encoder.fc_logvar.weight.zero_()
        source.encoder.fc_logvar.bias.copy_(torch.tensor([-7.0, 2.0]))
    checkpoint = {"posterior_config": dict(source.encoder.posterior_config),
                  "vae_state_dict": source.state_dict()}
    mode = posterior_logvar_mode_from_checkpoint(checkpoint)
    restored = SequentialVAE(3, latent_dim=2, hidden_dim=4, dropout=0,
                             posterior_logvar_mode=mode).eval()
    restored.load_state_dict(checkpoint["vae_state_dict"])
    _, logvar = restored.encoder(torch.ones(1, 3, 3))
    torch.testing.assert_close(logvar, torch.tensor([-7.0, 2.0]).expand_as(logvar))


@pytest.mark.parametrize("config", [
    {}, {"logvar_mode": "unknown"},
    {"logvar_mode": "unclamped", "minimum": -5, "maximum": 0},
])
def test_incomplete_or_incompatible_posterior_metadata_is_rejected(config):
    with pytest.raises(ValueError, match="posterior"):
        posterior_logvar_mode_from_checkpoint({"posterior_config": config})


def test_new_checkpoint_cannot_silently_fall_back_to_legacy_without_metadata():
    with pytest.raises(ValueError, match="posterior"):
        posterior_logvar_mode_from_checkpoint({"training_config": {"posterior_logvar_mode": "unclamped"}})


def test_conflicting_training_and_posterior_configuration_is_rejected():
    with pytest.raises(ValueError, match="posterior"):
        posterior_logvar_mode_from_checkpoint({
            "posterior_config": posterior_configuration("unclamped"),
            "training_config": {"posterior_logvar_mode": "legacy_clamped"},
        })
