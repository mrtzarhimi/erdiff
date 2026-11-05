import json

import torch
from torch import nn

from dataloader import NHPDataset
from diffusion import VPSDEScheduler
from mla_diagnostics import MLADiagnostics
from vae import SequentialVAE


class LinearDenoiser(nn.Module):
    def forward(self, z, t):
        return 0.25 * z


def test_probes_preserve_parameters_gradients_modes_and_training_rng(tmp_path):
    vae = SequentialVAE(n_neurons=2, hidden_dim=4, latent_dim=2)
    vae.freeze_for_alignment()
    denoiser = LinearDenoiser()
    spikes = torch.arange(24, dtype=torch.float32).reshape(3, 4, 2) / 24
    dataset = NHPDataset(spikes, torch.zeros(3, 4, 2), torch.zeros(3, dtype=torch.long))
    before = {name: value.clone() for name, value in vae.state_dict().items()}
    for parameter in vae.encoder.read_in.parameters():
        parameter.grad = torch.ones_like(parameter)
    modes = [module.training for module in vae.modules()]
    rng = torch.random.get_rng_state().clone()

    diagnostics = MLADiagnostics(dataset, torch.arange(3), 2, [0.05, 0.5], 42, 3, str(tmp_path))
    diagnostics.record(0, vae, denoiser, VPSDEScheduler(), torch.device("cpu"), 2)
    diagnostics.record(1, vae, denoiser, VPSDEScheduler(), torch.device("cpu"), 2)

    assert all(torch.equal(value, before[name]) for name, value in vae.state_dict().items())
    assert all(torch.equal(parameter.grad, torch.ones_like(parameter)) for parameter in vae.encoder.read_in.parameters())
    assert modes == [module.training for module in vae.modules()]
    assert torch.equal(rng, torch.random.get_rng_state())
    rows = [json.loads(line) for line in (tmp_path / "mla_diagnostics.jsonl").read_text().splitlines()]
    assert rows[0]["probes"] == rows[1]["probes"]
    assert rows[0]["variance"] == rows[1]["variance"]
    assert rows[0]["probes"][0]["fd_absolute_error"] < 0.01
