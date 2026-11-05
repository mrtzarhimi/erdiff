import torch
from torch import nn

from neuron_input_mapping import NeuronInputMap, mapping_from_checkpoint
from train_phase2 import initialize_target_read_in


def test_native_96_to_95_adaptation_zero_pads_one_channel():
    mapping = NeuronInputMap.create(raw_width=95, model_width=96, seed=42)
    spikes = torch.randn(4, 39, 95)

    mapped = mapping.apply(spikes)

    assert mapped.shape == (4, 39, 96)
    assert torch.equal(mapped[..., :95], spikes)
    assert torch.count_nonzero(mapped[..., 95]) == 0


def test_identity_mapping_preserves_channels_exactly():
    mapping = NeuronInputMap.create(raw_width=96, model_width=96, seed=42)
    spikes = torch.randn(2, 5, 96)

    assert torch.equal(mapping.apply(spikes), spikes)
    assert mapping.padding == 0
    assert torch.equal(mapping.kept_indices, torch.arange(96))


def test_wider_target_selection_is_deterministic_and_sorted():
    left = NeuronInputMap.create(raw_width=120, model_width=96, seed=42)
    right = NeuronInputMap.create(raw_width=120, model_width=96, seed=42)
    other = NeuronInputMap.create(raw_width=120, model_width=96, seed=43)

    assert torch.equal(left.kept_indices, right.kept_indices)
    assert left.kept_indices.unique().numel() == 96
    assert torch.equal(left.kept_indices, left.kept_indices.sort().values)
    assert not torch.equal(left.kept_indices, other.kept_indices)


def test_narrow_zero_padding_preserves_linear_predictions_with_projected_weights():
    raw_readin = nn.Linear(5, 7)
    mapping = NeuronInputMap.create(raw_width=5, model_width=8, seed=42)
    mapped_readin = mapping.map_readin(raw_readin)
    spikes = torch.randn(3, 11, 5)

    raw_prediction = raw_readin(spikes)
    mapped_prediction = mapped_readin(mapping.apply(spikes))

    assert torch.equal(mapped_readin.bias, raw_readin.bias)
    assert torch.count_nonzero(mapped_readin.weight[:, 5:]) == 0
    assert torch.allclose(mapped_prediction, raw_prediction, rtol=0, atol=1e-6)


def test_random_initialization_stays_random_when_raw_width_matches_source():
    source_readin = nn.Linear(6, 4)
    target_readin = initialize_target_read_in(
        source_read_in=source_readin,
        src_loader=None,
        tgt_loader=None,
        n_src_neurons=6,
        n_tgt_neurons=6,
        hidden_dim=4,
        device=torch.device("cpu"),
        init_mode="random",
    )

    assert not torch.equal(target_readin.weight, source_readin.weight)
    assert not torch.equal(target_readin.bias, source_readin.bias)


def test_checkpoint_mapping_is_required_and_validated_for_matched_runs():
    state = NeuronInputMap.create(95, 96, seed=42).state_dict()
    mapping, model_width = mapping_from_checkpoint(
        {"match_input_size": True, "input_mapping": state, "model_input_width": 96}, 95
    )
    assert mapping.model_width == model_width == 96

    try:
        mapping_from_checkpoint({"match_input_size": True, "model_input_width": 96}, 95)
    except ValueError as error:
        assert "missing its saved input mapping" in str(error)
    else:
        raise AssertionError("Matched checkpoint without a mapping was accepted.")

    try:
        mapping_from_checkpoint(
            {"match_input_size": True, "input_mapping": state, "model_input_width": 96}, 94
        )
    except ValueError as error:
        assert "raw width differs" in str(error)
    else:
        raise AssertionError("Mapping with an incompatible target width was accepted.")
