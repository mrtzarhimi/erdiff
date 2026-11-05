# Controlled Phase 2 Comparisons Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task by task. The user intends GPT-6 Luna to implement it. Do not dispatch agents or change models automatically. Steps use checkbox syntax for tracking.

**Goal:** Make Phase 2 noise, input width and gradient balancing independently configurable under matched budgets and mandatory frozen-diffusion invariants.

**Architecture:** Small modules own run manifests, fixed noise banks, neuron mappings, freeze checks and gradient diagnostics. Existing Phase 2, source-only synthetic checks and cross-day entry points consume these modules through explicit interfaces. A controlled-run wrapper resolves one immutable reference configuration and changes only declared experimental factors.

**Tech Stack:** Existing Python/PyTorch stack; standard-library hashing/JSON; no new packages. XeLaTeX source documentation only.

**Spec:** `docs/superpowers/specs/2026-11-05-controlled-phase2-design.md` is the authority for choices and limitations.

## Global Constraints

- No Python execution, imports, tests, data reads, numerical experiments, training, evaluation, downloads or XeLaTeX compilation without later authorization. Write meaningful checks and mark them unrun. Read-only repository inspection, editing, static diff review and requested commits are permitted.
- Diffusion is trained only in Phase 1; Phase 2 freezes all its parameters and buffers while preserving input gradients. Every run verifies that invariant.
- Native unequal input widths remain supported and default. Matching width does not imply matching neurons.
- Fixed Gaussian draws improve reproducibility, not leakage protection. No real target behavior enters optimization, gradient calibration, configuration selection or checkpoint selection.
- Model, split, noise and mapping base seeds are 42 in requested comparisons. Distinct generator namespaces keep independent streams independent despite that shared base value.
- Match source/checkpoint/data/splits and resolved hyperparameters, including 150 epochs in the primary historical-budget comparison. Record actual per-session updates and exposures; do not assume all sessions have the same trial count.
- Commit each task separately. Preserve the existing uncommitted provenance changes and audit edits until their explicit integration in Task 1; do not blanket-stage unrelated work.
- Existing variance stabilization remains an experimental variant. Keep a historical reference replay separate from the controlled variance-weight-10 baseline.

## Review Focus

- Legacy runs with missing provenance must be labeled unresolved or known-config replay, never silently promoted to verified reproduction.
- Noise draws must remain keyed to trial IDs when batch order/size or input channel count changes, and diagnostic noise must not be reused for fitting.
- A padded target must receive its saved mapping and compatible read-in in evaluation, with equal zero-step predictions under a functionally matched padded initialization.
- Zero or inactive component gradients must not produce huge calibration weights; the variance gradient is not calibrated at the inactive source floor.
- A frozen denoiser must retain nonzero input-gradient capability; parameter equality and requires_grad flags alone do not test this.

---

### Task 1: Bind the reference configuration and budgets

**Files:** create `phase2_run_manifest.py`, `tests/test_phase2_run_manifest.py`; review/integrate the pending `checkpoint_provenance.py` and its tests; modify Phase 1/2, both evaluators and orchestration only as needed for provenance.

**Interfaces:** `load_reference(path: str) -> dict`; `resolve_run(reference: dict, factor_overrides: dict) -> dict`; `save_run_manifest(path: str, resolved: dict) -> None`. Manifest schema version 1 contains source/data hashes, source and target split artifacts, resolved optimizer/loss/noise/sampler/initializer/preprocessing settings, code revision, seeds, epochs, update counts and exposures. Unknown historical fields carry explicit unresolved status.

- [ ] Read the existing dirty diff first; identify which provenance edits are ours and preserve all other work.
- [ ] Write checks named `test_rejects_undeclared_hyperparameter_override`, `test_mismatched_source_or_split_fails`, and `test_missing_historical_fields_prevent_exact_reproduction_claim`.
- [ ] Implement reference validation and resolved-config serialization. Reuse the pending hashing/pairing helper after review rather than adding another incompatible fingerprint implementation.
- [ ] Require exact saved target indices where a reference provides them; validate raw-window and data compatibility. Do not load historical read-in weights as an accidental initializer.
- [ ] Count optimizer updates and MLA exposures from the actual sampler; count full-variance forwards separately. Store requested and completed budgets.
- [ ] When verification is authorized: `pytest tests/test_phase2_run_manifest.py tests/test_checkpoint_provenance.py -q`; expected tests pass. Until then record static review only.
- [ ] Commit: `feat: lock Phase 2 reference configuration and artifact provenance`.

### Task 2: Enforce the frozen denoiser invariant

**Files:** create `phase2_freeze_guard.py`, `tests/test_phase2_freeze_guard.py`; modify `train_phase2.py` and the synthetic harness.

**Interfaces:** `FrozenDiffusionGuard(model)` stores an independent CPU snapshot of every state_dict entry; `verify(optimizers: list) -> dict` checks exact state equality, requires_grad=false, no parameter gradients, eval mode, and optimizer exclusion. `--frozen_check final|every_epoch` controls frequency, not whether freezing is required.

- [ ] Write tests for unchanged state, accidental optimizer inclusion, buffer mutation, and preserved gradient from denoiser output to its input while denoiser parameter gradients stay absent.
- [ ] Apply freeze/guard after loading the Phase 1 state and before optimization. Final verification is mandatory even when epoch checks are enabled.
- [ ] On a failed invariant, stop and save failure metadata rather than emitting successful benchmark metrics.
- [ ] When authorized: `pytest tests/test_phase2_freeze_guard.py -q`; expected tests pass. No execution in this implementation pass.
- [ ] Commit: `fix: verify frozen diffusion state throughout Phase 2`.

### Task 3: Add a trial-indexed fixed Gaussian bank

**Files:** create `alignment_noise_bank.py`, `tests/test_alignment_noise_bank.py`; modify `dataloader.py`, Phase 2, diffusion MLA helper and synthetic checks.

**Interfaces:** `SourcePosteriorBank` holds `mu`, `logvar`, and `trial_ids`; replace current tuple consumers consistently. `NoiseBatch` holds posterior eta, diffusion epsilon and uniform time draws. `AlignmentNoiseBank.get(domain: str, trial_ids: Tensor, replicate: int) -> NoiseBatch`. Domain namespaces distinguish source/target and training/probes. `NHPDataset(..., trial_ids=None)` preserves original extracted IDs across repeated views.

- [ ] Write checks for exact replay under batch permutation, independent eta/epsilon streams, distinct probe draws, unchanged Gaussian arrays across noise-level ablations, and no global RNG changes.
- [ ] Add `--alignment_noise_mode resampled|fixed_bank`, `--noise_seed`, `--noise_bank_replicates` (8). Default remains resampled. Key each fixed draw by domain, stable trial ID and replicate; cycle replicate `epoch_index % R`. Save the bank and its hash for reuse, rejecting incompatible identities/shapes.
- [ ] Fix posterior noise for source OT samples as well as target posterior/diffusion draws in fixed-bank mode. Preserve source/target independence even in a self-check. Source-bank trial selection and balanced sample order retain independent saved RNG streams.
- [ ] Route existing fixed-time/ranged-time controls through stored uniform draws. Never replace Gaussian noise with all zeros or one vector shared across all trials.
- [ ] Record that finite-bank fitting is a variant and use independent source-development probe noise to inspect corruption overfitting.
- [ ] When authorized: `pytest tests/test_alignment_noise_bank.py tests/test_mla_noise_levels.py -q`; expected tests pass. Otherwise mark unrun.
- [ ] Commit: `feat: add reproducible trial-indexed Gaussian noise banks`.

### Task 4: Add optional width matching without hiding initialization changes

**Files:** create `neuron_input_mapping.py`, `tests/test_neuron_input_mapping.py`; modify Phase 2, both evaluators, cross-day and synthetic shifts.

**Interfaces:** `NeuronInputMap` records raw width, model width, kept indices and padding; `apply(spikes) -> Tensor` transforms inputs; `map_readin(initial_readin) -> nn.Linear` preserves the initialized linear function for the zero-padding case. Serialize/restore mappings with target checkpoints.

- [ ] Write tests for native 96-to-95 adaptation, identity width, deterministic wider-target selection, narrow zero-padding, and exact padded zero-step predictions using projected common weights and unchanged bias.
- [ ] Add `--match_input_size` (off), `--input_mapping_seed` (42), and explicit `--readin_init random`. Resolve native initialization from raw width before mapping; retain existing named policies for reference runs. Do not let processed equal width silently trigger source-copy initialization.
- [ ] Apply the saved mapping after normalization and before the read-in in both adaptation and evaluation; padding must remain zero in model-input units. Construct model input width from the checkpoint rather than raw data width. Reject missing/incompatible mappings for matched checkpoints. Report both channel counts.
- [ ] Add a genuinely reduced-width `channel_removal` synthetic shift, distinct from existing zero-mask removal. Keep the same removed indices across candidates. The retained channel mapping is known for synthetic controls, not inferred from real target labels.
- [ ] Record wider-target selection as information loss, not a functionally equivalent padding experiment.
- [ ] When authorized: `pytest tests/test_neuron_input_mapping.py tests/test_synthetic_shifts.py -q`; expected tests pass. Otherwise mark unrun.
- [ ] Commit: `feat: add optional source-width input mapping with explicit initialization`.

### Task 5: Measure conflicts and add source-calibrated gradient scaling

**Files:** create `alignment_gradient_diagnostics.py`, `tests/test_alignment_gradient_diagnostics.py`; modify Phase 2, MLA diagnostics and manifest consumers.

**Interfaces:** `component_gradient_report(losses: dict, params: tuple) -> dict` returns raw/weighted norms and pairwise cosines without accumulating `.grad`; `calibrate_source_gradient_scale(...) -> dict` returns immutable c_M=1 and bounded c_O from fixed source training probes, or explicit invalid status.

- [ ] Write checks with analytic aligned/opposed/zero gradients, no parameter/optimizer/RNG mutation, inactive source variance penalty, and exact reuse of saved calibration factors.
- [ ] Add `--log_component_gradients` and `--gradient_balance none|source_calibrated` (none). Log effective coefficients and cosine signs; do not infer gradient dominance from scalar loss values.
- [ ] Calibrate on an independent source model/read-in copy with frozen denoiser and source-only fixed fixtures. Set c_O=clip(G_M/G_O, 0.01, 100), using RMS norms over probes; declare calibration invalid for nonfinite or effectively zero norms (threshold 1e-12 in the stored calibration dtype). Do not amplify an inactive variance gradient.
- [ ] Apply `(1-alpha)*(MLA + lambda*V) + alpha*c_O*OT`; preserve the current objective exactly when balance=none. Reuse source factors across target sessions; never fit them against target behavior or choose them to improve a target R².
- [ ] Save calibration sample IDs, schedule, noise-bank identity, raw norms and effective factors. Note that scaling is a variant and cannot correct opposed directions or semantic ambiguity by itself.
- [ ] When authorized: `pytest tests/test_alignment_gradient_diagnostics.py -q`; expected tests pass. Otherwise mark unrun.
- [ ] Commit: `feat: diagnose and optionally calibrate Phase 2 loss gradients`.

### Task 6: Expose individual and combined controlled runs

**Files:** create `controlled_phase2_runs.py`, `tests/test_controlled_phase2_runs.py`; modify `cross_day.py`, `source_synthetic_checks.py`, and `noted.tex`.

**Interfaces:** CLI `--reference_run PATH --protocol source_self|synthetic|cross_day --case reference|control|fixed_noise|matched_size|gradient_balance|all --output_root PATH`. Base seeds default 42 and must agree with the reference. Cases resolve to explicit flags and a saved resolved manifest before any future training starts.

- [ ] Write checks that all cases share baseline hyperparameters and per-session budget, change only declared factors, use unique output directories, restore saved target splits, and propagate frozen checks/mappings/noise controls to evaluation.
- [ ] Define cases exactly as in the spec: reference R150; controlled C0 with full variance weight 10/floor 0.9; F/S/G individual factors; ALL combines them. Include source self-copy identity and matched-initialization shape controls. A source self-check with equal width does not demonstrate unequal-width performance.
- [ ] Add real channel removal to the source-only development matrix. Preserve adaptation/evaluation interval grouping and source-test reservation. Match candidates' optimizer updates within each protocol and report different cohort exposures.
- [ ] Forward all new arguments through orchestration and checkpoint metadata. Do not overwrite existing runs or silently retrain Phase 1 when a specified checkpoint is missing; fail clearly.
- [ ] Update `noted.tex` with independent Gaussian streams, budget matching, width mapping limitations, gradient calibration math, freeze assertions and the ablation matrix. Do not compile.
- [ ] When authorized: `pytest tests/test_controlled_phase2_runs.py -q`; expected tests pass. No source/target pipeline execution is currently permitted.
- [ ] Commit: `feat: orchestrate budget-matched Phase 2 ablations`.

## Planned checking commands (not implemented CLI yet)

After implementation/review and a verified `reference_r150.json` have been prepared, the user can request execution. These examples describe the intended interface and must not be pasted into the current code as if available today.

```bash
# Independent source self-checks, all using the same reference budget.
for case in control fixed_noise matched_size gradient_balance all; do
  python controlled_phase2_runs.py \
    --reference_run ./reference_r150.json --protocol source_self \
    --case "$case" --output_root ./controlled_self42
done

# Meaningful input-size and recovery checks on source-only shifts.
for case in control fixed_noise matched_size gradient_balance all; do
  python controlled_phase2_runs.py \
    --reference_run ./reference_r150.json --protocol synthetic \
    --case "$case" --output_root ./controlled_synthetic42
done

# Only after source-only decisions are frozen: one exploratory all-enabled run.
python controlled_phase2_runs.py \
  --reference_run ./reference_r150.json --protocol cross_day \
  --case all --output_root ./controlled_cross_day_all42
```

The reference fixes model/split/noise/mapping seeds at 42; each stream has its own namespace. Every case verifies the frozen denoiser. The ALL case activates fixed Gaussian banks, optional width matching, component-gradient logging and source-calibrated gradient scaling in addition to C0's full-training variance penalty. Its improvement is unknown and cannot be presumed from activating more options.

## GPT-6 Luna implementation prompt

Read the design and this plan before editing. Implement one task per turn/commit, preserving the dirty provenance/audit work and logging decisions. Do not execute project code, imports, tests, data reads, numerical experiments, training, evaluation, downloads or XeLaTeX. Write the listed checks, inspect every diff, and explicitly mark runtime verification unperformed. Follow the defined interfaces; do not invent a new mathematical objective, change Phase 1 training, unfreeze diffusion, or tune to previously seen target scores. If historical evidence is missing, mark it unresolved and distinguish a named known-config replay from exact reproduction; continue independent implementation work. Stop for a research decision only when a required choice cannot be resolved from the design or a proposed invariant conflicts with the actual data. After each task report its commit and remaining unverified checks. At the end provide the implemented individual-case and ALL commands, with matched 150-epoch budgets and base seeds 42. Do not spawn agents or claim tests/performance passed.
