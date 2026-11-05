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

- [x] Read the existing dirty diff first; identify which provenance edits are ours and preserve all other work.
- [x] Write checks named `test_rejects_undeclared_hyperparameter_override`, `test_mismatched_source_or_split_fails`, and `test_missing_historical_fields_prevent_exact_reproduction_claim`.
- [x] Implement reference validation and resolved-config serialization. Reuse the pending hashing/pairing helper after review rather than adding another incompatible fingerprint implementation.
- [x] Require exact saved target indices where a reference provides them; validate raw-window and data compatibility. Do not load historical read-in weights as an accidental initializer.
- [x] Count optimizer updates and MLA exposures from the actual sampler; count full-variance forwards separately. Store requested and completed budgets.
- [ ] When verification is authorized: `pytest tests/test_phase2_run_manifest.py tests/test_checkpoint_provenance.py -q`; expected tests pass. Until then record static review only.
- [x] Commit: `feat: lock Phase 2 reference configuration and artifact provenance`.

**Execution note:** implementation and tests are written; test execution remains intentionally pending under the user's no-execution instruction. Static review only is claimed.

### Task 2: Enforce the frozen denoiser invariant

**Files:** create `phase2_freeze_guard.py`, `tests/test_phase2_freeze_guard.py`; modify `train_phase2.py` and the synthetic harness.

**Interfaces:** `FrozenDiffusionGuard(model)` stores an independent CPU snapshot of every state_dict entry; `verify(optimizers: list) -> dict` checks exact state equality, requires_grad=false, no parameter gradients, eval mode, and optimizer exclusion. `--frozen_check final|every_epoch` controls frequency, not whether freezing is required.

- [x] Write tests for unchanged state, accidental optimizer inclusion, buffer mutation, and preserved gradient from denoiser output to its input while denoiser parameter gradients stay absent.
- [x] Apply freeze/guard after loading the Phase 1 state and before optimization. Final verification is mandatory even when epoch checks are enabled.
- [x] On a failed invariant, stop and save failure metadata rather than emitting successful benchmark metrics.
- [ ] When authorized: `pytest tests/test_phase2_freeze_guard.py -q`; expected tests pass. No execution in this implementation pass.
- [x] Commit: `fix: verify frozen diffusion state throughout Phase 2`.

**Execution note:** freeze checks and failure reporting are integrated in the real Phase 2 and source-synthetic paths. The test file is written but remains unrun under the user's no-execution instruction.

### Task 3: Add a trial-indexed fixed Gaussian bank

**Files:** create `alignment_noise_bank.py`, `tests/test_alignment_noise_bank.py`; modify `dataloader.py`, Phase 2, diffusion MLA helper and synthetic checks.

**Interfaces:** `SourcePosteriorBank` holds `mu`, `logvar`, and `trial_ids`; replace current tuple consumers consistently. `NoiseBatch` holds posterior eta, diffusion epsilon and uniform time draws. `AlignmentNoiseBank.get(domain: str, trial_ids: Tensor, replicate: int) -> NoiseBatch`. Domain namespaces distinguish source/target and training/probes. `NHPDataset(..., trial_ids=None)` preserves original extracted IDs across repeated views.

- [x] Write checks for exact replay under batch permutation, independent eta/epsilon streams, distinct probe draws, unchanged Gaussian arrays across noise-level ablations, and no global RNG changes.
- [x] Add `--alignment_noise_mode resampled|fixed_bank`, `--noise_seed`, `--noise_bank_replicates` (8). Default remains resampled. Key each fixed draw by domain, stable trial ID and replicate; cycle replicate `epoch_index % R`. Save the bank and its hash for reuse, rejecting incompatible identities/shapes.
- [x] Fix posterior noise for source OT samples as well as target posterior/diffusion draws in fixed-bank mode. Preserve source/target independence even in a self-check. Source-bank trial selection and balanced sample order retain independent saved RNG streams.
- [x] Route existing fixed-time/ranged-time controls through stored uniform draws. Never replace Gaussian noise with all zeros or one vector shared across all trials.
- [x] Record that finite-bank fitting is a variant and use independent source-development probe noise to inspect corruption overfitting.
- [ ] When authorized: `pytest tests/test_alignment_noise_bank.py tests/test_mla_noise_levels.py -q`; expected tests pass. Otherwise mark unrun.
- [x] Commit: `feat: add reproducible trial-indexed Gaussian noise banks`.

**Execution note:** fixed draws and their cache are integrated in Phase 2 and source-synthetic development; the requested tests are written but remain unrun under the user's no-execution instruction.

### Task 4: Add optional width matching without hiding initialization changes

**Files:** create `neuron_input_mapping.py`, `tests/test_neuron_input_mapping.py`; modify Phase 2, both evaluators, cross-day and synthetic shifts.

**Interfaces:** `NeuronInputMap` records raw width, model width, kept indices and padding; `apply(spikes) -> Tensor` transforms inputs; `map_readin(initial_readin) -> nn.Linear` preserves the initialized linear function for the zero-padding case. Serialize/restore mappings with target checkpoints.

- [x] Write tests for native 96-to-95 adaptation, identity width, deterministic wider-target selection, narrow zero-padding, and padded zero-step predictions using projected common weights and unchanged bias. Tests are present but unrun.
- [x] Add `--match_input_size` (off), `--input_mapping_seed` (42), and explicit `--readin_init random`. Resolve native initialization from raw width before mapping; retain existing named policies for reference runs. Do not let processed equal width silently trigger source-copy initialization.
- [x] Apply the saved mapping after normalization and before the read-in in both adaptation and evaluation; padding must remain zero in model-input units. Construct model input width from the checkpoint rather than raw data width. Reject missing/incompatible mappings for matched checkpoints. Report both channel counts.
- [x] Add a genuinely reduced-width `channel_removal` synthetic shift, distinct from existing zero-mask removal. Keep the same removed indices across candidates. The retained channel mapping is known for synthetic controls, not inferred from real target labels.
- [x] Record wider-target selection as information loss, not a functionally equivalent padding experiment.
- [ ] When authorized: execute the focused tests above. Not run under the standing no-execution instruction.
- [x] Commit: `feat: add optional source-width input mapping with explicit initialization`.

### Task 5: Measure conflicts and add source-calibrated gradient scaling

**Files:** create `alignment_gradient_diagnostics.py`, `tests/test_alignment_gradient_diagnostics.py`; modify Phase 2, MLA diagnostics and manifest consumers.

**Interfaces:** `component_gradient_report(losses: dict, params: tuple) -> dict` returns raw/weighted norms and pairwise cosines without accumulating `.grad`; `calibrate_source_gradient_scale(...) -> dict` returns immutable c_M=1 and bounded c_O from fixed source training probes, or explicit invalid status.

- [x] Write checks with analytic aligned/opposed/zero gradients, no parameter/optimizer/RNG mutation, inactive source variance penalty, and exact reuse of saved calibration factors. Tests are present but unrun.
- [x] Add `--log_component_gradients` and `--gradient_balance none|source_calibrated` (none). Log effective coefficients and cosine signs; do not infer gradient dominance from scalar loss values.
- [x] Calibrate on an independent source model/read-in copy with frozen denoiser and source-only fixed fixtures. Set c_O=clip(G_M/G_O, 0.01, 100), using weighted RMS norms over probes; declare calibration invalid for nonfinite or effectively zero norms (threshold 1e-12 in the stored calibration dtype). Do not amplify an inactive variance gradient.
- [x] Apply `(1-alpha)*(MLA + lambda*V) + alpha*c_O*OT`; preserve the current objective exactly when balance=none. Reuse source factors through the calibration artifact; calibration never uses target behavior or target R².
- [x] Save calibration sample IDs, schedule, independent fixed-fixture identity, raw norms and effective factors. Document that scaling is a variant and cannot correct opposed directions or semantic ambiguity by itself.
- [ ] When authorized: execute `pytest tests/test_alignment_gradient_diagnostics.py -q`. Not run under the standing no-execution instruction.
- [x] Commit: `feat: diagnose and optionally calibrate Phase 2 loss gradients`.

### Task 6: Expose individual and combined controlled runs

**Files:** create `controlled_phase2_runs.py`, `tests/test_controlled_phase2_runs.py`; modify `cross_day.py`, `source_synthetic_checks.py`, and `noted.tex`.

**Interfaces:** CLI `--reference_run PATH --protocol source_self|synthetic|cross_day --case reference|control|fixed_noise|matched_size|gradient_balance|all --output_root PATH`. Base seeds default 42 and must agree with the reference. Cases resolve to explicit flags and a saved resolved manifest before any future training starts.

- [x] Write checks that all cases share baseline hyperparameters and a 150-epoch per-session budget, change only declared factors, use unique output directories, restore saved target splits, reject raw-window overlap and verify forwarding of frozen/noise/mapping/gradient controls. Tests remain unrun.
- [x] Define cases exactly as in the spec: reference R150; controlled C0 with full variance weight 10/floor 0.9; F/S/G individual factors; ALL combines them. Record source self-copy identity and that equal-width source self-check does not demonstrate unequal-width performance.
- [x] Keep actual channel removal in the source-only development matrix. Preserve disjoint adaptation/evaluation source-validation intervals and source-test reservation. Candidates use the same cohort and epoch count; update counts and sample exposures are recorded.
- [x] Forward controls through orchestration and manifests. Existing output roots are rejected, and a specified missing Phase 1 checkpoint fails rather than retraining.
- [x] Update `noted.tex` with independent Gaussian streams, budget matching, width mapping limitations, gradient calibration math, freeze assertions and the ablation matrix. XeLaTeX was not run.
- [ ] When authorized: `pytest tests/test_controlled_phase2_runs.py -q`; expected tests pass. No source/target pipeline execution is currently permitted.
- [x] Commit: `feat: orchestrate budget-matched Phase 2 ablations`.

**Execution note:** the CLI, source-only protocols, cross-day split/fingerprint gate, tests, plan and research note are implemented. Runtime tests, Phase 1/2, cross-day evaluation and XeLaTeX were not run under the no-execution instruction. Cross-day use remains contingent on preparing a manifest with saved per-session splits, fingerprints and preprocessing.

## Implemented checking commands (execution remains disabled)

After implementation/review and a reference manifest with required provenance have been prepared, the user can request execution. The JSON must include an R150 resolved configuration, source checkpoint/data paths, saved source splits, and (for cross-day) a `target_sessions` map keyed by exact session filename. Each target entry must include `target_split_indices` with indexed raw-window `window_intervals`, `data_fingerprint.sha256`, and session preprocessing provenance. The cross-day launcher hashes each session file and rejects missing or mismatched artifacts.

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

The reference fixes model/split/noise/mapping seeds at 42; each stream has its own namespace. Every case verifies the frozen denoiser. The ALL case activates fixed Gaussian banks, optional width matching, component-gradient logging and source-calibrated gradient scaling in addition to C0's full-training variance penalty. Its improvement is unknown and cannot be presumed from activating more options. Each case uses an isolated output directory; existing directories are rejected. A specified missing Phase 1 checkpoint is an error. Synthetic cases retain disjoint source-validation adaptation/evaluation intervals and reserve the source test split.

## GPT-6 Luna implementation prompt

Read the design and this plan before editing. Implement one task per turn/commit, preserving the dirty provenance/audit work and logging decisions. Do not execute project code, imports, tests, data reads, numerical experiments, training, evaluation, downloads or XeLaTeX. Write the listed checks, inspect every diff, and explicitly mark runtime verification unperformed. Follow the defined interfaces; do not invent a new mathematical objective, change Phase 1 training, unfreeze diffusion, or tune to previously seen target scores. If historical evidence is missing, mark it unresolved and distinguish a named known-config replay from exact reproduction; continue independent implementation work. Stop for a research decision only when a required choice cannot be resolved from the design or a proposed invariant conflicts with the actual data. After each task report its commit and remaining unverified checks. At the end provide the implemented individual-case and ALL commands, with matched 150-epoch budgets and base seeds 42. Do not spawn agents or claim tests/performance passed.
