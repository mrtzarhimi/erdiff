# Fair ERDiff Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make target alignment label-blind and evaluate only disjoint held-out trials with consistent source-trained velocity scaling and behavior-independent windows.

**Architecture:** The loader will partition trials deterministically with a local RNG, construct cue-relative windows without consulting target speed, and normalize behavior from source training statistics. Phase 2 will see target adaptation spikes only and use a fixed configured epoch count; evaluation will load the untouched test split and report predictions/targets in physical units.

**Tech Stack:** Python, PyTorch, NumPy, SciPy, existing DataLoader/checkpoint code.

**Spec:** `docs/superpowers/specs/2026-11-05-fair-er-diff-evaluation-design.md`

## Global Constraints

- Do not run project code, tests, numerical experiments, data reads, or training in this pass, per the user's explicit instruction.
- Do not use target velocity values for window construction, alignment, checkpoint selection, or any pre-test choice.
- Keep train/validation/test trial indices disjoint and deterministic; use source train velocity statistics for both domains.
- Preserve the existing model, diffusion, Sinkhorn, smoothing, and peak-lag research choices outside this scope.
- Commit each focused change independently; do not include unrelated files.

## Review Focus

- Target velocity must not select trial windows — pin with a static regression test around the extraction function.
- Split construction must be deterministic and disjoint — pin with an index-partition test.
- Target test labels must not enter alignment or selection — pin with a static/structural test that Phase 2 consumes train loader only and evaluation consumes test loader only.
- Target predictions and labels must share the source transform before metric computation — pin with a normalization/denormalization unit test.
- Legacy checkpoints without normalization stats must fail clearly in evaluation rather than silently use target-derived scaling — pin with a compatibility test.

---

### Task 1: Behavior-independent windows and disjoint data splits

**Files:**
- Modify: `dataloader.py`
- Create: `tests/test_evaluation_protocol.py`

**Interfaces:**
- `get_nhp_dataloaders(...)` returns `(train_loader, val_loader, test_loader, n_neurons, velocity_stats)`.
- `velocity_stats` contains source-fit mean/std tensors; optional supplied stats are applied without refitting.
- Train/validation/test defaults are 70%/15%/15% with seeded local-generator partitioning.

- [ ] Add tests for deterministic/disjoint index partitioning and cue-window construction that remains unchanged when velocity values change.
- [ ] Implement local-RNG split helper and cue-relative fixed window extraction; retain the explicit fixed neural lag, but do not inspect speed to choose windows.
- [ ] Add optional source velocity stats input; compute stats only from training trials when absent.
- [ ] Static review the updated loader and tests; do not execute them under the no-execution constraint.
- [ ] Commit only loader/tests with `fix: prevent behavior leakage in ERDiff data windows`.

### Task 2: Target alignment without label-based selection

**Files:**
- Modify: `train_phase1.py`
- Modify: `train_phase2.py`
- Modify: `cross_day.py`
- Modify: `tests/test_evaluation_protocol.py`

**Interfaces:**
- Phase 1 stores `velocity_mean` and `velocity_std` in its checkpoint.
- Phase 2 reads target training loader only, performs the configured number of epochs, saves the final read-in, and never evaluates or selects using target behavior labels.
- Phase 2 checkpoint stores source velocity stats for downstream evaluation.

- [ ] Add structural assertions for the fixed-epoch checkpoint path and target train-only loader use.
- [ ] Update Phase 1 to consume train/validation/test loaders and persist source stats.
- [ ] Update Phase 2 to use source-fitted stats, remove per-epoch target validation evaluation and best-R² selection, and save the final trained read-in.
- [ ] Update cross-day orchestration output names/arguments as needed so each session's Phase 2 checkpoint remains distinct.
- [ ] Static review all target velocity accesses; no tests or code will be run.
- [ ] Commit only this change with `fix: remove target-label alignment selection`.

### Task 3: Fair final evaluation in physical velocity units

**Files:**
- Modify: `evaluate.py`
- Modify: `evaluate_cross_day.py`
- Modify: `tests/test_evaluation_protocol.py`

**Interfaces:**
- Evaluation loaders use the deterministic test partition and the source training normalization statistics saved in the Phase 2 checkpoint.
- Metrics and plots compare both decoder predictions and target labels after conversion to physical velocity units.
- Checkpoints lacking source statistics raise a descriptive error.

- [ ] Add tests for exact inverse scaling and legacy checkpoint rejection.
- [ ] Change both evaluation entry points to load the test partition and apply source stats to predictions/labels before metrics and plots.
- [ ] Inspect the entire reporting path for use of validation data or target-fitted normalization.
- [ ] Update `ERDIFF_AUDIT.md` with changes and mark runtime verification not performed.
- [ ] Static review only; no project code, tests, data, or training executed.
- [ ] Commit evaluation/audit as `fix: evaluate ERDiff on untouched target trials`.

