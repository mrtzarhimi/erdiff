# Fair ERDiff Evaluation Design

## Intent

Make cross-session ERDiff results independent of target behavior labels during alignment and checkpoint selection, and calculate target metrics in units consistent with the source-trained decoder.

## Goals

- Use deterministic, disjoint trial-level train, validation, and final test partitions.
- Use target training spikes for adaptation only; do not inspect target validation/test velocities during alignment or select a checkpoint from them.
- Keep target final test trials untouched until final evaluation.
- Fit velocity normalization on source training trials and use that same transform for source decoder outputs and target labels. Convert metrics back to physical velocity units where applicable.
- Keep source validation R² selection as the source supervised training criterion.

## Non-goals

- Changing peak-speed centering, spike smoothing, neural lag, model architecture, diffusion objective, Sinkhorn definition, or auxiliary losses. The audit marks these as reproduction/research decisions requiring separate evidence.
- Launching project code, tests, experiments, dataset reads, or training in this pass, per the user's instruction.

## Approaches

1. **Recommended: fixed deterministic three-way trial split plus fixed epoch alignment.** Simple to audit, no target labels or test metrics in model selection, and no extra validation objective whose behavior could introduce selection bias.
2. Select epochs using an unlabeled target validation objective. This avoids label use, but selecting on a target-derived objective still adapts to the validation distribution and adds a new criterion not specified by the paper.
3. Continue target validation R² selection but report a separate test score. Rejected because labels still influence model selection and undermine an unsupervised adaptation claim.

## Design calls

- Use a seeded local RNG for trial partitioning so constructing loaders does not reset the global training/noise RNG.
- Default split proportions: 70% train, 15% validation, 15% test, disjoint at trial level. Validation is reserved for diagnostics and is not used by target alignment. Final target evaluation uses only test.
- Source Phase 1 continues selecting checkpoints using labeled source validation data, which is part of supervised source training. Save source training velocity mean/std in the source checkpoint.
- Normalize target labels with source training statistics. Denormalize source-decoder predictions and target labels to physical units before reporting RMSE; R² is then computed on the same physical values.
- Update the audit with resolution and static evidence. Since project execution remains prohibited, do not claim runtime/test verification.

## Files and expected outcome

- `dataloader.py`: deterministic split into three disjoint trial partitions; support source-provided velocity statistics; optional stats return for Phase 1.
- `train_phase1.py`: consume train/validation/test loaders; persist source velocity statistics in the source checkpoint.
- `train_phase2.py`: consume target training trials only for alignment; remove target-label checkpoint selection and per-epoch target labeled evaluation; save the final fixed-epoch read-in and source normalization metadata.
- `evaluate_cross_day.py` and `evaluate.py`: evaluate only the target test partition and apply source velocity statistics consistently.
- `cross_day.py`: retain orchestration but ensure the evaluation path consumes the untouched test partition.
- `ERDIFF_AUDIT.md`: record implemented resolutions, limitations, and verification status.

## Risks

- Existing checkpoints lack source velocity statistics and need a clear compatibility error or an explicit legacy pathway; silently fitting target statistics is prohibited.
- Very small sessions may produce an empty split or too few batches; split construction should validate counts and fail clearly.
- Trial-centered windows may overlap in raw time; no raw timeline group audit can be completed without reading session data, which is not authorized here.
