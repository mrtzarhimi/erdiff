# Controlled Phase 2 comparisons: design

Status: proposed design for implementation with GPT-6 Luna. No implementation or experiment is authorized by this planning artifact. Existing no-execution constraints remain in force.

## Intent

Preserve ERDiff's native ability to align different neuron counts. Make fixed Gaussian draws, optional input-size matching, and loss-gradient calibration independent experimental factors, while strictly freezing the Phase 1 denoiser. Recompare under the same source checkpoint, splits, optimizer and training budget rather than attributing a 50-versus-150-epoch change to an algorithm.

Fixed noise controls Monte Carlo variation; it does not prevent label leakage. Leakage protection comes from training-only adaptation, saved non-overlapping splits, source-fitted normalization, and exclusion of real target behavior from tuning/checkpoint selection. No claim of universal non-degradation or paper-level reproduction follows from these controls.

## Reference and budget

The intended historical reference is the corrected-checkpoint seed-42 run with 150 alignment epochs, mean baseline R² −8.78%, and aligned R² 24.89%. It is not the pre-binning `dd59baf` run. This identification is a planning assumption to confirm from the user's artifacts.

Create a verified run manifest from the original command/configuration, source checkpoint, data fingerprints, and saved source/target splits. Candidate values to verify are Adam, learning rate 0.001, weight decay 0, batch maximum 64, alpha 0.5, OT epsilon 31.2, smoothness 0, target augmentation off, and 150 epochs. The original sampler, initializer policy, variance setting, and precise code revision must be recorded; missing historical fields stay unresolved. A fresh known-config replay is permitted but must not be labeled an exact historical reproduction.

`--reference_run PATH` resolves all baseline parameters. Only declared experimental factor overrides may differ. Conflicting unapproved hyperparameter/seed overrides fail clearly. Record resolved arguments, source/data hashes, code revision, trial IDs, neuron mapping, noise-bank hash, update counts, sample exposures and full-variance forward exposures. Epochs and actual updates are both reported per session: different trial counts can give different updates per epoch. Match budgets between variants within each protocol/session, not by assuming every session has 450 updates. Keep 150 epochs for the primary historical-budget comparison. No target-dependent stopping.

## Arguments and invariants

1. `--alignment_noise_mode {resampled,fixed_bank}`, default `resampled`; `--noise_seed 42`; `--noise_bank_replicates 8`. A local, namespaced generator creates independent posterior noise, diffusion noise, source-posterior noise, and time draws. Fixed-bank entries are indexed by domain, saved trial ID and replicate, not batch position. Reuse the bank across paired runs and cycle replicates by epoch without regenerating Gaussian values. Diagnostic probes use a separate namespace. One Gaussian vector must not be shared by all trials, or reused as both posterior and diffusion noise. The VP family and saved Phase 1 schedule stay unchanged. Extra raw-spike Gaussian augmentation stays off unless explicitly declared. Fixed-bank reuse is an empirical training variant that can overfit its corruptions; independent source-development probe draws remain essential.

2. `--match_input_size`, default false; `--input_mapping_seed 42`. Native mode retains each domain's actual neuron count. Matching transforms target channels to the source checkpoint's input width: retain current ordering and zero-pad if narrower; deterministically select channels if wider; identity if equal. Apply the mapping to the already normalized model input; do not normalize padded zeros as if they were measured spike counts. Persist the exact mapping, its position in preprocessing, and use it in both optimization and evaluation. Never crop/retrain the source model implicitly. This matches tensor dimensions, not neuronal identity. Log both raw target channels and model input width. A known explicit channel map can be supported later but is not inferred from target behavior.

3. Add `--readin_init random` as an explicit policy independent of equal/unequal width; retain `source_or_random` and `ridge` as named existing policies. A size-matching ablation cannot silently change from random initialization to source copy. Resolve initialization from the raw input width before mapping; replay/project the same initialized read-in into the matched-width condition. Padding with zero channels and corresponding weights must give equal zero-step predictions. Wider-target selection discards information and is explicitly a different case, without an equivalence claim. Historical initialization remains a separate reference condition. Source self-check identity starts from source copy; it must not silently become a random-initialization experiment. Zero-padding with a functionally matched linear read-in is mathematically equivalent to native narrow input, so size matching alone is not expected to improve that case.

4. `--log_component_gradients` measures raw and weighted MLA/OT/variance gradient norms and pairwise cosine on fixed training-only probes, without optimizer steps or `.grad` accumulation. `--gradient_balance {none,source_calibrated}`, default `none`. Optional source calibration uses independent frozen source copies, fixed source training probes, and the same noise fixture to estimate RMS gradient norms G_M and G_O. Set c_M=1 and c_O=clip(G_M/G_O, 0.01, 100); if either norm is nonfinite or numerically inactive, disable calibration with an explicit invalid-calibration result instead of dividing by near zero. Store the factors once and reuse across target sessions. Do not calibrate the variance term at source identity: its floor can be inactive there. Its configured lambda remains explicit. Effective loss is `(1-alpha)*(MLA + lambda*V) + alpha*c_O*OT + smooth_weight*smooth`. Norm balance cannot resolve conflicting directions or guarantee semantics; this is a named variant, not a paper setting.

5. `--frozen_check {final,every_epoch}`, default `every_epoch`. Diffusion freezing itself is mandatory in both modes. Set denoiser eval mode, require every denoiser parameter to have requires_grad=false, exclude those parameters from all Phase 2 optimizers, and check exact parameter/buffer equality against an initial snapshot. Reject gradients on denoiser parameters and state drift. Preserve autograd through the denoiser input; wrapping its training forward in no_grad would remove the read-in guidance. Only Phase 1 trains diffusion. Changing bank/mapping/balancing settings never retrains it.

All arguments must propagate through `train_phase2.py`, cross-day orchestration, synthetic development, checkpoint metadata and evaluation where applicable. Evaluation verifies the source/read-in pair and preprocessing/mapping compatibility. The pending provenance edits must be reviewed and integrated deliberately, not lost or silently mixed with these tasks.

## Controlled run matrix

- R150: known-config historical replay, native channels and historical initializer/sampler, no new factors. Missing historical provenance remains explicit.
- C0: controlled baseline with the same 150-epoch hyperparameters, saved checkpoint/splits, balanced sampler, current full-training variance floor (weight 10, ratio 0.9), resampled noise, native size, no gradient balancing. Keep explicit initializer controls as described above. This is a method baseline, not a claim that R150 used a variance floor.
- F: C0 plus fixed-bank noise.
- S: C0 plus size matching, with initializer functionally matched to its native counterpart.
- G: C0 plus source-calibrated MLA/OT gradient scaling.
- ALL: C0 plus F, S and G. Frozen checks run in every case. ALL is not assumed superior.

Source self-check uses the source-copy identity initializer for R150/C0/F/G; equal-size S is an identity mapping and only tests that path's invariance. The actual native-versus-matched size test uses synthetic masked/removed channels and then real unequal-width sessions; preserve explicit initialization equivalence. The existing masked-removal synthetic condition retains input width, so add a distinct `channel_removal` condition that really reduces width. Mark it lossy; permutation and gain retain known inverse read-in calibration. Development cohorts remain source-validation adaptation/evaluation subsets with overlap-safe saved IDs; source test remains reserved.

First inspect source-only results for R150/C0/F/S/G/ALL. Freeze the protocol and configuration before the exploratory cross-day run; previously viewed target sessions cannot become untouched confirmation data. Report paired gains, harmed-session count, per-session/seed variability and invariants, not only mean R². Failed assertions or calibration are failures, not omitted rows or successful results.

## Boundaries for the implementer

Use GPT-6 Luna for sequential, small implementation tasks. Do not delegate or switch models implicitly. Commit each focused task. Do not execute Python, imports, tests, numerical checks, datasets, training, evaluation, downloads or XeLaTeX without a later explicit authorization. Read-only inspection, editing, static diff review and requested Git commits are permitted. Write checks and label them unrun; do not claim red/green verification. Update `noted.tex` without compiling it. Every verification and performance claim remains pending until observed.
