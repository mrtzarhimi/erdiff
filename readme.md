# ERDiff: Latent Dynamics Alignment with Diffusion Models

PyTorch implementation of the NeurIPS 2023 paper:  
**"Extraction and Recovery of Spatio-Temporal Structure in Latent Dynamics Alignment with Diffusion Models"** (Yule Wang, Chengrui Li, Zijing Wu, Anqi Wu).

---

## Overview

In brain-computer interface (BCI) decoding, raw neural population activities shift substantially across recording sessions and subjects. While low-dimensional neural manifolds can bridge these shifts, existing alignment approaches typically neglect the spatio-temporal structure across time and latent channels.

**ERDiff** tackles this via a two-phase protocol:
1. **Extraction (Phase 1):** Jointly trains a Sequential Variational Autoencoder (VAE), a kinematics behavior decoder, and a Diffusion Model (DM) over the source domain. The DM employs stacked **Spatio-Temporal Transformer Blocks (STBlocks)** to extract global trajectory dynamics and cross-dimensional correlations.
2. **Recovery / Alignment (Phase 2):** Performs target-domain adaptation via **Maximum Likelihood Alignment (MLA)** and the configured Sinkhorn loss. Target behavior labels are excluded from adaptation and checkpoint selection; the final configured epoch is evaluated on a disjoint held-out test partition.

Evaluation uses saved deterministic trial partitions, go-cue-relative input windows, source-training velocity normalization, and test-only target metrics. `--split_seed` controls the partition independently of `--seed`, which controls model randomness. Checkpoints preserve their split indices and evaluation reuses them. Overlapping windows are kept in the same partition, and spike smoothing is applied within each trial window. Legacy checkpoints without split metadata must be regenerated.

Phase 2 transport now follows the pinned 2023 author code: POT `ot.sinkhorn2`
on `(squared_trajectory_distance + 1e-4) / max(cost)`, with default
`--ot_epsilon 0.01` and `--sinkhorn_iters 1000`. Install the added dependency
with `pip install -r requirements-transport.txt` before using it. No package
installation was performed during this change. The objective returns transport
cost without an explicit entropy term, and differentiates through both the
solver and maximum normalization. The numerical solver now uses
`method="sinkhorn_log"` and float64 transport arithmetic to avoid the reported
iteration-zero failure of conventional float32 Sinkhorn at `reg=0.01`.
It targets the same transport objective; it is a numerical departure from
the author's default solver, not a new entropy term or a detached-plan gradient.
It differs from the paper's written
entropy-inclusive objective and from our former implementation.

Do not reuse `--phase2_ot_epsilon 31.2` from previous commands: it was calibrated
for unnormalized costs. Old gradient-calibration files and controlled reference
manifests must be regenerated for this objective. New manifests record the
objective and installed POT version; incomplete or mismatched references fail
validation. Use fresh output directories and compare fixed budgets/splits.

New Phase 1 models default to `--posterior_logvar_mode unclamped`: the learned
linear head returns log variance directly and sampling uses
`z = mu + exp(0.5 * logvar) * eta`. The former bounds `[-5, 0]` are available
as `--posterior_logvar_mode legacy_clamped` for controlled comparisons. The
existing bias -3 / Xavier gain 0.1 initialization is retained; it is a local
initialization choice and does not constrain the learned outputs.

To use unclamped variance, retrain **both the source VAE and its diffusion
model together in Phase 1**, then use that new checkpoint for Phase 2.
Phase 2 always restores the source checkpoint's posterior setting and keeps
the diffusion model frozen. Old checkpoints lacking posterior metadata
retain the legacy clamp; loading them does not activate the new behavior.
Both evaluators and synthetic checks follow the same checkpoint setting.
Unclamped variance and the mean-trajectory variance floor are separate:
the former controls uncertainty in `q(z|x)`, the latter opposes collapse of
posterior means. Neither change establishes recovery of decoding performance.

Phase 2 now stops on nonfinite posterior parameters/scales/samples, MLA or
transport intermediates, loss components, and read-in gradients. It rejects
nonfinite gradient norms before the optimizer step. On failure the training
entry point writes `phase2_numerical_failure.json` with the stage, epoch,
batch, trial IDs, and affected tensor shapes/dtypes/ranges; it raises instead
of saving a completed aligned checkpoint. Synthetic runs write the same
report in their run directory. Both evaluators reject nonfinite predictions
instead of reporting NaN benchmark metrics. Existing failed checkpoints
and results remain invalid; use fresh run directories and recalibrate for
the new transport solver. These runtime checks have not been executed here.

Diffusion training now defaults to `--diffusion_loss l1`, the element-mean
absolute noise-prediction error used in the pinned 2023 author code. The old
element-mean squared loss is available as `--diffusion_loss mse`.
Checkpoints record the loss/reduction. Adopting L1 requires Phase 1 retraining;
existing diffusion weights trained with MSE do not become L1-trained weights.
Posterior sampling, the continuous VP schedule and model architecture remain
separate local choices; this loss correction alone is not full reproduction.

Phase 2 now defaults to `--alignment_objective author_2023` (cross-day:
`--phase2_alignment_objective author_2023`). Its reference loss is
`1.2 * mean(SmoothL1(eps_pred, eps, beta=1)) + OT + skilling_divergence`.
MLA uses posterior means; OT retains sampled source/target posteriors, as in
the author alignment code. `--alpha` applies only to the legacy
`--alignment_objective paper_squared` mode. Author mode uses coefficients 1
for its complete MLA branch and OT; the 1.2 residual coefficient is already
inside that MLA branch. Logs expose denoising and Skilling separately.

The historical helper computes `mean(z_noisy * VJP(z_noisy, z0, ones) / 50)`.
For our linear VP forward marginal its VJP equals the detached signal
coefficient, which we use directly with the same value/gradient. This is the
author helper, not the paper's VP drift-divergence formula. We retain our
continuous VP schedule; the author uses a 50-step discrete schedule, so the
port remains a documented loss variant on our existing schedule.

Author mode requires a newly trained element-mean L1 source checkpoint and
rejects old or unspecified diffusion loss metadata. Use fresh output folders
and source calibration artifacts. For the reference **loss** comparison,
set gradient balancing to `none`, smoothness to `0`, and variance weight to
`0`. Keeping them enabled is supported and explicitly changes the objective:
`MLA + variance_weight*V + c_OT*OT + smooth_weight*S` in author mode. Previous
weights were assessed with a trajectory-sum squared MLA loss and need
source-only reassessment at this new scale. Fixed Gaussian banks, input-size
mapping, saved splits, final-epoch evaluation and frozen diffusion are retained.

---

## Project Structure

```text
.
├── dataloader.py        # Chewie_CO .mat parser, cue-relative windows, trial splits, source velocity scaling
├── vae.py               # Sequential VAE (ELU, GRU, Poisson NLL, analytic KL divergence)
├── diffusion.py         # STBlock Transformer denoiser, VP-SDE schedule, MLA & Sinkhorn losses
├── decoder_models.py    # Ridge kinematics regressor, direction classifier, R2 / RMSE metrics
├── train_phase1.py      # Source-domain offline cooperative training loop (Algorithm 1)
├── train_phase2.py      # Unsupervised target-domain linear probing / alignment loop (Algorithm 2)
├── evaluate.py          # Single-session held-out evaluation & manifold visualizer
├── evaluate_cross_day.py # Cross-day held-out evaluation
└── README.md
