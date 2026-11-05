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
