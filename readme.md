# ERDiff: Latent Dynamics Alignment with Diffusion Models

PyTorch implementation of the NeurIPS 2023 paper:  
**"Extraction and Recovery of Spatio-Temporal Structure in Latent Dynamics Alignment with Diffusion Models"** (Yule Wang, Chengrui Li, Zijing Wu, Anqi Wu).

---

## Overview

In brain-computer interface (BCI) decoding, raw neural population activities shift substantially across recording sessions and subjects. While low-dimensional neural manifolds can bridge these shifts, existing alignment approaches typically neglect the spatio-temporal structure across time and latent channels.

**ERDiff** tackles this via a two-phase protocol:
1. **Extraction (Phase 1):** Jointly trains a Sequential Variational Autoencoder (VAE), a kinematics behavior decoder, and a Diffusion Model (DM) over the source domain. The DM employs stacked **Spatio-Temporal Transformer Blocks (STBlocks)** to extract global trajectory dynamics and cross-dimensional correlations.
2. **Recovery / Alignment (Phase 2):** Performs test-time adaptation on the target domain via **Maximum Likelihood Alignment (MLA)** and Sinkhorn divergence. The DM, the VAE decoder, and the behavior decoder are kept frozen; only the linear read-in projection of the encoder is probed to fit the learned source manifold.

---

## Project Structure

```text
.
├── dataloader.py        # Chewie_CO .mat parser, time-binning (39 bins), velocity normalization
├── vae.py               # Sequential VAE (ELU, GRU, Poisson NLL, analytic KL divergence)
├── diffusion.py         # STBlock Transformer denoiser, VP-SDE schedule, MLA & Sinkhorn losses
├── decoder_models.py    # Ridge kinematics regressor, direction classifier, R2 / RMSE metrics
├── train_phase1.py      # Source-domain offline cooperative training loop (Algorithm 1)
├── train_phase2.py      # Unsupervised target-domain linear probing / alignment loop (Algorithm 2)
├── eval.py              # Zero-shot kinematics evaluation & 3D neural manifold visualizer
└── README.md