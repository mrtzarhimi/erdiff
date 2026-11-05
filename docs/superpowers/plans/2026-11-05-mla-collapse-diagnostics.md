# Source-only MLA collapse diagnostics

The user-supplied MLA-only self-check fell from 92.16% to −2.76% source validation R². The diagnostics below investigate this behavior without selecting settings on target test labels.

`--mla_diagnostics` requires `--source_self_check`. It records epoch zero, epochs 1 and 5, every `--mla_diagnostic_every` epochs (default 10), and the final epoch. `--mla_diagnostics_only` records epoch zero and exits before optimizer updates or an aligned checkpoint save.

Artifacts in the chosen save directory:

- `mla_diagnostics.jsonl`: per-latent mean variance, trial-to-trial and temporal variance, mean posterior variance, per-coordinate prediction variance and means, and ratios to initialization. Values use source-standardized decoder units.
- `mla_probe_fixture.pt`: selected source training trial IDs, fixed posterior noise, fixed diffusion noise, diagnostic seed, and probe timesteps. A separate local generator creates these draws. All future records reuse them.
- Each probe reports the actual VP signal coefficient/noise standard deviation and signal-to-noise power ratio, the production MLA residual, gradients with respect to sampled latents, posterior means/log-variances and the read-in, plus a central directional finite difference with the same tensors and noise.

Interpretation: falling trial-to-trial mean and prediction variance supports collapse of behavior-relevant variation. Falling posterior variance supports reduced encoder uncertainty. Positive `mu_radial_dot` means the negative mean-gradient points inward in unconstrained latent coordinates; it does not prove that the coupled read-in update follows that direction. Positive `logvar_grad_sum` means the unconstrained negative log-variance gradient favors reduced variance. A finite-difference discrepancy can also reflect float32 cancellation: inspect analytic/numeric derivatives, absolute error and perturbation size before labeling it a gradient defect. This checks the latent derivative; it does not numerically check every encoder parameter gradient.

Diagnostic forwards disable dropout, preserve model modes on exit, avoid DataLoader iterator RNG draws, and use autograd.grad without accumulating parameter gradients. They use only source training spikes. Source validation R² is reported separately by the existing self-check and never selects a checkpoint.

Execution status: code and focused side-effect/replay tests were written but not run. No dataset reads, model imports, training, or numerical experiments were executed during implementation. Existing uncommitted checkpoint provenance work is separate.
