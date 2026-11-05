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

## Source-only noising experiments

`--mla_fixed_t` fixes the training corruption time; alternatively `--mla_t_min` and `--mla_t_max` restrict its uniform sampling interval. They change MLA sampling, not the frozen source diffusion schedule. Without these flags, the source training time interval is used. A fixed time still consumes the ordinary uniform-time RNG draw so subsequent posterior/diffusion draws follow the same stream for matched seeds. The independent diagnostic probes always use `--mla_probe_times`, regardless of the training-time policy. Times are not noise standard deviations: the artifact records the actual VP coefficients.

First record probes with `--mla_diagnostics_only`. Then compare a 50-epoch MLA-only source self-check under the original uniform interval and fixed times 0.05, 0.2, and 0.5, holding source checkpoint, seeds, optimizer, batch policy, and probe fixture fixed. The shorter duration is diagnostic because the supplied collapse was already large by epoch 10. It is not a target-selected checkpoint rule. Noise-policy metadata is included in artifacts and the aligned checkpoint. No real target labels are involved.

Neither these training experiments nor the noise-control tests were run during implementation. A restricted/fixed-time policy is an explicit experimental variant rather than the paper's uniform-time default.
