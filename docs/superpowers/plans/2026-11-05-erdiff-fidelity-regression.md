# ERDiff fidelity and alignment regression plan

## 1. Establish the reference and the regression

- Preserve current commit `675d25d` and pre-change commit `dd59baf`. Commit each subsequent focused correction separately.
- Inspect available before/after results, checkpoints, commands, seeds, and dataset versions. Record source validation performance, pre-alignment target performance, and post-alignment target performance separately.
- Pin the [2023 author snapshot](https://github.com/yulewang97/ERDiff/tree/ae49f74e7fee70a2e3ea8205bf54d937ec552103) and [published paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/7abbcb05a5d55157ede410bb718e32d7-Paper-Conference.pdf). Use the locally available historical source rather than the modified current checkout.
- Create an equation/configuration checklist for preprocessing, VAE, diffusion, alignment, and evaluation. Record paper/code disagreements explicitly. Do not mix the 2023 implementation and later releases into one purported reproduction.

**Deliverable:** a reproducible regression description and source-backed reference configuration. Scores previously selected using target labels remain historical diagnostics, not fair benchmark results.

## 2. Validate the data and isolate recent changes

Before changing model capacity or tuning learning rates:

1. Verify actual sample spacing, spike units, event timestamps, trial durations, retained trial counts, neuron order, and behavior units.
2. Compare our extraction with the historical source. The author snapshot uses pre-extracted trials; recover their event/window definition before declaring the new go-cue window paper-faithful.
3. Compare these changes individually using the same evaluation cohort wherever valid:
   - Peak-centered versus cue-relative windows.
   - Smoothing kernel and boundary handling.
   - Training-data quantity after the three-way split.
   - Source-based versus session-based velocity scaling.
   - Fixed final epoch versus the previous label-selected checkpoint.
4. Treat alternative preprocessing as named experimental configurations. Keep target velocity out of input-window selection in the fair benchmark.
5. Preserve the separate split seed and checkpointed indices. Extend checkpoint metadata with complete preprocessing and model configurations, training seed, code revision, and data fingerprint. Reject incompatible checkpoint/data combinations.

**Checks:** exact normalization round trips, disjoint raw time intervals across partitions, unchanged test membership when the model seed changes, and identical zero-step baseline predictions.

## 3. Verify model mathematics before tuning

Work through these gates in order:

- **Source model:** verify reconstruction, KL, behavior-loss reductions, posterior variance, and decoder units. Keep behavior supervision where the paper's neural experiment specifies it; do not remove it merely because the introductory ELBO omits it.
- **Source density training:** determine whether the selected reference trains diffusion on posterior means or samples, and use that choice consistently in training and alignment.
- **Source latent bank:** disable augmentation and random sample omission when constructing the bank and deterministic initialization statistics. Currently, `augment_multiplier=1` still adds noise.
- **Temporal structure:** trace attention axes and test sensitivity to temporal permutations. Our denoiser lacks explicit temporal position encoding; determine whether this matches the historical architecture. Any added positional encoding must be documented as a variant unless supported by the reference.
- **Diffusion objective:** verify forward-noise coefficients, timestep distribution, noise/score conversion, weighting, and loss reductions with matched tensors, timesteps, and noise.
- **Alignment:** verify gradients reach the read-in through the frozen encoder and denoiser. Compare parameter snapshots to confirm frozen components remain unchanged.
- **Transport loss:** compare values and gradients against the paper's stated entropy-regularized objective. Do not silently replace it with a debiased Sinkhorn divergence.
- **Extra penalties:** expose smoothness, ridge penalties, weight decay, and initialization choices in configuration. Establish each reference setting before testing deviations.

Use small numerical checks and controlled synthetic shifts before expensive training. Include zero-update alignment, source-to-source alignment, and known recoverable linear shifts. Source-to-source deterioration is diagnostic; minimizing latent-density loss does not itself guarantee better behavior decoding.

## 4. Choose meaningful hyperparameters using source-only development

Use the agreed **source-only development protocol**:

- Fit models on source training trials.
- Construct fixed synthetic domains from source validation trials using neuron permutations, gain changes, and controlled neuron removal. Save transformations and corruption seeds.
- Reserve source test trials for confirmation after development choices are frozen.
- Keep all real target behavior labels outside hyperparameter and checkpoint selection.

Start from the verified reference configuration. Express training budgets as optimizer updates and sample exposures, alongside epochs, because augmentation and batch size change the work per epoch.

For alignment development:

1. Compare learning rates `{1e-4, 5e-4, 2e-3}` and budgets `{50, 150}` epochs with the reference objective: six configurations.
2. At the selected learning rate and budget, compare the reference transport weight with `{0, 0.1, 0.5}`, removing duplicates.
3. Evaluate development candidates with model seeds `{0, 1, 2}` and one fixed split seed, `42`.
4. Select by mean source-development decoding R² across fixed shifts; break ties by lower RMSE, then fewer updates. Confirm the selected configuration on source test trials with seeds `{0, 1, 2, 3, 4}`.
5. Log separate loss values and weighted contributions, read-in gradient norms, clipping frequency, latent variance, and denoising error by noise level. Use these measurements to detect domination, collapse, or ineffective guidance.

Do not expand the search until a concrete diagnostic justifies it. Record any expansion before examining real target test scores.

## 5. Final evaluation, acceptance, and execution boundaries

- Freeze configuration and alignment duration before final target evaluation.
- Evaluate pre-alignment and post-alignment states on identical saved target test trials for seeds `{0, 1, 2, 3, 4}`.
- Report per-session R², physical-unit RMSE, paired alignment gains, mean and standard deviation across seeds, and the number of sessions harmed by alignment.
- Identify target results already inspected during development. Prefer untouched sessions for final confirmation where available.
- If alignment still reduces performance, report that outcome and investigate with source diagnostics. Do not select favorable target epochs, seeds, or sessions.

**Acceptance criteria:** computational checks pass; every reference setting has evidence or a documented unresolved discrepancy; evaluation partitions and units are consistent; and the regression is explained by controlled comparisons. Implementation fidelity and reproduction of paper-level performance remain separate conclusions.

**Execution boundary:** inspection and specification may proceed without running the pipeline. No pipeline runs are authorized here. Numerical checks and training experiments require separate execution authorization. When code implementation begins, update the audit with measured evidence and commit each focused change.
