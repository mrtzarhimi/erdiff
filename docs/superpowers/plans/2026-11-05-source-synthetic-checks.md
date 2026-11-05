# Full-pool variance and source-only synthetic checks

## Full-training variance option

`train_phase2.py --mla_variance_scope full_train` recomputes differentiable encoder means on every unique clean adaptation training trial at each optimizer step. It uses neither augmented copies nor held-out trials. The source reference is frozen source-training mean variance. The minibatch MLA path and update count remain unchanged. The default remains `batch`; zero variance weight still disables the constraint. Full-pool graphs increase memory/computation and are not replaced with detached moment estimates.

## Development protocol

`source_synthetic_checks.py` loads only one source checkpoint/session. Source training trials provide the posterior bank and variance reference. The saved source validation trials are split into disjoint adaptation/evaluation groups using split seed 42, keeping overlapping raw windows together. Both groups must have at least two trials. With few validation trials these cohorts can be small and results uncertain. Source test and real target metrics are not used.

Transforms are fixed and saved using shift seed 42:

- Identity control.
- Full channel permutation; exactly recoverable with the corresponding read-in column permutation.
- Per-channel gains uniformly drawn from [0.5, 1.5); exactly recoverable by inverse read-in scaling.
- Zero-masking approximately 20% of channels, retaining input width. This is lossy and is not equivalent to changing the model's input neuron count.

Oracle read-ins calibrate recoverability for identity/permutation/gain but are never passed to the optimizer as initialization. Each actual run starts from the same source read-in, resets model/optimizer and noise seeds, and uses the same split and shift for weights 1 and 10. Adaptation batches contain spikes only. Source validation labels measure development decoding at fixed intervals; they never select an epoch. Final duration is 50 epochs by default. Changing target scores is not part of this protocol.

Small development cohorts may form only one optimization batch. For example, 14 adaptation trials with maximum batch size 64 give one update per epoch, whereas the original 135-trial source self-check gives three. Compare recorded optimizer updates and trial exposures, not epoch counts alone. Synthetic recovery after a short budget is a diagnostic, not a convergence claim.

## Artifacts

`synthetic_manifest.pt` saves actual trial indices, overlap intervals, shift parameters, seeds, arguments, and checkpoint/data fingerprints. `summary.csv` records clean/oracle/pre/post R², raw-unit RMSE, paired gain, adaptation variance ratios and update counts. Each condition saves `history.json`, `synthetic_readin.pt`, and optional fixed-noise diagnostic artifacts. Synthetic checkpoints are separate from real-target evaluation checkpoints. The source validation set already selected the original checkpoint, so this is development evidence, not an unbiased final test.

## Verification boundary

Focused tests were written for live full-pool gradients, exact permutation/gain inverses, reproducible transforms, and non-overlapping development cohorts. They were not run. No imports, numerical checks, data loads, training, or XeLaTeX compilation were executed. Results remain unknown. Existing uncommitted provenance work is excluded from these changes.
