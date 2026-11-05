# Phase 2 balanced optimization batches

The source self-check previously optimized the same 135 trials in fixed batches of 64, 64, and 7. Each batch received one optimizer step, leaving a small, high-variance tail at the end of every epoch.

The source posterior bank and initialization statistics retain their deterministic, complete, unaugmented loaders. Optimization now has a separate sampler: shuffle with a local generator seeded from the model seed, split into `ceil(N / maximum_batch_size)` batches of nearly equal sizes, and include every adaptation sample once per epoch. For 135 trials and maximum batch size 64, the resulting sizes are 45/45/45. The number of updates remains three per epoch. When the explicit augmentation variant is enabled, the same policy covers the repeated augmented dataset.

Both the sampler and DataLoader iterator use dedicated RNG generators so their ordering is independent of posterior/diffusion noise draws. Split seed and saved split indices remain responsible for membership. Sampler configuration and optimizer update count are stored in the target checkpoint.

For the source validation diagnostic, hold the source checkpoint, model seed 42, split seed 42, duration 150, and all remaining settings fixed. Compare alpha=0 (MLA-only objective), alpha=1 (OT-only objective), and alpha=0.5 (both). The printed component values are unweighted; the objective applies alpha. Source validation values are diagnostics and never select target epochs or checkpoints.

Execution boundary: no project code, tests, training, or numerical checks were run. Focused tests for complete coverage, balanced sizes, seeded epoch replay, and global RNG isolation were written but not executed. Earlier uncommitted checkpoint-provenance work is outside this sampler change.
