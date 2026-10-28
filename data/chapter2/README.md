# Chapter 2 experiment data

All records in this directory are generated from the robot JSON files and the
scripts under `experiments/`. They are not reconstructed from dissertation tables.

## Metadata

- `metadata/robot_registry.csv`: model identity, parameter file, commit and checksum.
- `metadata/environment.json`: environment used for the latest DH export.
- `metadata/analytical_environment.json`: analytical IK, timing, path and singularity environment.

## Processed data

- `dh_original.csv`, `dh_transformed.csv`: source and generated MDH tables.
- `dh_validation.csv`: 50,000 original/transformed FK comparisons.
- `d_offset_validation.csv`: synthetic nonzero `d_offset` propagation records.
- `ik_targets.csv`: 40,000 frozen FK-derived targets shared by every IK method.
- `path_trials.csv`: offline Yaskawa line and arc branch-continuity records.
- `singularity_trials.csv`: Yaskawa wrist-singularity records.

The corresponding `*_summary.csv` files are generated from trial-level records.

## Unified recomputation

See [the uv runbook](../../experiments/IK_COMPARISON_RUNBOOK.md). Runs are
written under `ik_recompute_20261028/`. Each complete run contains `manifest.json`,
`initial_guesses.csv`, `attempts.csv`, `targets.csv` and `summary.csv`.
The common criterion is finite, joint-limit-valid, periodic-equivalent solutions
with millimetre-normalized `ME <= 1e-3`. Solver status is diagnostic. Numerical
methods try at most ten shared starts. Timing includes retries and acceptance.
The export script produces both replacement LaTeX tables from complete runs.
