# Chapter 2 experiment data

All records in this directory are generated from the robot JSON files and the
scripts under `experiments/`. They are not reconstructed from dissertation tables.

## Metadata

- `metadata/robot_registry.csv`: model identity, parameter file, commit and checksum.
- `metadata/environment.json`: environment used for the latest DH export.
- `metadata/analytical_environment.json`: analytical IK, timing, path and singularity environment.
- `metadata/legacy_numerical_environment.json`: pinned Paper 1 numerical baseline environment.

## Processed data

- `dh_original.csv`, `dh_transformed.csv`: source and generated MDH tables.
- `dh_validation.csv`: 50,000 original/transformed FK comparisons.
- `d_offset_validation.csv`: synthetic nonzero `d_offset` propagation records.
- `ik_targets.csv`: 40,000 frozen FK-derived targets shared by every IK method.
- `ik_analytical_trials.csv`: all eight analytical branches for every target.
- `ik_numerical_legacy_trials.csv`: three RTB 1.0.3 numerical methods for every target.
- `ik_timing_trials.csv`: complete online analytical IK latency per call.
- `path_trials.csv`: offline Yaskawa line and arc branch-continuity records.
- `singularity_trials.csv`: Yaskawa wrist-singularity records.

The corresponding `*_summary.csv` files are generated from these trial-level
records. For legacy numerical methods, the dissertation uses
`unified_combined_failures`: a trial succeeds only when the toolbox reports
`success=True` and the millimetre-normalized matrix error satisfies
`ME <= 1e-3`. The other columns are retained for protocol diagnostics.
