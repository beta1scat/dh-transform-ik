# Paper Method Reproduction

This workflow reproduces the FK → IK → FK evaluation process accompanying
*A general framework for the analytical inverse kinematics solution of industrial robots*.
It uses the shared `dh_transform_ik` implementation and the five robot models in
`robots/`.

## Protocol

- Generate joint configurations uniformly with a 5% margin at each joint limit;
  target seed is 42.
- Generate targets through the original MDH forward kinematics.
- Analytical method: 10,000 targets per robot by default, cached MDH transformation,
  eight named candidate slots, and the first candidate passing FK reconstruction.
- Numerical baseline: one Robotics Toolbox `ikine_LM` call per target with default
  initialization and termination settings; 200 targets per robot by default.
- Each method uses a prefix of the same robot-specific target list. Sample counts
  are recorded separately; the baseline count can be set by argument.
- Pose success requires finite joint values and a matrix mean absolute error no
  greater than `1e-3`, with translation expressed in millimetres. The LM baseline
  additionally requires its `success` status to be true.
- Joint-limit rejection is not part of this pose-reconstruction criterion.
- Timing includes only the solver call, excluding cached model construction,
  target generation, FK acceptance and file output.

This is the paper-style reproduction protocol. Chapter 2 uses a different
joint-limit-aware, multi-start comparison; see [its README](../thesis_chapter2/README.md).

## Run

Execute from the repository root. Commands use pinned versions through uv.

Analytical method only:

```bash
uv run --python 3.10 --with "numpy==1.23.5" python -m experiments.paper.benchmark --analytical-only --analytical-trials 10000 --seed 42 --margin 0.05 --output-dir data/paper/analytical
```

Analytical method and single-call LM baseline:

```bash
uv run --python 3.10 --with "roboticstoolbox-python==1.0.3" --with "spatialmath-python==1.0.5" --with "scipy==1.8.1" --with "numpy==1.23.5" python -m experiments.paper.benchmark --analytical-trials 10000 --numerical-trials 200 --seed 42 --margin 0.05 --output-dir data/paper/comparison
```

To use equal target counts, set `--numerical-trials 10000`. Keep the other settings
fixed when comparing measurements. Each output directory must be new.

## Outputs

| File | Contents |
|---|---|
| `targets.csv` | Source joint configurations and complete target transforms |
| `trials.csv` | Per-target pose success, solver status, error, selected branch and time |
| `summary.csv` | Robot/method sample counts, successes, mean accepted error and solver time |
| `manifest.json` | Protocol arguments, source hashes, environment and completion status |

Failed targets remain in the denominator and timing mean. Reconstruction error
is averaged over successful targets. Source joint angles are not provided as IK
initial guesses or reference configurations. The results measure the shared
implementation under this documented protocol.
