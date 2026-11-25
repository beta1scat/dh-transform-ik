# dh-transform-ik

**A General Framework for Analytical Inverse Kinematics Based on D-H Parameter Templates**

This repository provides the Python implementation accompanying *A general framework for the analytical inverse kinematics solution of industrial robots*.

The framework separates robot-specific Modified Denavit–Hartenberg (MDH) representations from configuration-specific analytical solvers. Compatible robot models are mapped to a common parameter template, allowing an analytical solver to be reused across robots in the same mechanical configuration family.

The implemented solver covers the 6R spherical-wrist family represented by the Fanuc LR Mate 200iD template. Extending the framework to another family requires its own template and analytical solver; changing the robot parameter file alone does not provide a solver for a different configuration.

## Features

- MDH parameter normalization and joint-variable mapping for compatible robot models.
- Closed-form IK with eight named shoulder, elbow and wrist candidate slots.
- Cached parameter transformations for repeated target poses.
- Optional reference configurations for resolving singular joint freedoms.
- FK consistency, IK accuracy and timing, wrist-singularity, and Cartesian-path experiments.

## How It Works

1. **Normalize the robot model.** Establish compatibility in joint types, joint count and serial topology, then apply the MDH isomorphism criteria implemented by `transform_dh`.
2. **Transform the target pose.** Remove the fixed base and terminal transforms to express the target in template coordinates.
3. **Solve analytical IK.** Enumerate the shoulder, elbow and wrist branches of the template solver.
4. **Map the joint solutions.** Convert the template joint variables to the target robot's direction and zero-offset conventions.

Unsupported MDH transformations raise `UnsupportedDHTransformationError`. Candidate solutions must be checked against the target robot's joint limits and FK accuracy before use.

## Installation

Requirements: Python 3.9 or later, uv, and NumPy. From the cloned repository:

```bash
cd dh-transform-ik
uv sync
```

The experiment runbook specifies a pinned Python 3.10 environment and additional numerical-IK dependencies for reproducible comparisons.

## Quick Start

### FK → IK → FK

The following example constructs a target from a known joint configuration and computes its analytical candidates. Robot parameter files and FK inputs use degrees; IK outputs use radians. Positions are expressed in millimetres.

```python
import numpy as np
from dh_transform_ik import load_robot, forward_kinematics_chain, solve_ik

template = load_robot("robots/fanuc_lr_mate_200id.json")
robot = load_robot("robots/kuka_kr50_r2500.json")

q_deg = [(link.theta_min + link.theta_max) / 2 for link in robot[:6]]
target = forward_kinematics_chain(robot, q_deg)
solutions = solve_ik(target, template, robot)

for branch, q_rad in solutions.items():
    if not np.all(np.isfinite(q_rad)):
        continue
    reconstructed = forward_kinematics_chain(robot, np.degrees(q_rad).tolist())
    error = np.mean(np.abs(target - reconstructed))
    print(f"{branch}: q_deg={np.degrees(q_rad)}, matrix_error={error:.3e}")
```

A finite candidate is not automatically a joint-limit-valid solution. The experiment utilities perform periodic-equivalence, actual-limit and FK acceptance checks consistently across analytical and numerical methods.

### Cache the Parameter Transformation

```python
from dh_transform_ik import transform_dh, solve_ik

transformed = transform_dh(template, robot)
solutions = solve_ik(target, template, robot, transformed_dh=transformed)
```

For repeated calls on the same robot model, reuse `transformed`.

### Reference Configuration at Singularities

```python
reference_q = np.radians(q_deg)
solutions = solve_ik(
    target,
    template,
    robot,
    transformed_dh=transformed,
    reference_q=reference_q,
)
```

`reference_q` contains six joint angles in the **target robot's joint coordinates**, in radians. For a sequence of targets it can be the previous accepted solution. It is not a required input for independent pose solving.

At a numerically singular wrist, the solver jointly selects a coupled q4/q6 pair within their actual joint limits, minimizing squared joint displacement from the reference. At a shoulder-axis degeneracy, the reference supplies a representative q1. Non-singular poses retain their analytical branch decomposition. Reference-based singular selection does not by itself establish a continuous executable trajectory.

## Configuration Names

| Position | Codes | Meaning |
|---|---|---|
| First | `F` / `B` | Front / Back shoulder branch |
| Second | `U` / `D` | Up / Down elbow branch |
| Third | `N` / `F` | No-flip / Flip wrist branch |

The eight slots are `FUN`, `FUF`, `FDN`, `FDF`, `BUN`, `BUF`, `BDN` and `BDF`.
Branches that do not produce a candidate contain `NaN`. At singularities, multiple slots may represent coincident solutions; eight slots do not necessarily mean eight distinct feasible configurations.

## Robot Models

| Model | Parameter file |
|---|---|
| Fanuc LR Mate 200iD — template | `robots/fanuc_lr_mate_200id.json` |
| Kuka KR 50 R2500 | `robots/kuka_kr50_r2500.json` |
| ABB IRB2600-20/1.65 | `robots/abb_irb2600_20_165.json` |
| Yaskawa GP12 | `robots/yaskawa_gp12.json` |
| Estun ER8-720-MI | `robots/estun_er8_720mi.json` |

Each model contains six joint entries and one terminal fixed-transform entry. Parameter fields are:

| Field | Meaning | Unit |
|---|---|---|
| `alpha` | Link twist | degrees |
| `a` | Link length | mm |
| `thetaOffset` | Fixed joint-angle offset | degrees |
| `d` | Link offset | mm |
| `thetaMin`, `thetaMax` | Joint-variable limits | degrees |

The MDH convention is `Rx(alpha) Tx(a) Rz(thetaOffset + theta) Tz(d)`. A new model in the implemented family must satisfy the transformation criteria. Other families require separate analytical equations and branch definitions.

## Project Structure

```text
dh-transform-ik/
  dh_transform_ik/
    dh.py                         # MDH parameters and forward kinematics
    transform.py                  # Parameter normalization and joint mapping
    ik.py                         # Analytical IK and singular candidate selection
  robots/                         # Five robot parameter sets
  experiments/
    common/                       # Shared FK, metrics, model builder and IO
    paper/                        # Paper reproduction entry and README
    thesis_chapter2/              # Dissertation experiments and README
  data/
    paper/                        # Paper outputs (generated)
    thesis_chapter2/              # Chapter 2 outputs (generated)
  tests/
    test_singular_reference.py    # Singular-reference regression checks
  pyproject.toml
  LICENSE
```

## Experiments

Experiment entry points and outputs are organized separately:

| Workflow | Documentation | Outputs |
|---|---|---|
| Paper method reproduction | [Paper README](experiments/paper/README.md) | `data/paper/` |
| Dissertation Chapter 2 | [Chapter 2 README](experiments/thesis_chapter2/README.md) | `data/thesis_chapter2/` |

Both workflows import the same `dh_transform_ik` implementation. Shared experiment utilities live in `experiments/common/`; sampling, acceptance, timing and report rules are specified by each workflow. Run entries as Python modules from the repository root, for example `uv run python -m experiments.paper.benchmark --analytical-only`.

The paper workflow provides a pose-reconstruction benchmark with a single-call LM baseline. Chapter 2 provides full-range targets, joint-limit-aware acceptance, multi-start numerical baselines, singularity and path analysis. The two protocols have separate outputs and should be reported according to their documented definitions.

Measured results depend on the implementation, protocol and platform. Generated datasets and results are ignored by Git and must be transferred separately when needed.

## Citation

If this code is used in research, please cite:

```bibtex
@article{niu2025general,
  title={A general framework for the analytical inverse kinematics solution of industrial robots},
  author={Niu, Qun and Zhao, Jieliang and Liang, Lulu and Xing, Jin and Li, Hongkun and Wang, Ziru},
  journal={Proceedings of the Institution of Mechanical Engineers, Part C: Journal of Mechanical Engineering Science},
  volume={239},
  number={12},
  pages={4499--4511},
  year={2025}
}
```

## License

MIT License. See [LICENSE](LICENSE).
