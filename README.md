# dh-transform-ik

**A General Framework for Analytical Inverse Kinematics of 6-DOF Industrial Robots**

One analytical IK solver for *any* 6-DOF robot with a spherical wrist. Instead of deriving a custom IK solver for each robot, supply the robot's Modified D-H parameters and get all 8 joint configurations in microseconds.

## Key Results

| Metric | This Method | Numerical (LM) |
|--------|------------|-----------------|
| Success rate | **100%** | 96-99% |
| Mean error | **~1e-14** | ~1e-4 |
| Speed | **~50 us/call** | ~5-10 ms/call |
| Speedup | **~100-200x** | baseline |
| Solutions returned | **All 8 configs** | 1 config |

Run `uv run --group benchmark python examples/benchmark.py` to reproduce these results with actual measured values on your machine.

### Accuracy (10,000 trials per robot)

| Robot | Success Rate | Mean Error |
|-------|-------------|------------|
| Fanuc LR Mate 200iD | 10000/10000 | 2.1e-14 |
| Kuka KR50 R2500 | 10000/10000 | 1.2e-13 |
| ABB IRB2600-20/1.65 | 10000/10000 | 5.3e-14 |
| Yaskawa GP12 | 10000/10000 | 6.4e-14 |
| Estun ER8-720-MI | 10000/10000 | 2.5e-14 |

## How It Works

```
                    DH Transform                    Analytical IK
                    (Algorithm 1)                   (closed-form)
                                                    
  Robot's MDH   ──────────────>  Template MDH  ──────────────>  8 Solutions
  Parameters         once           Space           per pose     (template)
                                                                    │
                                                                    │ Joint Mapping
                                                                    v
                                                                8 Solutions
                                                                (your robot)
```

**Pipeline:**

1. **Transform DH** (once per robot): Map the robot's Modified D-H parameters to a standard template structure using algebraic commutation rules of homogeneous transformation matrices.

2. **Solve IK** (per target pose): Transform the target pose into the template's coordinate frame, solve the analytical IK (which only needs to be derived once for the template), then map the joint angles back.

3. **Get all 8 configurations**: Front/Back x Up/Down x No-flip/Flip -- returned as a named dictionary.

## Quick Start

```bash
git clone https://github.com/your-username/dh-transform-ik.git
cd dh-transform-ik
uv sync
```

```python
import numpy as np
from dh_transform_ik import load_robot, solve_ik, forward_kinematics_chain

# Load robots (template = Fanuc LR Mate 200iD)
template = load_robot("robots/fanuc_lr_mate_200id.json")
robot = load_robot("robots/kuka_kr50_r2500.json")

# Define target pose (4x4 homogeneous matrix, position in mm)
target = np.eye(4)
target[:3, 3] = [1000, 200, 800]
target[:3, :3] = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]  # example orientation

# Solve IK -- returns all 8 configurations
solutions = solve_ik(target, template, robot)

for name, joints in solutions.items():
    if not any(np.isnan(joints)):
        print(f"{name}: {np.round(np.degrees(joints), 2)}")
```

### FK -> IK Roundtrip

```python
import math
from dh_transform_ik import forward_kinematics_chain, solve_ik, load_robot

template = load_robot("robots/fanuc_lr_mate_200id.json")
robot = load_robot("robots/kuka_kr50_r2500.json")

# Forward kinematics at a known configuration
q_deg = [10, 20, 30, 40, 50, 60]
target = forward_kinematics_chain(robot, q_deg)

# Solve IK
solutions = solve_ik(target, template, robot)

# Verify: FUN config should match the input
q_fun = [math.degrees(j) for j in solutions["FUN"]]
print(f"Input:  {q_deg}")
print(f"Output: {[round(x, 6) for x in q_fun]}")
# Input:  [10, 20, 30, 40, 50, 60]
# Output: [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
```

### Cache for Repeated Calls

```python
from dh_transform_ik import transform_dh, solve_ik

# Pre-compute the DH transformation (done once)
transformed = transform_dh(template, robot)

# Use cached transform for fast repeated IK calls
for target in targets:
    solutions = solve_ik(target, template, robot, transformed_dh=transformed)
```

## Configuration Names

The 8 IK configurations are named using a 3-letter code:

| Code | Meaning |
|------|---------|
| **F** / **B** | **Front** / **Back** -- shoulder configuration (theta1 differs by pi) |
| **U** / **D** | **Up** / **Down** -- elbow configuration (two theta3 solutions) |
| **N** / **F** | **No-flip** / **Flip** -- wrist configuration (theta4/5/6 alternatives) |

Full list: `FUN`, `FUF`, `FDN`, `FDF`, `BUN`, `BUF`, `BDN`, `BDF`

## Supported Robots

Any 6-DOF industrial robot with a spherical wrist (joints 4, 5, 6 axes intersect at a point). The following robots are included as examples:

| Robot | Manufacturer | Reach | Payload |
|-------|-------------|-------|---------|
| LR Mate 200iD | Fanuc | 717 mm | 7 kg |
| KR50 R2500 | Kuka | 2500 mm | 50 kg |
| IRB2600-20/1.65 | ABB | 1650 mm | 20 kg |
| GP12 | Yaskawa | 1440 mm | 12 kg |
| ER8-720-MI | Estun | 720 mm | 8 kg |

To add a new robot, create a JSON file with the Modified D-H parameters (7 entries: 6 joints + end-effector):

```json
[
    {"i": 0, "d": 0, "a": 0, "alpha": 0, "thetaOffset": 0, "thetaMin": -170, "thetaMax": 170},
    {"i": 1, "d": 0, "a": 50, "alpha": -90, "thetaOffset": -90, "thetaMin": -100, "thetaMax": 145},
    ...
]
```

Fields: `alpha` (deg), `a` (mm), `thetaOffset` (deg), `d` (mm), `thetaMin`/`thetaMax` (deg).

## Project Structure

```
dh-transform-ik/
  pyproject.toml         # Project config & dependencies (uv)
  dh_transform_ik/
    __init__.py          # Public API
    dh.py                # DHParam class, FK, utilities
    transform.py         # DH parameter transformation (Algorithm 1)
    ik.py                # Analytical IK solver + high-level solve_ik()
  robots/                # Robot DH parameter JSON files
  examples/
    basic_usage.py       # Simple IK solve + verification
    all_robots.py        # All 5 robots demonstrated
    benchmark.py         # Reproduce paper Tables 3 & 4 (+ numerical comparison)
  tests/
    test_ik.py           # FK -> IK -> FK roundtrip tests
```

## API Reference

### `solve_ik(target_pose, template_dh, robot_dh, transformed_dh=None)`

High-level API. Solves IK in one call.

- **target_pose**: 4x4 numpy array (homogeneous transform, position in mm)
- **template_dh**: Template robot DH parameters (from `load_robot`)
- **robot_dh**: Target robot DH parameters (from `load_robot`)
- **transformed_dh**: Optional pre-computed transform (from `transform_dh`)
- **Returns**: `dict` mapping config names to joint angle lists (radians)

### `load_robot(json_path)`

Load robot DH parameters from a JSON file. Returns a list of `DHParam` objects.

### `transform_dh(template_dh, arbitrary_dh)`

Transform arbitrary robot DH parameters to match the template structure. Returns a list of 8 `DHParam` objects (base + 6 joints + end-effector).

### `inverse_kinematics(target_pose, dh_list)`

Low-level analytical IK solver. Operates in template space.

### `forward_kinematics_chain(dh_list, joint_angles_deg)`

Compute FK for a robot. Returns 4x4 homogeneous transform.

### `CONFIGS`

List of configuration names: `['FUN', 'FUF', 'FDN', 'FDF', 'BUN', 'BUF', 'BDN', 'BDF']`

## Running Tests

```bash
uv run python tests/test_ik.py
```

Expected output:
```
--- DH Transform FK Preservation Test ---
  [PASS] Kuka KR50 R2500: mapped FK error = 5.69e-14
  [PASS] ABB IRB2600: mapped FK error = 5.37e-15
  [PASS] Yaskawa GP12: mapped FK error = 7.16e-15
  [PASS] Estun ER8-720-MI: mapped FK error = 2.73e-15

--- FK -> IK -> FK Roundtrip Tests (100 random configs each) ---
  [PASS] Fanuc LR Mate 200iD (template): 100/100 passed
  [PASS] Kuka KR50 R2500: 100/100 passed
  [PASS] ABB IRB2600: 100/100 passed
  [PASS] Yaskawa GP12: 100/100 passed
  [PASS] Estun ER8-720-MI: 100/100 passed
```

## Requirements

- Python 3.9+
- [uv](https://docs.astral.sh/uv/) (recommended package manager)
- NumPy (installed automatically by `uv sync`)

### Running Examples

```bash
# Basic usage
uv run python examples/basic_usage.py

# All robots demo
uv run python examples/all_robots.py

# Benchmark (analytical only)
uv run python examples/benchmark.py

# Benchmark with numerical comparison (installs roboticstoolbox)
uv run --group benchmark python examples/benchmark.py
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{niu2025general,
  title={A general framework for the analytical inverse kinematics solution of industrial robots},
  author={Niu, Qun and Zhao, Jieliang and Liang, Lulu and Xing, Jin and Li, Hongkun and Wang, Ziru},
  journal={Proceedings of the Institution of Mechanical Engineers, Part C: Journal of Mechanical Engineering Science},
  volume={239},
  number={12},
  pages={4499--4511},
  year={2025},
  publisher={SAGE Publications Sage UK: London, England}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
