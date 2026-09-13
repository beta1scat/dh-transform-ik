"""General analytical IK framework based on D-H parameter templates.

The framework associates each mechanical configuration family with a D-H
template and a separately derived analytical solver. This package currently
implements only the 6R spherical-wrist family represented by the Fanuc LR Mate
200iD template. Compatible same-configuration D-H representations can reuse
that solver when they satisfy the framework's D-H-level isomorphism criteria.

Other families, including UR-type non-spherical-wrist manipulators, require a
configuration-specific template and solver. They are not implemented here, and
passing another template to ``solve_ik`` does not select another solver.

Quick Start:
    >>> import numpy as np
    >>> from dh_transform_ik import load_robot, solve_ik
    >>>
    >>> template = load_robot("robots/fanuc_lr_mate_200id.json")
    >>> robot = load_robot("robots/kuka_kr50_r2500.json")
    >>>
    >>> # Define target pose (4x4 homogeneous matrix, position in mm)
    >>> target = np.eye(4)
    >>> target[:3, 3] = [500, 200, 800]
    >>>
    >>> solutions = solve_ik(target, template, robot)
    >>> for name, joints in solutions.items():
    ...     print(f"{name}: {np.round(np.degrees(joints), 2)}")

Reference:
    Niu Q, Zhao J, Liang L, et al. "A general framework for the analytical
    inverse kinematics solution of industrial robots." Proceedings of the
    Institution of Mechanical Engineers, Part C: Journal of Mechanical
    Engineering Science, 2025, 239(12): 4499-4511.
"""

from .dh import (
    DHParam,
    load_robot,
    forward_kinematics_mdh,
    forward_kinematics_chain,
    normalize_angle,
)
from .transform import (
    UnsupportedDHTransformationError,
    transform_dh,
    transform_joints_all,
    transform_joints_single,
)
from .ik import (
    inverse_kinematics,
    solve_ik,
    CONFIGS,
)

__version__ = "1.0.0"

__all__ = [
    # Data structures
    "DHParam",
    # I/O
    "load_robot",
    # Forward kinematics
    "forward_kinematics_mdh",
    "forward_kinematics_chain",
    # DH transformation (core contribution)
    "UnsupportedDHTransformationError",
    "transform_dh",
    "transform_joints_all",
    "transform_joints_single",
    # Inverse kinematics
    "inverse_kinematics",
    "solve_ik",
    "CONFIGS",
    # Utilities
    "normalize_angle",
]
