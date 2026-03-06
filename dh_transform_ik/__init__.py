"""
dh-transform-ik: A General Framework for Analytical Inverse Kinematics.

This package provides an analytical inverse kinematics solver that works for
any 6-DOF industrial robot with a spherical wrist. Instead of writing a
separate IK solver for each robot, you only need the robot's Modified D-H
parameters -- the framework transforms them to a standard template and solves
IK analytically, yielding all 8 configurations at ~200x the speed of numerical
methods with machine-precision accuracy.

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
