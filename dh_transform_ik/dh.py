"""
D-H parameter data structures, forward kinematics, and utility functions.

This module provides the core data structures for Modified Denavit-Hartenberg
(MDH) parameters and the fundamental operations needed for robot kinematics.

Reference:
    Niu et al., "A General Framework for the Analytical Inverse Kinematics
    Solution of Industrial Robots", Proc. IMechE Part C, 2025.
"""

import json
import copy
from math import pi, sin, cos
from typing import List

import numpy as np


class DHParam:
    """Modified Denavit-Hartenberg parameters for a single robot link.

    The MDH convention describes the transformation from frame {i-1} to frame {i}
    as: Rx(alpha) * Tx(a) * Rz(theta) * Tz(d)

    Attributes:
        alpha:        Link twist angle about X-axis (degrees).
        a:            Link length along X-axis (mm).
        theta_offset: Fixed offset added to the joint variable theta (degrees).
        d:            Link offset along Z-axis (mm).
        theta_min:    Minimum joint angle limit (degrees).
        theta_max:    Maximum joint angle limit (degrees).
        is_co_orient: Whether the Z-axis is co-oriented with the template frame.
                      Used after DH transformation to track axis direction.
        zero_offset:  Angular offset between the template and target robot's
                      zero position (degrees). Used for joint angle mapping.
    """

    def __init__(
        self,
        alpha: float = 0.0,
        a: float = 0.0,
        theta_offset: float = 0.0,
        d: float = 0.0,
        theta_min: float = -180.0,
        theta_max: float = 180.0,
    ):
        self.alpha = alpha
        self.a = a
        self.theta_offset = theta_offset
        self.d = d
        self.theta_min = theta_min
        self.theta_max = theta_max
        self.is_co_orient: bool = True
        self.zero_offset: float = 0.0

    def __repr__(self) -> str:
        return (
            f"DHParam(alpha={self.alpha}, a={self.a}, "
            f"theta_offset={self.theta_offset}, d={self.d}, "
            f"is_co_orient={self.is_co_orient}, zero_offset={self.zero_offset})"
        )

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "alpha": self.alpha,
            "a": self.a,
            "theta_offset": self.theta_offset,
            "d": self.d,
            "theta_min": self.theta_min,
            "theta_max": self.theta_max,
            "is_co_orient": self.is_co_orient,
            "zero_offset": self.zero_offset,
        }


def load_robot(json_path: str) -> List[DHParam]:
    """Load robot DH parameters from a JSON file.

    The JSON file should contain an array of objects, each with fields:
    ``i``, ``d``, ``a``, ``alpha``, ``thetaOffset``, ``thetaMin``, ``thetaMax``.

    Args:
        json_path: Path to the JSON file.

    Returns:
        A list of DHParam objects (typically 7 entries: 6 joints + end-effector).
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    dh_list = []
    for entry in data:
        dh_list.append(
            DHParam(
                alpha=entry["alpha"],
                a=entry["a"],
                theta_offset=entry["thetaOffset"],
                d=entry["d"],
                theta_min=entry.get("thetaMin", -180),
                theta_max=entry.get("thetaMax", 180),
            )
        )
    return dh_list


def forward_kinematics_mdh(
    alpha: float, a: float, theta_offset: float, d: float, theta: float
) -> np.ndarray:
    """Compute 4x4 homogeneous transformation for one MDH link.

    Implements: T = Rx(alpha) * Tx(a) * Rz(theta_offset + theta) * Tz(d)

    All angle inputs are in **degrees**.

    Args:
        alpha:        Link twist (degrees).
        a:            Link length (mm).
        theta_offset: Fixed joint offset (degrees).
        d:            Link offset (mm).
        theta:        Joint variable (degrees).

    Returns:
        4x4 numpy array (homogeneous transformation matrix).
    """
    al = alpha * pi / 180.0
    t = (theta_offset + theta) * pi / 180.0
    return np.array(
        [
            [cos(t), -sin(t), 0.0, a],
            [sin(t) * cos(al), cos(t) * cos(al), -sin(al), -d * sin(al)],
            [sin(t) * sin(al), cos(t) * sin(al), cos(al), d * cos(al)],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def forward_kinematics_chain(
    dh_list: List[DHParam], joint_angles_deg: List[float]
) -> np.ndarray:
    """Compute the full forward kinematics chain for a robot.

    Multiplies the individual link transforms to get the end-effector pose
    relative to the base frame.

    Args:
        dh_list:          List of DHParam (7 entries: 6 joints + end-effector).
        joint_angles_deg: Joint angles in degrees (6 values for a 6-DOF robot).

    Returns:
        4x4 homogeneous transformation matrix (base to end-effector).
    """
    T = np.eye(4, dtype=np.float64)
    for i in range(len(dh_list)):
        theta = joint_angles_deg[i] if i < len(joint_angles_deg) else 0.0
        dh = dh_list[i]
        T = T @ forward_kinematics_mdh(dh.alpha, dh.a, dh.theta_offset, dh.d, theta)
    return T


def normalize_angle(theta: float) -> float:
    """Normalize angle to the range [-pi, pi].

    Args:
        theta: Angle in radians.

    Returns:
        Normalized angle in radians.
    """
    # Use modular arithmetic for robustness
    theta = theta % (2.0 * pi)
    if theta > pi:
        theta -= 2.0 * pi
    elif theta < -pi:
        theta += 2.0 * pi
    return theta


def create_dh_list(n: int) -> List[DHParam]:
    """Create a list of n default DHParam objects.

    Useful for pre-allocating the transformed DH parameter list.

    Args:
        n: Number of DHParam entries to create.

    Returns:
        List of n default DHParam objects.
    """
    return [DHParam() for _ in range(n)]
