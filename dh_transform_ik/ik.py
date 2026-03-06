"""
Analytical inverse kinematics solver for 6-DOF industrial robots.

This module provides a closed-form analytical IK solver that computes all 8
possible joint configurations for a 6-DOF robot with a spherical wrist.
The solver operates on the template robot's DH parameters, and the results
can be mapped to any robot with the same configuration using the DH
parameter transformation.

The 8 configurations correspond to all combinations of:
    - Front / Back      (shoulder: theta1)
    - Up / Down         (elbow: theta3)
    - No-flip / Flip    (wrist: theta4, theta5, theta6)

Reference:
    Niu et al., "A General Framework for the Analytical Inverse Kinematics
    Solution of Industrial Robots", Proc. IMechE Part C, 2025.
    See Section III: "Analytical Inverse Kinematics".
"""

from math import pi, sin, cos, atan2, acos, sqrt, pow as fpow
from typing import Dict, List, Optional, Tuple
import math

import numpy as np

from .dh import DHParam, forward_kinematics_mdh, normalize_angle
from .transform import transform_dh, transform_joints_all


# The 8 IK configuration names
CONFIGS = ["FUN", "FUF", "FDN", "FDF", "BUN", "BUF", "BDN", "BDF"]
"""
Configuration naming convention:
    F/B = Front/Back (shoulder configuration, theta1 differs by pi)
    U/D = Up/Down (elbow configuration, two theta3 solutions)
    N/F = No-flip/Flip (wrist configuration, theta4/5/6 alternatives)
"""


def _no_flip_to_flip(
    theta4: float, theta5: float, theta6: float
) -> Tuple[float, float, float]:
    """Convert a no-flip wrist solution to its flip equivalent.

    The wrist singularity gives two equivalent orientations:
    the "no-flip" and "flip" solutions are related by adding/subtracting
    pi to theta4 and theta6, and negating theta5.

    Args:
        theta4, theta5, theta6: No-flip wrist angles (radians).

    Returns:
        (theta4, theta5, theta6) for the flip configuration.
    """
    if theta4 > 0:
        t4 = theta4 - pi
    else:
        t4 = theta4 + pi
    t5 = -theta5
    if theta6 > 0:
        t6 = theta6 - pi
    else:
        t6 = theta6 + pi
    t6 = normalize_angle(t6)
    return (t4, t5, t6)


def inverse_kinematics(
    target_pose: np.ndarray,
    dh_list: List[DHParam],
) -> Dict[str, List[float]]:
    """Analytical inverse kinematics for 6-DOF robots with spherical wrist.

    Computes all 8 possible joint configurations using geometric and algebraic
    methods. The solver uses the following approach:
        - Theta1: From base plane geometry (atan2)
        - Theta3: From the cosine law (elbow angle)
        - Theta2: From planar arm equations
        - Theta4, 5, 6: From rotation matrix decomposition

    Singularity handling:
        - Wrist singularity (theta5 ~ 0): theta4 is set to 0, theta6 is
          computed from the remaining rotation matrix.
        - Unreachable positions: NaN values are returned for those configs.

    Args:
        target_pose: 4x4 homogeneous transformation matrix representing the
                     desired end-effector pose. Position in mm, rotation as 3x3.
        dh_list:     Transformed DH parameters (8 entries from ``transform_dh``).
                     Indices [1..6] are used for the 6 joints.

    Returns:
        Dictionary mapping configuration names to joint angle lists (radians).
        Keys: 'FUN', 'FUF', 'FDN', 'FDF', 'BUN', 'BUF', 'BDN', 'BDF'.
        NaN values indicate unreachable configurations.

    Example:
        >>> solutions = inverse_kinematics(target_pose, transformed_dh)
        >>> for name, joints in solutions.items():
        ...     if not any(math.isnan(j) for j in joints):
        ...         print(f"{name}: {[round(math.degrees(j), 2) for j in joints]}")
    """
    # Extract link parameters from the transformed DH list.
    # The transformed list has 8 entries: [0]=base, [1..6]=joints, [7]=end.
    # The IK equations use the link lengths and offsets from joints 1-4.
    # In the original code, the base entry was popped before passing to IK,
    # so dhList[0] was joint 1, dhList[1] was joint 2, etc.
    # Here we keep the base entry, so indices are shifted by +1.
    a1 = dh_list[2].a  # link 1 length (from joint 2 frame = J1-to-J2)
    a2 = dh_list[3].a  # link 2 length (from joint 3 frame = J2-to-J3)
    a3 = dh_list[4].a  # link 3 length (from joint 4 frame = J3-to-J4)
    d1 = dh_list[1].d  # joint 1 d offset
    d4 = dh_list[4].d  # joint 4 d offset (wrist offset)

    A1, A2, A3, D1, D4 = abs(a1), abs(a2), abs(a3), abs(d1), abs(d4)

    # Extract target pose components
    R = target_pose[:3, :3]
    n_x, n_y, n_z = R[0, 0], R[1, 0], R[2, 0]
    o_x, o_y, o_z = R[0, 1], R[1, 1], R[2, 1]
    a_x, a_y, a_z = R[0, 2], R[1, 2], R[2, 2]
    p_x, p_y, p_z = target_pose[0, 3], target_pose[1, 3], target_pose[2, 3]

    theta1 = [0.0] * 8
    theta2 = [0.0] * 8
    theta3 = [0.0] * 8
    theta4 = [0.0] * 8
    theta5 = [0.0] * 8
    theta6 = [0.0] * 8

    # ------------------------------------------------------------------
    # Theta1: Base rotation (front/back)
    # Front: theta1 = atan2(py, px)
    # Back:  theta1 = atan2(py, px) +/- pi
    # ------------------------------------------------------------------
    t1_front = atan2(p_y, p_x)
    if t1_front > 0:
        t1_back = t1_front - pi
    else:
        t1_back = t1_front + pi

    for i in range(4):
        theta1[i] = t1_front  # Front: indices 0-3
    for i in range(4, 8):
        theta1[i] = t1_back  # Back: indices 4-7

    # ------------------------------------------------------------------
    # Theta3: Elbow angle via cosine law
    # Two solutions: Up (indices 0,1,4,5) and Down (indices 2,3,6,7)
    # ------------------------------------------------------------------
    # Distance from J1 to wrist center (Front)
    l52 = sqrt(fpow(sqrt(fpow(p_x, 2) + fpow(p_y, 2)) - A1, 2) + fpow(p_z - D1, 2))
    # Distance from J1 to wrist center (Back)
    l52_B = sqrt(fpow(sqrt(fpow(p_x, 2) + fpow(p_y, 2)) + A1, 2) + fpow(p_z - D1, 2))
    # Distance from J3 to J5 (wrist center)
    l43 = sqrt(fpow(A3, 2) + fpow(D4, 2))

    # Front configurations (indices 0-3)
    cos_ang3 = (fpow(l43, 2) + fpow(A2, 2) - fpow(l52, 2)) / (2.0 * l43 * A2)
    if l52 <= (A2 + l43) and abs(cos_ang3) <= 1.0:
        t3_up = -pi + (acos(cos_ang3) + atan2(D4, A3))
        t3_down = pi - (acos(cos_ang3) - atan2(D4, A3))
    else:
        t3_up = float("nan")
        t3_down = float("nan")
    theta3[0] = theta3[1] = t3_up  # Front-Up
    theta3[2] = theta3[3] = t3_down  # Front-Down

    # Back configurations (indices 4-7)
    cos_ang3_B = (fpow(l43, 2) + fpow(A2, 2) - fpow(l52_B, 2)) / (2.0 * l43 * A2)
    if l52_B <= (A2 + l43) and abs(cos_ang3_B) <= 1.0:
        t3_up_B = -pi + (acos(cos_ang3_B) + atan2(D4, A3))
        t3_down_B = pi - (acos(cos_ang3_B) - atan2(D4, A3))
    else:
        t3_up_B = float("nan")
        t3_down_B = float("nan")
    theta3[4] = theta3[5] = t3_up_B  # Back-Up
    theta3[6] = theta3[7] = t3_down_B  # Back-Down

    for i in range(8):
        theta3[i] = normalize_angle(theta3[i])

    # ------------------------------------------------------------------
    # Theta2: Shoulder angle from position equations
    # ------------------------------------------------------------------
    for idx in range(8):
        if math.isnan(theta3[idx]):
            theta2[idx] = float("nan")
            continue
        mcos = a2 + a3 * cos(theta3[idx]) - d4 * sin(theta3[idx])
        nsin = -a3 * sin(theta3[idx]) - d4 * cos(theta3[idx])
        var2_z = p_z - d1

        if abs(sin(theta1[idx])) > 1e-10:
            var2_xy = p_y / sin(theta1[idx]) - a1
        else:
            var2_xy = p_x / cos(theta1[idx]) - a1

        denom = fpow(mcos, 2) + fpow(nsin, 2)
        cos_theta2 = (nsin * var2_xy + mcos * var2_z) / denom

        if abs(nsin) > 1e-10:
            sin_theta2 = (mcos * cos_theta2 - var2_z) / nsin
        else:
            sin_theta2 = (var2_xy - nsin * cos_theta2) / mcos

        theta2[idx] = atan2(sin_theta2, cos_theta2)

    # ------------------------------------------------------------------
    # Theta4, 5, 6: Wrist angles from rotation matrix decomposition
    # ------------------------------------------------------------------
    for idx in range(8):
        if math.isnan(theta2[idx]) or math.isnan(theta3[idx]):
            theta4[idx] = float("nan")
            theta5[idx] = float("nan")
            theta6[idx] = float("nan")
            continue

        s1, c1 = sin(theta1[idx]), cos(theta1[idx])
        s2, c2 = sin(theta2[idx]), cos(theta2[idx])
        s3, c3 = sin(theta3[idx]), cos(theta3[idx])
        s23 = sin(theta2[idx] + theta3[idx])
        c23 = cos(theta2[idx] + theta3[idx])

        # Elements from the rotation matrix equation
        cos_theta6 = (
            n_z * (c3 * s2 - c2 * s3)
            - n_x * c1 * (c2 * c3 + s2 * s3)
            - n_y * s1 * (c2 * c3 + s2 * s3)
        )
        sin_theta6 = (
            o_z * (c3 * s2 - c2 * s3)
            - o_x * c1 * (c2 * c3 + s2 * s3)
            - o_y * s1 * (c2 * c3 + s2 * s3)
        )
        cos_theta5 = (
            a_z * (c3 * s2 - c2 * s3)
            - a_x * c1 * (c2 * c3 + s2 * s3)
            - a_y * s1 * (c2 * c3 + s2 * s3)
        )
        sin_theta5 = -sqrt(fpow(cos_theta6, 2) + fpow(sin_theta6, 2))

        theta5[idx] = atan2(sin_theta5, cos_theta5)

        cos_theta4 = (
            a_z * (c2 * c3 + s2 * s3)
            - a_x * c1 * (c3 * s2 - c2 * s3)
            + a_x * c1 * c3 * s2
            - a_x * c1 * c2 * s3
            + a_y * s1 * c3 * s2
            - a_y * s1 * c2 * s3
        )
        # Simplify: cos_theta4 = a_z*c23 + (a_x*c1 + a_y*s1)*s23
        # but keep original form for fidelity with the derivation
        cos_theta4 = (
            a_z * c23
            - a_x * c1 * (-s23)
            + a_x * c1 * s23
            - a_y * s1 * (-s23)
            + a_y * s1 * s23
        )
        # Use the correct simplified form from the original code:
        cos_theta4 = (
            a_z * (c2 * c3 + s2 * s3)
            - a_x * c1 * (c2 * s3 - c3 * s2)
            - a_y * s1 * (c2 * s3 - c3 * s2)
        )
        sin_theta4 = a_x * s1 - a_y * c1

        # Wrist singularity check: when theta5 ~ 0, theta4 and theta6
        # are coupled (gimbal lock). Set theta4 = 0 and compute theta6
        # from the remaining rotation.
        if abs(sin(theta5[idx])) <= 1e-5:
            theta4[idx] = 0.0
            # Compute FK chain from joint 1 to joint 5 to extract theta6.
            # Use dh_list[1..5] (joints 1-5), NO base transform prepended
            # because target_pose is already in the template space with
            # base removed.
            T05 = np.eye(4, dtype=np.float64)
            joint_vals = [
                theta1[idx],
                theta2[idx],
                theta3[idx],
                theta4[idx],
                theta5[idx],
            ]
            for j in range(5):
                dh = dh_list[j + 1]  # dh_list[1..5]
                t_deg = math.degrees(joint_vals[j])
                T_j = forward_kinematics_mdh(
                    dh.alpha, dh.a, dh.theta_offset, dh.d, t_deg
                )
                T05 = T05 @ T_j
            # A56 = T05^-1 * target_pose
            T05_inv = np.linalg.inv(T05)
            A56 = T05_inv @ target_pose
            cos6 = A56[0, 0]
            sin6 = -A56[2, 0]
            theta6[idx] = atan2(sin6, cos6)
        else:
            theta4[idx] = atan2(
                sin_theta4 / sin(theta5[idx]),
                -cos_theta4 / sin(theta5[idx]),
            )
            theta6[idx] = atan2(
                -sin_theta6 / sin(theta5[idx]),
                cos_theta6 / sin(theta5[idx]),
            )

    # Apply flip transformation for odd indices (F configurations)
    for idx in range(8):
        if math.isnan(theta4[idx]):
            continue
        if idx % 2 == 1:  # Flip configurations: FUF, FDF, BUF, BDF
            theta4[idx], theta5[idx], theta6[idx] = _no_flip_to_flip(
                theta4[idx], theta5[idx], theta6[idx]
            )

    # Normalize all wrist angles
    for idx in range(8):
        if not math.isnan(theta4[idx]):
            theta4[idx] = normalize_angle(theta4[idx])
            theta5[idx] = normalize_angle(theta5[idx])
            theta6[idx] = normalize_angle(theta6[idx])

    # Build result dictionary
    result = {}
    for i, name in enumerate(CONFIGS):
        result[name] = [
            theta1[i],
            theta2[i],
            theta3[i],
            theta4[i],
            theta5[i],
            theta6[i],
        ]
    return result


def solve_ik(
    target_pose: np.ndarray,
    template_dh: List[DHParam],
    robot_dh: List[DHParam],
    transformed_dh: Optional[List[DHParam]] = None,
) -> Dict[str, List[float]]:
    """High-level API: Solve inverse kinematics for any robot in one call.

    This function combines all three steps of the general IK framework:
        1. Transform the robot's DH parameters to the template structure
        2. Transform the target pose to the template's base frame
        3. Solve analytical IK and map joints back to the robot's space

    The DH transformation (step 1) can be cached by passing ``transformed_dh``
    to avoid recomputation on repeated calls.

    Args:
        target_pose:    4x4 homogeneous transformation matrix. Position in mm.
        template_dh:    Template robot DH parameters (7 entries).
        robot_dh:       Target robot DH parameters (7 entries).
        transformed_dh: Pre-computed transformed DH (8 entries). If None,
                        it will be computed from template_dh and robot_dh.

    Returns:
        Dictionary mapping configuration names to joint angle lists (radians)
        in the **target robot's** joint space.

    Example:
        >>> from dh_transform_ik import load_robot, solve_ik
        >>> template = load_robot("robots/fanuc_lr_mate_200id.json")
        >>> robot = load_robot("robots/kuka_kr50_r2500.json")
        >>> target = np.eye(4); target[0, 3] = 1000; target[2, 3] = 500
        >>> solutions = solve_ik(target, template, robot)
    """
    # Step 1: Transform DH parameters (can be cached)
    if transformed_dh is None:
        transformed_dh = transform_dh(template_dh, robot_dh)

    # Step 2: Compute base and end-effector frame transforms
    T_base = forward_kinematics_mdh(
        transformed_dh[0].alpha,
        transformed_dh[0].a,
        transformed_dh[0].theta_offset,
        transformed_dh[0].d,
        0.0,
    )
    T_end = forward_kinematics_mdh(
        transformed_dh[7].alpha,
        transformed_dh[7].a,
        transformed_dh[7].theta_offset,
        transformed_dh[7].d,
        0.0,
    )

    T_base_inv = np.linalg.inv(T_base)
    T_end_inv = np.linalg.inv(T_end)

    # Transform the target pose into the template's coordinate frame
    target_in_template = T_base_inv @ target_pose @ T_end_inv

    # Step 3: Solve IK in template space
    std_solutions = inverse_kinematics(target_in_template, transformed_dh)

    # Step 4: Map joint angles back to the robot's joint space
    std_list = [std_solutions[name] for name in CONFIGS]
    arb_list = transform_joints_all(std_list, transformed_dh)

    result = {}
    for i, name in enumerate(CONFIGS):
        result[name] = arb_list[i]
    return result
