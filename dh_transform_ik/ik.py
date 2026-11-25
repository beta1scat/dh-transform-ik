"""Analytical inverse kinematics for the Fanuc-template 6R configuration.

This module provides a closed-form analytical IK solver that enumerates 8
theoretical joint-configuration slots for a 6-DOF robot with a spherical wrist;
unreachable branches are represented by NaN values.
The solver operates on the Fanuc-template spherical-wrist geometry, and its
results can be mapped to robots with the same mechanical configuration whose
D-H representations satisfy the framework's D-H-level isomorphism criteria.
It is not a solver for arbitrary 6R configurations. Other families require
their own analytical equations, branch conventions, and dispatch mechanism.

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

    The "no-flip" and "flip" solutions are related by adding/subtracting
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


# Only numerically degenerate wrists have a free angle. Nearby nonsingular
# targets retain their unique wrist decomposition instead of being approximated.
WRIST_SINGULAR_TOLERANCE = 1e-10
SHOULDER_AXIS_TOLERANCE_MM = 1e-9


def _wrist_theta6(target_pose, dh_list, first_five):
    """Recover the last joint from a specified first-five-joint transform."""
    transform = np.eye(4, dtype=float)
    for index, angle in enumerate(first_five):
        dh = dh_list[index + 1]
        transform = transform @ forward_kinematics_mdh(
            dh.alpha, dh.a, dh.theta_offset, dh.d, math.degrees(angle)
        )
    relative_rotation = transform[:3, :3].T @ target_pose[:3, :3]
    return atan2(-relative_rotation[2, 0], relative_rotation[0, 0])


def _closest_coupled_wrist(q4, q6, slope, lower, upper, reference):
    """Project a reference onto q6 = slope*q4 + offset (mod 2*pi).

    All feasible turns of the coupling relation are considered. The objective
    is actual joint displacement, so an extra full turn is not treated as free.
    """
    offset = q6 - slope * q4
    possible = (offset + slope * lower[0], offset + slope * upper[0])
    first = math.ceil((lower[1] - max(possible)) / (2 * pi) - 1e-12)
    last = math.floor((upper[1] - min(possible)) / (2 * pi) + 1e-12)
    candidates = []
    for turn in range(first, last + 1):
        intercept = offset + 2 * pi * turn
        ends = sorted((slope * (lower[1] - intercept),
                       slope * (upper[1] - intercept)))
        left, right = max(lower[0], ends[0]), min(upper[0], ends[1])
        if left > right:
            continue
        optimum = (reference[0] + slope * (reference[1] - intercept)) / 2.0
        selected4 = min(max(optimum, left), right)
        selected6 = slope * selected4 + intercept
        cost = (selected4 - reference[0]) ** 2 + (selected6 - reference[1]) ** 2
        candidates.append((cost, selected4, selected6))
    if not candidates:
        return None
    _, selected4, selected6 = min(candidates)
    return selected4, selected6


def inverse_kinematics(
    target_pose: np.ndarray,
    dh_list: List[DHParam],
    reference_q: Optional[List[float]] = None,
) -> Dict[str, List[float]]:
    """Analytical IK for the Fanuc-template 6R spherical-wrist family.

    Enumerates 8 theoretical joint configurations using geometric and algebraic
    methods. Unreachable branches are represented by NaN values. The solver
    uses the following approach:
        - Theta1: From base plane geometry (atan2)
        - Theta3: From the cosine law (elbow angle)
        - Theta2: From planar arm equations
        - Theta4, 5, 6: From rotation matrix decomposition

    Singularity handling:
        - Numerically singular wrist: theta4 uses the reference (or zero),
          and theta6 is recovered from the remaining rotation. solve_ik
          subsequently resolves the coupled pair within actual joint limits.
        - Unreachable positions: NaN values are returned for those configs.

    Args:
        target_pose: 4x4 homogeneous transformation matrix representing the
                     desired end-effector pose. Position in mm, rotation as 3x3.
        dh_list:     Transformed DH parameters (8 entries from ``transform_dh``).
                     Indices [1..6] are used for the 6 joints.
        reference_q: Optional six-joint reference configuration in the
                     transformed template space, in radians. It is used only
                     to resolve the free angle at shoulder or wrist
                     singularities; ordinary branches remain fully enumerated.

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
    if reference_q is not None:
        reference_q = np.asarray(reference_q, dtype=float)
        if reference_q.shape != (6,) or not np.all(np.isfinite(reference_q)):
            raise ValueError("reference_q must contain six finite joint angles in radians")

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
    if reference_q is not None and math.hypot(p_x, p_y) <= SHOULDER_AXIS_TOLERANCE_MM:
        t1_front = normalize_angle(float(reference_q[0]))
    else:
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
            - a_x * c1 * (c2 * s3 - c3 * s2)
            - a_y * s1 * (c2 * s3 - c3 * s2)
        )
        sin_theta4 = a_x * s1 - a_y * c1

        # At numerical degeneracy choose a representative free angle.
        # solve_ik subsequently resolves the pair using actual joint limits.
        if abs(sin(theta5[idx])) <= WRIST_SINGULAR_TOLERANCE:
            theta4[idx] = (
                normalize_angle(float(reference_q[3]))
                if reference_q is not None
                else 0.0
            )
            theta6[idx] = _wrist_theta6(
                target_pose, dh_list,
                [theta1[idx], theta2[idx], theta3[idx], theta4[idx], theta5[idx]],
            )
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
    reference_q: Optional[List[float]] = None,
) -> Dict[str, List[float]]:
    """Solve IK for a robot compatible with the Fanuc-template configuration.

    This function combines the three framework steps for the one analytical
    solver currently implemented in this package:
        1. Transform the robot's DH parameters to the template structure
        2. Transform the target pose to the template's base frame
        3. Solve analytical IK and map joints back to the robot's space

    The DH transformation (step 1) can be cached by passing ``transformed_dh``
    to avoid recomputation on repeated calls.

    Args:
        target_pose:    4x4 homogeneous transformation matrix. Position in mm.
        template_dh:    Fanuc-configuration template D-H parameters (7 entries).
        robot_dh:       Target robot DH parameters (7 entries).
        transformed_dh: Pre-computed transformed DH (8 entries). If None,
                        it will be computed from template_dh and robot_dh.
        reference_q: Optional six-joint reference configuration in the target
                     robot's joint space, in radians. It is internally mapped
                     to template space and used only for singularity-aware
                     candidate generation. At a singular wrist the q4/q6 pair
                     is selected jointly within actual limits by minimizing
                     squared joint displacement from this reference.

    Returns:
        Dictionary mapping configuration names to joint angle lists (radians)
        in the **target robot's** joint space.

    Raises:
        UnsupportedDHTransformationError: If ``transformed_dh`` is not supplied
            and the template/target D-H pair is outside the transformation
            method's applicable conditions.

    Note:
        ``template_dh`` does not select an IK implementation. This function
        always calls :func:`inverse_kinematics`, whose equations are specific
        to the Fanuc-template 6R spherical-wrist family. ``transform_dh``
        evaluates the D-H-level isomorphism criteria after joint types and
        serial topology are known to match; it does not register or select a
        solver for a newly introduced configuration family.

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
    reference_q_template = None
    if reference_q is not None:
        reference_q = np.asarray(reference_q, dtype=float)
        if reference_q.shape != (6,) or not np.all(np.isfinite(reference_q)):
            raise ValueError("reference_q must contain six finite joint angles in radians")
        reference_q_template = []
        for index, value in enumerate(reference_q):
            dh = transformed_dh[index + 1]
            mapped = -value if not dh.is_co_orient else value
            reference_q_template.append(
                mapped - math.radians(dh.zero_offset)
            )

    std_solutions = inverse_kinematics(
        target_in_template, transformed_dh, reference_q=reference_q_template
    )

    # Step 4: Map joint angles back to the robot's joint space
    std_list = [std_solutions[name] for name in CONFIGS]
    arb_list = transform_joints_all(std_list, transformed_dh)

    # A singular wrist is a one-parameter family, not just q4 = reference[3].
    # Resolve q4 and q6 together after mapping, in actual actuator coordinates.
    lower = np.radians([robot_dh[i].theta_min for i in (3, 5)])
    upper = np.radians([robot_dh[i].theta_max for i in (3, 5)])
    for index, name in enumerate(CONFIGS):
        standard = std_solutions[name]
        if not np.all(np.isfinite(standard)):
            continue
        if abs(sin(standard[4])) > WRIST_SINGULAR_TOLERANCE:
            continue
        probe = list(standard[:5])
        probe[3] += pi / 2
        probe6 = _wrist_theta6(target_in_template, transformed_dh, probe)
        template_slope = 1.0 if normalize_angle(probe6 - standard[5]) > 0 else -1.0
        direction4 = 1.0 if transformed_dh[4].is_co_orient else -1.0
        direction6 = 1.0 if transformed_dh[6].is_co_orient else -1.0
        slope = template_slope * direction4 * direction6
        current = arb_list[index]
        reference = (reference_q[3], reference_q[5]) if reference_q is not None else (current[3], current[5])
        pair = _closest_coupled_wrist(
            current[3], current[5], slope, lower, upper, reference
        )
        if pair is None:
            arb_list[index] = [float("nan")] * 6
        else:
            current[3], current[5] = pair

    return {name: arb_list[index] for index, name in enumerate(CONFIGS)}
