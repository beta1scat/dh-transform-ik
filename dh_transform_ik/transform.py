"""D-H representation transformation within one configuration family.

This module implements Algorithm 1 from the paper: transforming compatible
Modified D-H parameters into the template's coordinate frame structure using
algebraic commutation rules of homogeneous transformation matrices.

The transformation enables an analytical IK solver written for a configuration
template to be applied to robots in the same configuration family. Joint types,
joint count, and serial topology are assumed to match; this module evaluates
the remaining D-H-level isomorphism criteria from accumulated twists and axial
offset propagation. It does not verify that a matching analytical solver has
been implemented for a newly introduced template family.

Reference:
    Niu et al., "A General Framework for the Analytical Inverse Kinematics
    Solution of Industrial Robots", Proc. IMechE Part C, 2025.
    See Section III: "Transform D-H parameters to D-H parameters template"
    and Algorithm 1.
"""

from math import pi
from typing import List, Optional, Tuple

import numpy as np

from .dh import DHParam, create_dh_list


def _discrete_twist_state(angle: float) -> Optional[float]:
    """Return the canonical 0/180-degree state, or None if unsupported."""
    normalized = angle % 360.0
    if np.isclose(normalized, 0.0, rtol=0.0, atol=1e-9) or np.isclose(
        normalized, 360.0, rtol=0.0, atol=1e-9
    ):
        return 0.0
    if np.isclose(normalized, 180.0, rtol=0.0, atol=1e-9):
        return 180.0
    return None


class UnsupportedDHTransformationError(ValueError):
    """Raised when a D-H pair fails the framework's isomorphism criteria."""


def transform_dh(
    template_dh: List[DHParam],
    arbitrary_dh: List[DHParam],
) -> List[DHParam]:
    """Transform a target D-H representation to a template structure.

    This is the core contribution of the paper. Given a template robot's DH
    parameters and a target robot's compatible DH parameters with the same
    configuration, this function computes a new set of DH parameters that maps
    the target robot's kinematics into the template's coordinate frame.

    The transformation uses the following commutation rules of basic
    homogeneous transformation units (Equations 7-11 in the paper):

        1. Rx(*) * Tx(*) = Tx(*) * Rx(*)    (same-axis rotation/translation)
        2. Rz(*) * Tz(*) = Tz(*) * Rz(*)    (same-axis rotation/translation)
        3. Tz(*) * Tx(*) = Tx(*) * Tz(*)    (translations on different axes)
        4. Rx(pi) * Rz(*) = Rz(-*) * Rx(pi) (rotation commutation)
        5. Rz(pi) * Rx(*) = Rx(-*) * Rz(pi) (rotation commutation)
        6. Rx(pi) * Tz(*) = Tz(-*) * Rx(pi) (rotation/translation commutation)
        7. Rz(pi) * Tx(*) = Tx(-*) * Rz(pi) (rotation/translation commutation)

    Args:
        template_dh:  DH parameters of the template robot (7 entries).
        arbitrary_dh: DH parameters of the target robot (7 entries).

    Returns:
        Transformed DH parameter list (8 entries):
          [0] = base frame transformation
          [1..6] = 6 joint frames (matching template's structure)
          [7] = end-effector frame
        Each entry contains ``is_co_orient`` and ``zero_offset`` fields that
        record the joint angle mapping relationship.

    Raises:
        UnsupportedDHTransformationError: If the template and target fail the
            framework's D-H-level isomorphism criteria: accumulated twists must
            be 0 or 180 degrees, and axial offsets must have a supported
            propagation path. Joint types and serial topology must already be
            known to match before calling this function.
    """
    std = template_dh
    old = arbitrary_dh
    new = create_dh_list(8)

    al_offset = [0.0] * 7
    d_offset = [0.0] * 7
    inverse_d = [1.0] * 7

    # ----------------------------------------------------------------
    # Part 1: Extract base frame transformation (Eq. 15 in the paper)
    #
    # Compare the first DH entry (Base-to-J1) of both robots to determine
    # the rigid transformation between their base coordinate frames.
    # ----------------------------------------------------------------

    # Alpha offset: difference in X-rotation between base frames
    base_twist_state = _discrete_twist_state(old[0].alpha - std[0].alpha)
    if base_twist_state is None:
        raise UnsupportedDHTransformationError(
            "Template and target fail the isomorphism criterion at the base "
            "frame: accumulated alpha offset must be 0 or 180 degrees."
        )
    al_offset[0] = base_twist_state
    new[0].alpha = 0
    new[1].alpha = std[0].alpha

    # Link length offset
    new[0].a = old[0].a - std[0].a
    new[1].a = std[0].a

    # Theta offset
    new[0].theta_offset = 0
    new[1].theta_offset = std[0].theta_offset
    if al_offset[0] == 180:
        # Base X-axes are anti-parallel: joint rotation direction is flipped
        # Rule: Rx(pi) * Rz(theta) = Rz(-theta) * Rx(pi)
        new[1].is_co_orient = False
        inverse_d[0] = -1
    else:
        new[1].is_co_orient = True
    new[1].zero_offset = (
        std[0].theta_offset - inverse_d[0] * old[0].theta_offset
    )

    # Handle d offset between base frames
    # When one robot has d=0 and the other has d!=0, the difference
    # must be absorbed into the base transformation
    d0 = np.array([old[0].d, std[0].d])
    if not np.all(d0 == 0) and not np.all(d0 != 0):
        new[1].d = std[0].d
        template_twist_state = _discrete_twist_state(new[1].alpha)
        if template_twist_state == 180:
            # Rule: Rx(pi) * Tz(d) = Tz(-d) * Rx(pi)
            new[0].d = inverse_d[0] * (std[0].d - old[0].d)
        elif template_twist_state == 0:
            new[0].d = inverse_d[0] * (old[0].d - std[0].d)
        else:
            raise UnsupportedDHTransformationError(
                "Template and target fail the isomorphism criterion at the "
                "base frame: a nonzero d-offset requires template alpha to "
                "be 0 or 180 degrees."
            )
    else:
        new[0].d = 0
        new[1].d = inverse_d[0] * old[0].d

    # ----------------------------------------------------------------
    # Part 2: Extract joint-to-joint transformations (Eq. 16-17 in paper)
    #
    # For each joint i (1 through 5), compare the DH parameters and
    # determine the coordinate mapping, tracking accumulated alpha
    # offsets and d offsets through the chain.
    # ----------------------------------------------------------------
    for idx in range(1, 6):
        # Alpha: accumulated rotation offset determines axis co-orientation
        diff_al = _discrete_twist_state(old[idx].alpha - std[idx].alpha)
        accumulated_state = (
            None
            if diff_al is None
            else _discrete_twist_state(diff_al + al_offset[idx - 1])
        )
        if accumulated_state is None:
            raise UnsupportedDHTransformationError(
                f"Template and target fail the isomorphism criterion at joint "
                f"{idx}: accumulated alpha offset must be 0 or 180 degrees."
            )
        al_offset[idx] = accumulated_state
        new[idx + 1].alpha = std[idx].alpha
        if al_offset[idx] == 180:
            inverse_d[idx] = -1

        # Link length: preserved from the arbitrary robot
        new[idx + 1].a = old[idx].a

        # Theta offset and joint direction
        new[idx + 1].theta_offset = std[idx].theta_offset
        if (diff_al == 180 and new[idx].is_co_orient) or (
            diff_al == 0 and not new[idx].is_co_orient
        ):
            # Joint rotation is reversed relative to template
            # Rule: Rx(pi) * Rz(theta) = Rz(-theta) * Rx(pi)
            new[idx + 1].is_co_orient = False
            new[idx + 1].zero_offset = std[idx].theta_offset + old[idx].theta_offset
        else:
            # Joint rotation matches template direction
            new[idx + 1].is_co_orient = True
            new[idx + 1].zero_offset = std[idx].theta_offset - old[idx].theta_offset

        # Handle d parameter: when one is zero and the other is not,
        # propagate the offset to the next link
        d_tmp = np.array([old[idx].d, std[idx].d])
        if not np.all(d_tmp == 0) and not np.all(d_tmp != 0):
            if d_offset[idx - 1] != 0:
                raise UnsupportedDHTransformationError(
                    f"Template and target fail the isomorphism criterion at "
                    f"joint {idx}: consecutive d-offset propagation is not "
                    "supported."
                )
            new[idx + 1].d = std[idx].d
            next_twist_state = _discrete_twist_state(std[idx + 1].alpha)
            if next_twist_state == 180:
                d_offset[idx] = std[idx].d - old[idx].d
            elif next_twist_state == 0:
                d_offset[idx] = old[idx].d - std[idx].d
            else:
                raise UnsupportedDHTransformationError(
                    f"Template and target fail the isomorphism criterion at "
                    f"joint {idx}: a nonzero d-offset requires the next "
                    "template alpha to be 0 or 180 degrees."
                )
        else:
            new[idx + 1].d = inverse_d[idx] * (old[idx].d + d_offset[idx - 1])

    # ----------------------------------------------------------------
    # Part 3: Extract end-effector frame transformation
    #
    # The last link (J6-to-End) carries forward the accumulated alpha
    # offset and preserves the arbitrary robot's end-effector parameters.
    # ----------------------------------------------------------------
    new[7].d = old[6].d + d_offset[5]
    new[7].theta_offset = old[6].theta_offset
    new[7].a = old[6].a
    new[7].alpha = (old[6].alpha + al_offset[5]) % 360

    return new


def transform_joints_all(
    std_solutions: List[List[float]],
    transformed_dh: List[DHParam],
) -> List[List[float]]:
    """Map the 8 theoretical IK configuration slots to the target robot space.

    After solving IK in the template's coordinate frame, the resulting joint
    angles must be mapped back to the arbitrary robot's joint conventions.
    This mapping uses the ``is_co_orient`` and ``zero_offset`` fields computed
    during DH transformation.

    For each joint i:
        - If co-oriented:  q_arb = q_std + zero_offset
        - If anti-oriented: q_arb = -(q_std + zero_offset)

    Args:
        std_solutions:  Tuple of 8 solutions, each a list of 6 joint angles
                        in radians (output of ``inverse_kinematics``).
        transformed_dh: Transformed DH parameter list (8 entries, output of
                        ``transform_dh``). Indices [1..6] are used.

    Returns:
        List of 8 solutions, each a list of 6 joint angles in radians,
        in the arbitrary robot's joint space.
    """
    arb_solutions = []
    for config in std_solutions:
        arb_q = [0.0] * len(config)
        for i in range(len(config)):
            dh = transformed_dh[i + 1]  # indices 1..6 for joints
            arb_q[i] = config[i] + dh.zero_offset * pi / 180.0
            if not dh.is_co_orient:
                arb_q[i] = -arb_q[i]
        arb_solutions.append(arb_q)
    return arb_solutions


def transform_joints_single(
    std_solution: List[float],
    transformed_dh: List[DHParam],
) -> List[float]:
    """Convert a single IK solution from template space to arbitrary robot space.

    Args:
        std_solution:   A single solution: list of 6 joint angles in radians.
        transformed_dh: Transformed DH parameter list (8 entries).

    Returns:
        List of 6 joint angles in radians, in the arbitrary robot's joint space.
    """
    arb_q = [0.0] * len(std_solution)
    for i in range(len(std_solution)):
        dh = transformed_dh[i + 1]
        arb_q[i] = std_solution[i] + dh.zero_offset * pi / 180.0
        if not dh.is_co_orient:
            arb_q[i] = -arb_q[i]
    return arb_q
