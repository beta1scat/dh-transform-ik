"""
Tests for the dh-transform-ik package.

Validates the full FK -> IK -> FK roundtrip for all supported robots.
Each test generates random joint configurations, computes FK to get end-effector
poses, solves IK, then verifies the solutions reproduce the original poses.
"""

import math
import os
import sys
import random

import numpy as np

# Add parent directory to path so we can import the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dh_transform_ik import (
    load_robot,
    forward_kinematics_chain,
    forward_kinematics_mdh,
    transform_dh,
    solve_ik,
    CONFIGS,
)

# Path to robot data files
ROBOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "robots")
TEMPLATE_PATH = os.path.join(ROBOTS_DIR, "fanuc_lr_mate_200id.json")

# Robots to test (all except Fanuc template)
TEST_ROBOTS = {
    "Kuka KR50 R2500": "kuka_kr50_r2500.json",
    "ABB IRB2600": "abb_irb2600_20_165.json",
    "Yaskawa GP12": "yaskawa_gp12.json",
    "Estun ER8-720-MI": "estun_er8_720mi.json",
}

# Error tolerance for FK -> IK -> FK roundtrip
TOLERANCE = 1e-3


def _random_joint_angles(dh_list, num_joints=6):
    """Generate random joint angles within joint limits.

    Args:
        dh_list: Robot DH parameters (original, 7 entries).
        num_joints: Number of joints (default 6).

    Returns:
        List of joint angles in degrees.
    """
    angles = []
    for i in range(num_joints):
        lo = dh_list[i].theta_min
        hi = dh_list[i].theta_max
        # Shrink range slightly to avoid edge cases
        margin = (hi - lo) * 0.05
        angles.append(random.uniform(lo + margin, hi - margin))
    return angles


def _compute_fk(dh_list, joint_angles_deg):
    """Compute forward kinematics for the original robot.

    Args:
        dh_list: Original robot DH parameters (7 entries).
        joint_angles_deg: Joint angles in degrees (6 values).

    Returns:
        4x4 homogeneous transformation matrix (end-effector pose in mm).
    """
    T = np.eye(4, dtype=np.float64)
    for i in range(len(dh_list)):
        dh = dh_list[i]
        theta = joint_angles_deg[i] if i < len(joint_angles_deg) else 0.0
        T = T @ forward_kinematics_mdh(dh.alpha, dh.a, dh.theta_offset, dh.d, theta)
    return T


def _pose_error(T1, T2):
    """Compute mean absolute error between two homogeneous transforms.

    Args:
        T1, T2: 4x4 homogeneous transformation matrices.

    Returns:
        Mean absolute error across all matrix elements.
    """
    return np.mean(np.abs(T1 - T2))


def test_fk_roundtrip(robot_name, robot_file, num_trials=100, verbose=True):
    """Test FK -> IK -> FK roundtrip for a specific robot.

    Args:
        robot_name: Human-readable robot name.
        robot_file: JSON filename in robots/ directory.
        num_trials: Number of random configurations to test.
        verbose: Print results.

    Returns:
        (num_successes, num_failures, mean_error)
    """
    template_dh = load_robot(TEMPLATE_PATH)
    robot_dh = load_robot(os.path.join(ROBOTS_DIR, robot_file))

    # Pre-compute the DH transformation (done once)
    transformed_dh = transform_dh(template_dh, robot_dh)

    successes = 0
    failures = 0
    total_error = 0.0

    for trial in range(num_trials):
        # Generate random joint config and compute FK
        q_deg = _random_joint_angles(robot_dh)
        T_target = _compute_fk(robot_dh, q_deg)

        # Solve IK
        solutions = solve_ik(
            T_target, template_dh, robot_dh, transformed_dh=transformed_dh
        )

        # Check each solution
        trial_ok = False
        best_error = float("inf")

        for config_name, q_rad in solutions.items():
            if any(math.isnan(j) for j in q_rad):
                continue

            # Convert to degrees for FK
            q_ik_deg = [math.degrees(j) for j in q_rad]

            # Compute FK with IK solution
            T_computed = _compute_fk(robot_dh, q_ik_deg)

            # Compute error
            error = _pose_error(T_target, T_computed)
            best_error = min(best_error, error)

            if error < TOLERANCE:
                trial_ok = True
                total_error += error
                break

        if trial_ok:
            successes += 1
        else:
            failures += 1
            if verbose and failures <= 3:
                print(f"  Trial {trial}: FAILED (best_error={best_error:.6e})")
                print(f"    Input joints: {[round(a, 2) for a in q_deg]}")

    mean_error = total_error / max(successes, 1)

    if verbose:
        status = "PASS" if failures == 0 else "FAIL"
        print(
            f"[{status}] {robot_name}: "
            f"{successes}/{num_trials} passed, "
            f"{failures} failures, "
            f"mean_error={mean_error:.2e}"
        )

    return successes, failures, mean_error


def test_fanuc_self_ik(num_trials=100, verbose=True):
    """Test that the template robot (Fanuc) can solve its own IK.

    This is the baseline test - if this fails, the core IK solver has issues.
    """
    template_dh = load_robot(TEMPLATE_PATH)

    # For the template, the transformed DH is identity-like
    transformed_dh = transform_dh(template_dh, template_dh)

    successes = 0
    failures = 0
    total_error = 0.0

    for trial in range(num_trials):
        q_deg = _random_joint_angles(template_dh)
        T_target = _compute_fk(template_dh, q_deg)

        solutions = solve_ik(
            T_target, template_dh, template_dh, transformed_dh=transformed_dh
        )

        trial_ok = False
        for config_name, q_rad in solutions.items():
            if any(math.isnan(j) for j in q_rad):
                continue
            q_ik_deg = [math.degrees(j) for j in q_rad]
            T_computed = _compute_fk(template_dh, q_ik_deg)
            error = _pose_error(T_target, T_computed)
            if error < TOLERANCE:
                trial_ok = True
                total_error += error
                break

        if trial_ok:
            successes += 1
        else:
            failures += 1

    mean_error = total_error / max(successes, 1)

    if verbose:
        status = "PASS" if failures == 0 else "FAIL"
        print(
            f"[{status}] Fanuc LR Mate 200iD (template): "
            f"{successes}/{num_trials} passed, "
            f"{failures} failures, "
            f"mean_error={mean_error:.2e}"
        )

    return successes, failures, mean_error


def test_dh_transform_preserves_fk():
    """Test that DH transformation preserves FK for mapped joint angles.

    For any arbitrary joint configuration on the original robot, the
    corresponding mapped configuration on the transformed robot must
    produce the same end-effector pose.
    """
    template_dh = load_robot(TEMPLATE_PATH)

    print("\n--- DH Transform FK Preservation Test ---")
    all_pass = True
    for robot_name, robot_file in TEST_ROBOTS.items():
        robot_dh = load_robot(os.path.join(ROBOTS_DIR, robot_file))
        transformed_dh = transform_dh(template_dh, robot_dh)

        # Use a known joint configuration
        q_arb_deg = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]

        # Compute FK at this config for original robot
        T_orig = _compute_fk(robot_dh, q_arb_deg)

        # Map arbitrary joint angles to standard (template) space
        # Invert: arb_q = std_q + zo*pi/180; if !co: arb_q = -arb_q
        # => if co: std_q = arb_q - zo*pi/180
        #    if !co: -arb_q = std_q + zo*pi/180 => std_q = -arb_q - zo*pi/180
        q_std_deg = []
        for i in range(6):
            dh = transformed_dh[i + 1]
            arb_rad = q_arb_deg[i] * math.pi / 180.0
            if dh.is_co_orient:
                std_rad = arb_rad - dh.zero_offset * math.pi / 180.0
            else:
                std_rad = -arb_rad - dh.zero_offset * math.pi / 180.0
            q_std_deg.append(math.degrees(std_rad))

        # Compute FK on transformed robot: base * joints * end
        T_trans = np.eye(4, dtype=np.float64)
        T_trans = T_trans @ forward_kinematics_mdh(
            transformed_dh[0].alpha,
            transformed_dh[0].a,
            transformed_dh[0].theta_offset,
            transformed_dh[0].d,
            0.0,
        )
        for i in range(6):
            dh = transformed_dh[i + 1]
            T_trans = T_trans @ forward_kinematics_mdh(
                dh.alpha, dh.a, dh.theta_offset, dh.d, q_std_deg[i]
            )
        T_trans = T_trans @ forward_kinematics_mdh(
            transformed_dh[7].alpha,
            transformed_dh[7].a,
            transformed_dh[7].theta_offset,
            transformed_dh[7].d,
            0.0,
        )

        error = _pose_error(T_orig, T_trans)
        status = "PASS" if error < 1e-10 else "FAIL"
        if error >= 1e-10:
            all_pass = False
        print(f"  [{status}] {robot_name}: mapped FK error = {error:.2e}")

    return all_pass


def main():
    """Run all tests."""
    random.seed(42)  # Reproducible results
    np.random.seed(42)

    print("=" * 60)
    print("dh-transform-ik Test Suite")
    print("=" * 60)

    # Test 1: DH Transform preserves FK
    test_dh_transform_preserves_fk()

    # Test 2: Template self-IK
    print("\n--- FK -> IK -> FK Roundtrip Tests (100 random configs each) ---")
    test_fanuc_self_ik(num_trials=100)

    # Test 3: All robots
    all_pass = True
    for robot_name, robot_file in TEST_ROBOTS.items():
        _, failures, _ = test_fk_roundtrip(robot_name, robot_file, num_trials=100)
        if failures > 0:
            all_pass = False

    # Summary
    print("\n" + "=" * 60)
    if all_pass:
        print("All tests PASSED.")
    else:
        print("Some tests FAILED. See details above.")
    print("=" * 60)


if __name__ == "__main__":
    main()
