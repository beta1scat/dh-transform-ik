"""
Benchmark: Reproduce the accuracy and speed results from the paper.

Tests FK -> IK -> FK roundtrips for each robot, measuring success rate,
accuracy, and computation time for BOTH the analytical method (this package)
and the numerical Levenberg-Marquardt method (roboticstoolbox).

Results can be compared with Table 3 and Table 4 in:
    Niu et al., "A General Framework for the Analytical Inverse Kinematics
    Solution of Industrial Robots", Proc. IMechE Part C, 2025.

Requirements:
    uv run --group benchmark python examples/benchmark.py
"""

import math
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # not needed with uv

from dh_transform_ik import (
    load_robot,
    forward_kinematics_chain,
    forward_kinematics_mdh,
    solve_ik,
    transform_dh,
    CONFIGS,
)

try:
    import roboticstoolbox as rtb
    from spatialmath import SE3

    HAS_RTB = True
except ImportError:
    HAS_RTB = False

robots_dir = os.path.join(os.path.dirname(__file__), "..", "robots")
template = load_robot(os.path.join(robots_dir, "fanuc_lr_mate_200id.json"))

ROBOTS = {
    "Fanuc LR Mate 200iD": "fanuc_lr_mate_200id.json",
    "Kuka KR50 R2500": "kuka_kr50_r2500.json",
    "ABB IRB2600-20/1.65": "abb_irb2600_20_165.json",
    "Yaskawa GP12": "yaskawa_gp12.json",
    "Estun ER8-720-MI": "estun_er8_720mi.json",
}

NUM_TRIALS_ANALYTICAL = 10000
NUM_TRIALS_NUMERICAL = 200  # Numerical methods are slow; 200 trials is enough
TOLERANCE = 1e-3  # mm-level tolerance for the 4x4 matrix comparison


def compute_fk(dh_list, joint_angles_deg):
    """Compute FK for original robot (7 DH entries)."""
    T = np.eye(4, dtype=np.float64)
    for i in range(len(dh_list)):
        dh = dh_list[i]
        theta = joint_angles_deg[i] if i < len(joint_angles_deg) else 0.0
        T = T @ forward_kinematics_mdh(dh.alpha, dh.a, dh.theta_offset, dh.d, theta)
    return T


def random_joints(dh_list):
    """Generate random joint angles within limits."""
    angles = []
    for i in range(6):
        lo, hi = dh_list[i].theta_min, dh_list[i].theta_max
        margin = (hi - lo) * 0.05
        angles.append(random.uniform(lo + margin, hi - margin))
    return angles


def build_rtb_robot(dh_list):
    """Build a roboticstoolbox DHRobot from our DH parameter list.

    Args:
        dh_list: List of DHParam objects (7 entries: 6 joints + end-effector).

    Returns:
        rtb.robot.DHRobot instance.
    """
    links = []
    for i in range(6):
        dh = dh_list[i]
        links.append(
            rtb.RevoluteMDH(
                alpha=math.radians(dh.alpha),
                a=dh.a / 1000.0,  # mm -> m
                offset=math.radians(dh.theta_offset),
                d=dh.d / 1000.0,  # mm -> m
            )
        )
    # End-effector (7th entry) as tool transform
    ee = dh_list[6]
    al_r = math.radians(ee.alpha)
    t_r = math.radians(ee.theta_offset)
    tool_T = np.array(
        [
            [math.cos(t_r), -math.sin(t_r), 0.0, ee.a / 1000.0],
            [
                math.sin(t_r) * math.cos(al_r),
                math.cos(t_r) * math.cos(al_r),
                -math.sin(al_r),
                -ee.d / 1000.0 * math.sin(al_r),
            ],
            [
                math.sin(t_r) * math.sin(al_r),
                math.cos(t_r) * math.sin(al_r),
                math.cos(al_r),
                ee.d / 1000.0 * math.cos(al_r),
            ],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    robot = rtb.DHRobot(links, tool=SE3(tool_T), name="benchmark_robot")
    return robot


def benchmark_analytical(robot_name, robot_file):
    """Run the analytical IK benchmark for one robot."""
    robot_dh = load_robot(os.path.join(robots_dir, robot_file))
    transformed = transform_dh(template, robot_dh)

    successes = 0
    total_error = 0.0
    total_time = 0.0

    for _ in range(NUM_TRIALS_ANALYTICAL):
        q_deg = random_joints(robot_dh)
        target = compute_fk(robot_dh, q_deg)

        t0 = time.perf_counter()
        solutions = solve_ik(target, template, robot_dh, transformed_dh=transformed)
        t1 = time.perf_counter()
        total_time += t1 - t0

        # Check if at least one solution reconstructs the pose
        for name in CONFIGS:
            joints = solutions[name]
            if any(math.isnan(j) for j in joints):
                continue
            q_ik_deg = [math.degrees(j) for j in joints]
            T_check = compute_fk(robot_dh, q_ik_deg)
            error = np.mean(np.abs(target - T_check))
            if error < TOLERANCE:
                successes += 1
                total_error += error
                break

    mean_error = total_error / max(successes, 1)
    avg_time_us = total_time / NUM_TRIALS_ANALYTICAL * 1e6

    return successes, NUM_TRIALS_ANALYTICAL, mean_error, avg_time_us


def benchmark_numerical(robot_name, robot_file):
    """Run the numerical IK (Levenberg-Marquardt) benchmark for one robot.

    Args:
        robot_name: Display name of the robot.
        robot_file: JSON filename in robots/ directory.

    Returns:
        (successes, total_trials, mean_error, avg_time_us)
    """
    robot_dh = load_robot(os.path.join(robots_dir, robot_file))
    rtb_robot = build_rtb_robot(robot_dh)

    successes = 0
    total_error = 0.0
    total_time = 0.0

    for _ in range(NUM_TRIALS_NUMERICAL):
        # Generate random joint angles (radians) within limits
        q_rad = np.array(
            [
                math.radians(
                    random.uniform(
                        robot_dh[i].theta_min
                        + (robot_dh[i].theta_max - robot_dh[i].theta_min) * 0.05,
                        robot_dh[i].theta_max
                        - (robot_dh[i].theta_max - robot_dh[i].theta_min) * 0.05,
                    )
                )
                for i in range(6)
            ]
        )

        # FK using roboticstoolbox (returns SE3, positions in meters)
        ee_se3 = rtb_robot.fkine(q_rad)

        # Numerical IK (Levenberg-Marquardt)
        t0 = time.perf_counter()
        sol = rtb_robot.ikine_LM(ee_se3)
        t1 = time.perf_counter()
        total_time += t1 - t0

        # Check solution
        if sol.success:
            fk_check = rtb_robot.fkine(sol.q)
            error_matrix = fk_check.A - ee_se3.A
            error = np.mean(np.abs(error_matrix))
            if error < TOLERANCE:
                successes += 1
                total_error += error

    mean_error = total_error / max(successes, 1)
    avg_time_us = total_time / NUM_TRIALS_NUMERICAL * 1e6

    return successes, NUM_TRIALS_NUMERICAL, mean_error, avg_time_us


def main():
    random.seed(42)
    np.random.seed(42)

    print("=" * 80)
    print(
        f"  Analytical IK Benchmark: "
        f"{NUM_TRIALS_ANALYTICAL} FK -> IK -> FK roundtrips per robot"
    )
    print("=" * 80)
    print()

    # ------------------------------------------------------------------
    # Analytical method benchmark
    # ------------------------------------------------------------------
    print("Analytical Method Results:")
    print(
        f"{'Robot':<25} {'Success':>12} {'Rate':>8} {'Mean Error':>14} {'Avg Time':>12}"
    )
    print("-" * 75)

    analytical_results = {}
    for robot_name, robot_file in ROBOTS.items():
        successes, trials, mean_error, avg_time = benchmark_analytical(
            robot_name, robot_file
        )
        analytical_results[robot_name] = (successes, trials, mean_error, avg_time)
        rate = successes / trials * 100
        print(
            f"{robot_name:<25} {successes:>7}/{trials}  {rate:6.1f}%  "
            f"{mean_error:14.2e}  {avg_time:9.1f} us"
        )

    # ------------------------------------------------------------------
    # Numerical method benchmark (Levenberg-Marquardt)
    # ------------------------------------------------------------------
    if not HAS_RTB:
        print()
        print("=" * 80)
        print("  Numerical IK benchmark SKIPPED (roboticstoolbox not installed)")
        print("  Install with: uv sync --group benchmark")
        print("=" * 80)
    else:
        print()
        print("=" * 80)
        print(
            f"  Numerical IK Benchmark (Levenberg-Marquardt): "
            f"{NUM_TRIALS_NUMERICAL} trials per robot"
        )
        print("=" * 80)
        print()
        print(
            f"{'Robot':<25} {'Success':>12} {'Rate':>8} "
            f"{'Mean Error':>14} {'Avg Time':>12}"
        )
        print("-" * 75)

        numerical_results = {}
        for robot_name, robot_file in ROBOTS.items():
            successes, trials, mean_error, avg_time = benchmark_numerical(
                robot_name, robot_file
            )
            numerical_results[robot_name] = (successes, trials, mean_error, avg_time)
            rate = successes / trials * 100
            print(
                f"{robot_name:<25} {successes:>5}/{trials}  {rate:6.1f}%  "
                f"{mean_error:14.2e}  {avg_time:9.1f} us"
            )

        # Print comparison summary
        print()
        print("--- Comparison: Analytical vs Numerical (LM) ---")
        print(f"{'Robot':<25} {'Analytical':>12} {'Numerical':>12} {'Speedup':>10}")
        print("-" * 62)
        for robot_name in ROBOTS:
            a_time = analytical_results[robot_name][3]
            n_time = numerical_results[robot_name][3]
            speedup = n_time / a_time if a_time > 0 else float("inf")
            print(
                f"{robot_name:<25} {a_time:9.1f} us  {n_time:9.1f} us  {speedup:8.0f}x"
            )

    # ------------------------------------------------------------------
    # Overall summary
    # ------------------------------------------------------------------
    print()
    print("=" * 80)
    print("  Summary")
    print("=" * 80)
    avg_analytical = sum(v[3] for v in analytical_results.values()) / len(
        analytical_results
    )
    print(f"Analytical method average: {avg_analytical:.1f} us/call")

    if HAS_RTB and numerical_results:  # type: ignore[possibly-undefined]
        avg_numerical = sum(v[3] for v in numerical_results.values()) / len(
            numerical_results
        )
        overall_speedup = avg_numerical / avg_analytical
        print(
            f"Numerical (LM) average:   {avg_numerical:.1f} us/call  "
            f"(speedup: {overall_speedup:.0f}x)"
        )
    else:
        print("Install roboticstoolbox to see numerical method comparison:")
        print("  uv sync --group benchmark")

    print("=" * 80)


if __name__ == "__main__":
    main()
