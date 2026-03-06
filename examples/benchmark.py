"""
Benchmark: Reproduce the accuracy and speed results from the paper.

Tests 10,000 random FK -> IK -> FK roundtrips for each robot,
measuring success rate, accuracy, and computation time.

Results can be compared with Table 3 and Table 4 in:
    Niu et al., "A General Framework for the Analytical Inverse Kinematics
    Solution of Industrial Robots", Proc. IMechE Part C, 2025.
"""

import math
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dh_transform_ik import (
    load_robot,
    forward_kinematics_chain,
    forward_kinematics_mdh,
    solve_ik,
    transform_dh,
    CONFIGS,
)

robots_dir = os.path.join(os.path.dirname(__file__), "..", "robots")
template = load_robot(os.path.join(robots_dir, "fanuc_lr_mate_200id.json"))

ROBOTS = {
    "Fanuc LR Mate 200iD": "fanuc_lr_mate_200id.json",
    "Kuka KR50 R2500": "kuka_kr50_r2500.json",
    "ABB IRB2600-20/1.65": "abb_irb2600_20_165.json",
    "Yaskawa GP12": "yaskawa_gp12.json",
    "Estun ER8-720-MI": "estun_er8_720mi.json",
}

NUM_TRIALS = 10000
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


def benchmark_robot(robot_name, robot_file):
    """Run the full benchmark for one robot."""
    robot_dh = load_robot(os.path.join(robots_dir, robot_file))
    transformed = transform_dh(template, robot_dh)

    successes = 0
    total_error = 0.0
    total_time = 0.0

    for _ in range(NUM_TRIALS):
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
    avg_time_us = total_time / NUM_TRIALS * 1e6

    return successes, mean_error, avg_time_us


def main():
    random.seed(42)
    np.random.seed(42)

    print("=" * 78)
    print(f"  Benchmark: {NUM_TRIALS} FK -> IK -> FK roundtrips per robot")
    print("=" * 78)
    print()

    # Accuracy table (comparable to Table 3 in the paper)
    print("Accuracy Results:")
    print(f"{'Robot':<25} {'Success':>10} {'Rate':>8} {'Mean Error':>14}")
    print("-" * 60)

    results = {}
    for robot_name, robot_file in ROBOTS.items():
        successes, mean_error, avg_time = benchmark_robot(robot_name, robot_file)
        results[robot_name] = (successes, mean_error, avg_time)
        rate = successes / NUM_TRIALS * 100
        print(
            f"{robot_name:<25} {successes:>7}/{NUM_TRIALS}  {rate:6.1f}%  {mean_error:14.2e}"
        )

    # Speed table (comparable to Table 4 in the paper)
    print()
    print("Speed Results:")
    print(f"{'Robot':<25} {'Avg Time (us)':>15} {'Calls/sec':>12}")
    print("-" * 55)

    for robot_name, (_, _, avg_time) in results.items():
        calls_per_sec = 1e6 / avg_time
        print(f"{robot_name:<25} {avg_time:13.1f}   {calls_per_sec:10,.0f}")

    print()
    print("=" * 78)
    print("Note: The paper reports ~200x speedup over numerical methods.")
    print("Typical numerical IK (LM method) takes ~5-10 ms per call.")
    print(
        f"This analytical method: ~{sum(v[2] for v in results.values()) / len(results):.0f} us per call."
    )
    print("=" * 78)


if __name__ == "__main__":
    main()
