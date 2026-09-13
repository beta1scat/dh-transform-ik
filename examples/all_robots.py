"""Demonstrate IK solving for all five included same-configuration models.

Shows the FK -> IK -> FK roundtrip for each robot with a random joint
configuration, verifying that the Fanuc-template analytical IK produces
machine-precision roundtrip solutions for these spherical-wrist models.
"""

import math
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # not needed with uv

from dh_transform_ik import (
    load_robot,
    forward_kinematics_chain,
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

random.seed(42)


def random_joints(dh_list):
    """Generate random joint angles within limits."""
    angles = []
    for i in range(6):
        lo, hi = dh_list[i].theta_min, dh_list[i].theta_max
        margin = (hi - lo) * 0.05
        angles.append(random.uniform(lo + margin, hi - margin))
    return angles


print("=" * 70)
print("  dh-transform-ik: All Robots Demo")
print("=" * 70)

for robot_name, robot_file in ROBOTS.items():
    robot_dh = load_robot(os.path.join(robots_dir, robot_file))
    transformed = transform_dh(template, robot_dh)

    # Generate a random reachable pose
    q_deg = random_joints(robot_dh)
    target = forward_kinematics_chain(robot_dh, q_deg)

    # Solve IK
    solutions = solve_ik(target, template, robot_dh, transformed_dh=transformed)

    print(f"\n--- {robot_name} ---")
    print(f"Input joints (deg): [{', '.join(f'{a:.1f}' for a in q_deg)}]")
    print(
        f"Target pos (mm):    [{target[0, 3]:.1f}, {target[1, 3]:.1f}, {target[2, 3]:.1f}]"
    )

    valid = 0
    for name in CONFIGS:
        joints = solutions[name]
        if not any(math.isnan(j) for j in joints):
            valid += 1
            q_ik_deg = [math.degrees(j) for j in joints]
            T_check = forward_kinematics_chain(robot_dh, q_ik_deg)
            err = np.max(np.abs(target - T_check))
            print(
                f"  {name}: [{', '.join(f'{d:8.2f}' for d in q_ik_deg)}]  err={err:.1e}"
            )

    print(f"  Valid configs: {valid}/8")

print("\n" + "=" * 70)
print("All robots solved successfully.")
print("=" * 70)
