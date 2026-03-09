"""
Basic usage example for dh-transform-ik.

Demonstrates solving IK for a Kuka KR50 R2500 robot using the
Fanuc LR Mate 200iD as the template robot.
"""

import math
import os
import sys

import numpy as np

# Add parent directory to path (not needed if installed via `uv sync`)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dh_transform_ik import (
    load_robot,
    forward_kinematics_chain,
    solve_ik,
    transform_dh,
    CONFIGS,
)

# ---------------------------------------------------------------
# 1. Load robot DH parameters
# ---------------------------------------------------------------
robots_dir = os.path.join(os.path.dirname(__file__), "..", "robots")
template = load_robot(os.path.join(robots_dir, "fanuc_lr_mate_200id.json"))
robot = load_robot(os.path.join(robots_dir, "kuka_kr50_r2500.json"))

print("Template: Fanuc LR Mate 200iD")
print("Robot:    Kuka KR50 R2500")
print()

# ---------------------------------------------------------------
# 2. Compute FK for a known joint configuration
# ---------------------------------------------------------------
joint_angles_deg = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
target_pose = forward_kinematics_chain(robot, joint_angles_deg)

print(f"Input joint angles (deg): {joint_angles_deg}")
print(
    f"Target position (mm):     [{target_pose[0, 3]:.2f}, {target_pose[1, 3]:.2f}, {target_pose[2, 3]:.2f}]"
)
print()

# ---------------------------------------------------------------
# 3. Solve IK (one-call API)
# ---------------------------------------------------------------
solutions = solve_ik(target_pose, template, robot)

print("IK Solutions (8 configurations):")
print(
    f"{'Config':<6} {'theta1':>8} {'theta2':>8} {'theta3':>8} {'theta4':>8} {'theta5':>8} {'theta6':>8}  Valid"
)
print("-" * 72)

for name in CONFIGS:
    joints = solutions[name]
    if any(math.isnan(j) for j in joints):
        print(
            f"{name:<6}  {'NaN':>8} {'NaN':>8} {'NaN':>8} {'NaN':>8} {'NaN':>8} {'NaN':>8}  No"
        )
    else:
        degs = [math.degrees(j) for j in joints]
        print(
            f"{name:<6} {degs[0]:8.2f} {degs[1]:8.2f} {degs[2]:8.2f} {degs[3]:8.2f} {degs[4]:8.2f} {degs[5]:8.2f}  Yes"
        )

# ---------------------------------------------------------------
# 4. Verify: FK with IK solution should match original target
# ---------------------------------------------------------------
print()
print("Verification (FK roundtrip):")
for name in CONFIGS:
    joints = solutions[name]
    if any(math.isnan(j) for j in joints):
        continue
    q_deg = [math.degrees(j) for j in joints]
    T_check = forward_kinematics_chain(robot, q_deg)
    error = np.max(np.abs(target_pose - T_check))
    print(f"  {name}: max error = {error:.2e}")

# ---------------------------------------------------------------
# 5. For repeated calls, cache the DH transformation
# ---------------------------------------------------------------
print()
print("Cached DH transformation for repeated IK calls:")
transformed = transform_dh(template, robot)

import time

n_calls = 10000
start = time.perf_counter()
for _ in range(n_calls):
    _ = solve_ik(target_pose, template, robot, transformed_dh=transformed)
elapsed = time.perf_counter() - start

print(f"  {n_calls} IK calls in {elapsed:.3f}s ({elapsed / n_calls * 1e6:.1f} us/call)")
