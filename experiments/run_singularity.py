"""Evaluate analytical IK behavior around the Yaskawa wrist singularity."""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from common import DEFAULT_SEED, PROCESSED_DIR, PROJECT_ROOT, forward_kinematics
except ImportError:  # Support module execution from the repository root.
    from experiments.common import (
        DEFAULT_SEED,
        PROCESSED_DIR,
        PROJECT_ROOT,
        forward_kinematics,
    )

from dh_transform_ik import CONFIGS, load_robot, solve_ik, transform_dh


TOLERANCE = 1e-3
SINGULAR_VALUES = (
    -1e-2,
    -1e-3,
    -1e-4,
    -1e-5,
    -1e-6,
    0.0,
    1e-6,
    1e-5,
    1e-4,
    1e-3,
    1e-2,
)
Q_FIELDS = [f"q{i}_rad" for i in range(1, 7)]
REFERENCE_DELTA_FIELDS = [f"reference_delta_q{i}_rad" for i in range(1, 7)]
PREVIOUS_DELTA_FIELDS = [f"previous_delta_q{i}_rad" for i in range(1, 7)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Yaskawa GP12 analytical IK wrist-singularity study."
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--trials-output",
        type=Path,
        default=PROCESSED_DIR / "singularity_trials.csv",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=PROCESSED_DIR / "singularity_summary.csv",
    )
    return parser.parse_args()


def wrapped_delta(q: np.ndarray, reference: np.ndarray) -> np.ndarray:
    difference = q - reference
    return np.arctan2(np.sin(difference), np.cos(difference))


def pose_metrics(target: np.ndarray, actual: np.ndarray):
    absolute = np.abs(target - actual)
    position_mm = float(np.linalg.norm(target[:3, 3] - actual[:3, 3]))
    relative_rotation = target[:3, :3].T @ actual[:3, :3]
    cosine = float(np.clip((np.trace(relative_rotation) - 1.0) / 2.0, -1.0, 1.0))
    return (
        float(np.mean(absolute)),
        float(np.max(absolute)),
        position_mm,
        math.degrees(math.acos(cosine)),
    )


def within_limits(q_rad: np.ndarray, robot_dh) -> bool:
    q_deg = np.degrees(q_rad)
    return all(
        robot_dh[i].theta_min - 1e-9 <= q_deg[i] <= robot_dh[i].theta_max + 1e-9
        for i in range(6)
    )


def reference_configuration(robot_dh, rng: np.random.Generator) -> np.ndarray:
    lower = np.array([item.theta_min for item in robot_dh[:6]], dtype=float)
    upper = np.array([item.theta_max for item in robot_dh[:6]], dtype=float)
    q_deg = rng.uniform(lower + 0.25 * (upper - lower), upper - 0.25 * (upper - lower))
    q_deg[4] = 0.0
    return np.radians(q_deg)


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    template = load_robot(str(PROJECT_ROOT / "robots" / "fanuc_lr_mate_200id.json"))
    robot_dh = load_robot(str(PROJECT_ROOT / "robots" / "yaskawa_gp12.json"))
    transformed = transform_dh(template, robot_dh)
    base_q = reference_configuration(robot_dh, rng)
    args.trials_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)

    trial_fields = [
        "q5_target_rad",
        "branch",
        *Q_FIELDS,
        "finite",
        "within_limits",
        "matrix_mean_error",
        "matrix_max_error",
        "position_error_mm",
        "rotation_error_deg",
        "passed_me_1e_3",
        *REFERENCE_DELTA_FIELDS,
        "reference_jump_rad",
        *PREVIOUS_DELTA_FIELDS,
        "previous_jump_rad",
    ]
    previous_by_branch = {}
    summaries = defaultdict(
        lambda: {"finite": 0, "limited": 0, "passed": 0, "jumps": [], "errors": []}
    )
    with args.trials_output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=trial_fields)
        writer.writeheader()
        for q5_value in SINGULAR_VALUES:
            reference_q = base_q.copy()
            reference_q[4] = q5_value
            target = np.asarray(
                forward_kinematics(robot_dh, np.degrees(reference_q)), dtype=float
            )
            solutions = solve_ik(
                target, template, robot_dh, transformed_dh=transformed
            )
            for branch in CONFIGS:
                q_rad = np.asarray(solutions.get(branch, [math.nan] * 6), dtype=float)
                finite = q_rad.shape == (6,) and bool(np.all(np.isfinite(q_rad)))
                limited = finite and within_limits(q_rad, robot_dh)
                metrics = (math.nan, math.nan, math.nan, math.nan)
                reference_delta = np.full(6, math.nan)
                previous_delta = np.full(6, math.nan)
                reference_jump = math.nan
                previous_jump = math.nan
                if finite:
                    actual = np.asarray(
                        forward_kinematics(robot_dh, np.degrees(q_rad)), dtype=float
                    )
                    metrics = pose_metrics(target, actual)
                    reference_delta = wrapped_delta(q_rad, reference_q)
                    reference_jump = float(np.linalg.norm(reference_delta))
                    if branch in previous_by_branch:
                        previous_delta = wrapped_delta(q_rad, previous_by_branch[branch])
                        previous_jump = float(np.linalg.norm(previous_delta))
                    previous_by_branch[branch] = q_rad
                passed = finite and metrics[0] < TOLERANCE
                item = summaries[q5_value]
                item["finite"] += int(finite)
                item["limited"] += int(limited)
                item["passed"] += int(passed)
                if passed:
                    item["errors"].append(metrics[0])
                if math.isfinite(previous_jump):
                    item["jumps"].append(previous_jump)

                output = {
                    "q5_target_rad": q5_value,
                    "branch": branch,
                    "finite": finite,
                    "within_limits": limited,
                    "matrix_mean_error": metrics[0],
                    "matrix_max_error": metrics[1],
                    "position_error_mm": metrics[2],
                    "rotation_error_deg": metrics[3],
                    "passed_me_1e_3": passed,
                    "reference_jump_rad": reference_jump,
                    "previous_jump_rad": previous_jump,
                }
                output.update({name: q_rad[i] for i, name in enumerate(Q_FIELDS)})
                output.update(
                    {
                        name: reference_delta[i]
                        for i, name in enumerate(REFERENCE_DELTA_FIELDS)
                    }
                )
                output.update(
                    {
                        name: previous_delta[i]
                        for i, name in enumerate(PREVIOUS_DELTA_FIELDS)
                    }
                )
                writer.writerow(output)

    summary_fields = [
        "q5_target_rad",
        "branches",
        "finite_branches",
        "within_limit_branches",
        "pose_passed_branches",
        "mean_passed_matrix_error",
        "max_passed_matrix_error",
        "mean_previous_jump_rad",
        "max_previous_jump_rad",
    ]
    with args.summary_output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary_fields)
        writer.writeheader()
        for q5_value in SINGULAR_VALUES:
            item = summaries[q5_value]
            writer.writerow(
                {
                    "q5_target_rad": q5_value,
                    "branches": len(CONFIGS),
                    "finite_branches": item["finite"],
                    "within_limit_branches": item["limited"],
                    "pose_passed_branches": item["passed"],
                    "mean_passed_matrix_error": float(np.mean(item["errors"]))
                    if item["errors"]
                    else math.nan,
                    "max_passed_matrix_error": max(item["errors"])
                    if item["errors"]
                    else math.nan,
                    "mean_previous_jump_rad": float(np.mean(item["jumps"]))
                    if item["jumps"]
                    else math.nan,
                    "max_previous_jump_rad": max(item["jumps"])
                    if item["jumps"]
                    else math.nan,
                }
            )


if __name__ == "__main__":
    main()
