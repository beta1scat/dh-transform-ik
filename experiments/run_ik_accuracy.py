"""Evaluate all analytical IK branches against fixed FK targets."""

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


ROBOT_FILES = {
    "abb_irb2600_20_165": "abb_irb2600_20_165.json",
    "estun_er8_720mi": "estun_er8_720mi.json",
    "kuka_kr50_r2500": "kuka_kr50_r2500.json",
    "yaskawa_gp12": "yaskawa_gp12.json",
}
Q_FIELDS = [f"q{i}_rad" for i in range(1, 7)]
Q_DEG_FIELDS = [f"q{i}_deg" for i in range(1, 7)]
T_FIELDS = [f"target_{row}{col}" for row in range(4) for col in range(4)]
TOLERANCE = 1e-3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the analytical IK accuracy study.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--input", type=Path, default=PROCESSED_DIR / "ik_targets.csv")
    parser.add_argument(
        "--trials-output",
        type=Path,
        default=PROCESSED_DIR / "ik_analytical_trials.csv",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=PROCESSED_DIR / "ik_analytical_summary.csv",
    )
    return parser.parse_args()


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
        robot_dh[index].theta_min - 1e-9
        <= q_deg[index]
        <= robot_dh[index].theta_max + 1e-9
        for index in range(6)
    )


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    template = load_robot(str(PROJECT_ROOT / "robots" / "fanuc_lr_mate_200id.json"))
    contexts = {}
    stats = defaultdict(
        lambda: {
            "trials": 0,
            "pose_success": 0,
            "finite": 0,
            "within_limits": 0,
            "valid_limit": 0,
            "passed_errors": [],
            "best_matrix_errors": [],
            "best_position_errors": [],
            "best_rotation_errors": [],
        }
    )
    trial_fields = [
        "trial",
        "robot",
        "branch",
        *Q_FIELDS,
        *Q_DEG_FIELDS,
        "finite",
        "within_limits",
        "matrix_mean_error",
        "matrix_max_error",
        "position_error_mm",
        "rotation_error_deg",
        "passed_me_1e_3",
        "valid_limit_solution",
    ]
    args.trials_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)

    with args.input.open("r", newline="", encoding="utf-8") as source, args.trials_output.open(
        "w", newline="", encoding="utf-8"
    ) as destination:
        reader = csv.DictReader(source)
        missing = {"trial", "robot", *T_FIELDS} - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Target CSV is missing fields: {sorted(missing)}")
        writer = csv.DictWriter(destination, fieldnames=trial_fields)
        writer.writeheader()

        for target_row in reader:
            robot_name = target_row["robot"]
            if robot_name not in ROBOT_FILES:
                raise ValueError(f"Unknown robot in target CSV: {robot_name}")
            if robot_name not in contexts:
                robot_dh = load_robot(
                    str(PROJECT_ROOT / "robots" / ROBOT_FILES[robot_name])
                )
                contexts[robot_name] = (
                    robot_dh,
                    transform_dh(template, robot_dh),
                )
            robot_dh, transformed = contexts[robot_name]
            target = np.array([float(target_row[name]) for name in T_FIELDS]).reshape(4, 4)
            solutions = solve_ik(target, template, robot_dh, transformed_dh=transformed)
            trial_passed = False
            passed_metrics = []
            stats[robot_name]["trials"] += 1

            for branch in CONFIGS:
                q_rad = np.asarray(solutions.get(branch, [math.nan] * 6), dtype=float)
                finite = q_rad.shape == (6,) and bool(np.all(np.isfinite(q_rad)))
                limited = finite and within_limits(q_rad, robot_dh)
                metrics = (math.nan, math.nan, math.nan, math.nan)
                if finite:
                    actual = np.asarray(
                        forward_kinematics(robot_dh, np.degrees(q_rad)), dtype=float
                    )
                    metrics = pose_metrics(target, actual)
                passed = finite and metrics[0] < TOLERANCE
                valid_limit = passed and limited
                trial_passed = trial_passed or passed
                stats[robot_name]["finite"] += int(finite)
                stats[robot_name]["within_limits"] += int(limited)
                stats[robot_name]["valid_limit"] += int(valid_limit)
                if passed:
                    stats[robot_name]["passed_errors"].append(metrics[0])
                    passed_metrics.append(metrics)

                output = {
                    "trial": target_row["trial"],
                    "robot": robot_name,
                    "branch": branch,
                    "finite": finite,
                    "within_limits": limited,
                    "matrix_mean_error": metrics[0],
                    "matrix_max_error": metrics[1],
                    "position_error_mm": metrics[2],
                    "rotation_error_deg": metrics[3],
                    "passed_me_1e_3": passed,
                    "valid_limit_solution": valid_limit,
                }
                output.update({name: q_rad[i] for i, name in enumerate(Q_FIELDS)})
                output.update(
                    {name: math.degrees(q_rad[i]) for i, name in enumerate(Q_DEG_FIELDS)}
                )
                writer.writerow(output)
            stats[robot_name]["pose_success"] += int(trial_passed)
            if passed_metrics:
                best = min(passed_metrics, key=lambda item: item[0])
                stats[robot_name]["best_matrix_errors"].append(best[0])
                stats[robot_name]["best_position_errors"].append(best[2])
                stats[robot_name]["best_rotation_errors"].append(best[3])

    summary_fields = [
        "robot",
        "trials",
        "pose_success_trials",
        "pose_success_rate",
        "finite_branches",
        "within_limit_branches",
        "valid_limit_solutions",
        "mean_passed_matrix_error",
        "max_passed_matrix_error",
        "mean_best_matrix_error",
        "max_best_matrix_error",
        "mean_best_position_error_mm",
        "max_best_position_error_mm",
        "mean_best_rotation_error_deg",
        "max_best_rotation_error_deg",
    ]
    with args.summary_output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary_fields)
        writer.writeheader()
        for robot_name in ROBOT_FILES:
            item = stats[robot_name]
            errors = item["passed_errors"]
            best_errors = item["best_matrix_errors"]
            position_errors = item["best_position_errors"]
            rotation_errors = item["best_rotation_errors"]
            writer.writerow(
                {
                    "robot": robot_name,
                    "trials": item["trials"],
                    "pose_success_trials": item["pose_success"],
                    "pose_success_rate": item["pose_success"] / item["trials"]
                    if item["trials"]
                    else math.nan,
                    "finite_branches": item["finite"],
                    "within_limit_branches": item["within_limits"],
                    "valid_limit_solutions": item["valid_limit"],
                    "mean_passed_matrix_error": float(np.mean(errors))
                    if errors
                    else math.nan,
                    "max_passed_matrix_error": max(errors) if errors else math.nan,
                    "mean_best_matrix_error": float(np.mean(best_errors))
                    if best_errors
                    else math.nan,
                    "max_best_matrix_error": max(best_errors)
                    if best_errors
                    else math.nan,
                    "mean_best_position_error_mm": float(np.mean(position_errors))
                    if position_errors
                    else math.nan,
                    "max_best_position_error_mm": max(position_errors)
                    if position_errors
                    else math.nan,
                    "mean_best_rotation_error_deg": float(np.mean(rotation_errors))
                    if rotation_errors
                    else math.nan,
                    "max_best_rotation_error_deg": max(rotation_errors)
                    if rotation_errors
                    else math.nan,
                }
            )


if __name__ == "__main__":
    main()
