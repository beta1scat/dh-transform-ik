"""Run the three Paper 1 numerical IK methods in the pinned legacy environment."""

from __future__ import annotations

import argparse
import csv
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3

try:
    from common import PROCESSED_DIR, PROJECT_ROOT, pose_errors
except ImportError:
    from experiments.common import PROCESSED_DIR, PROJECT_ROOT, pose_errors

from dh_transform_ik import forward_kinematics_mdh, load_robot


ROBOT_FILES = {
    "abb_irb2600_20_165": "abb_irb2600_20_165.json",
    "estun_er8_720mi": "estun_er8_720mi.json",
    "kuka_kr50_r2500": "kuka_kr50_r2500.json",
    "yaskawa_gp12": "yaskawa_gp12.json",
}
METHODS = ("ikine_min", "ikine_LM", "ikine_LMS")
T_FIELDS = [f"target_{row}{column}" for row in range(4) for column in range(4)]
Q_FIELDS = [f"q{i}_rad" for i in range(1, 7)]
ME_TOLERANCE = 1e-3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=10_000, help="Targets per robot")
    parser.add_argument("--input", type=Path, default=PROCESSED_DIR / "ik_targets.csv")
    parser.add_argument(
        "--trials-output",
        type=Path,
        default=PROCESSED_DIR / "ik_numerical_legacy_trials.csv",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=PROCESSED_DIR / "ik_numerical_legacy_summary.csv",
    )
    args = parser.parse_args()
    if args.trials <= 0:
        parser.error("--trials must be positive")
    return args


def build_rtb_robot(dh_list):
    links = []
    for dh in dh_list[:6]:
        links.append(
            rtb.RevoluteMDH(
                alpha=math.radians(dh.alpha),
                a=dh.a / 1000.0,
                offset=math.radians(dh.theta_offset),
                d=dh.d / 1000.0,
            )
        )
    end = dh_list[6]
    tool_mm = forward_kinematics_mdh(
        end.alpha, end.a, end.theta_offset, end.d, 0.0
    )
    tool_m = tool_mm.copy()
    tool_m[:3, 3] /= 1000.0
    return rtb.DHRobot(links, tool=SE3(tool_m), name="paper1_legacy_model")


def target_to_metres(target_mm: np.ndarray) -> np.ndarray:
    target_m = target_mm.copy()
    target_m[:3, 3] /= 1000.0
    return target_m


def solve(robot, method: str, target: SE3):
    return getattr(robot, method)(target)


def solution_data(solution):
    q = np.asarray(getattr(solution, "q", [math.nan] * 6), dtype=float)
    success = bool(getattr(solution, "success", np.all(np.isfinite(q))))
    iterations = getattr(solution, "iterations", "")
    reason = getattr(solution, "reason", "")
    residual = getattr(solution, "residual", math.nan)
    return q, success, iterations, reason, residual


def main() -> None:
    args = parse_args()
    targets = defaultdict(list)
    with args.input.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            robot = row["robot"]
            if robot in ROBOT_FILES and len(targets[robot]) < args.trials:
                targets[robot].append(
                    (
                        int(row["trial"]),
                        np.array([float(row[field]) for field in T_FIELDS]).reshape(4, 4),
                    )
                )

    fields = [
        "robot",
        "trial",
        "method",
        *Q_FIELDS,
        "solver_success",
        "experiment_success",
        "iterations",
        "reason",
        "residual",
        "legacy_matrix_mean_error_m",
        "matrix_mean_error",
        "matrix_max_error",
        "position_error_mm",
        "rotation_error_deg",
        "time_ns",
    ]
    args.trials_output.parent.mkdir(parents=True, exist_ok=True)
    summaries = []
    with args.trials_output.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        for robot_name, filename in ROBOT_FILES.items():
            if len(targets[robot_name]) < args.trials:
                raise ValueError(f"Only {len(targets[robot_name])} targets for {robot_name}")
            model = build_rtb_robot(
                load_robot(str(PROJECT_ROOT / "robots" / filename))
            )
            for method in METHODS:
                errors = []
                durations = []
                successes = 0
                solver_successes = 0
                for trial, target_mm in targets[robot_name]:
                    target_m = target_to_metres(target_mm)
                    start = time.perf_counter_ns()
                    try:
                        result = solve(model, method, SE3(target_m))
                        duration = time.perf_counter_ns() - start
                        q, solver_success, iterations, reason, residual = solution_data(
                            result
                        )
                    except Exception as error:  # Preserve failed trials in raw data.
                        duration = time.perf_counter_ns() - start
                        q = np.full(6, math.nan)
                        solver_success = False
                        iterations = ""
                        reason = f"{type(error).__name__}: {error}"
                        residual = math.nan

                    durations.append(duration)
                    legacy_me = math.nan
                    matrix_mean = matrix_max = position_mm = rotation_deg = math.nan
                    experiment_success = False
                    if q.shape == (6,) and np.all(np.isfinite(q)):
                        actual_m = np.asarray(model.fkine(q).A, dtype=float)
                        legacy_me = float(np.mean(np.abs(target_m - actual_m)))
                        target_for_error = target_m.copy()
                        actual_for_error = actual_m.copy()
                        target_for_error[:3, 3] *= 1000.0
                        actual_for_error[:3, 3] *= 1000.0
                        metrics = pose_errors(target_for_error, actual_for_error)
                        matrix_mean = metrics.matrix_mean_abs_error
                        matrix_max = metrics.matrix_max_abs_error
                        position_mm = metrics.position_error_mm
                        rotation_deg = metrics.rotation_error_deg
                        experiment_success = (
                            solver_success and matrix_mean <= ME_TOLERANCE
                        )
                    solver_successes += int(solver_success)
                    successes += int(experiment_success)
                    if experiment_success:
                        errors.append(matrix_mean)
                    row = {
                        "robot": robot_name,
                        "trial": trial,
                        "method": method,
                        "solver_success": solver_success,
                        "experiment_success": experiment_success,
                        "iterations": iterations,
                        "reason": reason,
                        "residual": residual,
                        "legacy_matrix_mean_error_m": legacy_me,
                        "matrix_mean_error": matrix_mean,
                        "matrix_max_error": matrix_max,
                        "position_error_mm": position_mm,
                        "rotation_error_deg": rotation_deg,
                        "time_ns": duration,
                    }
                    row.update({field: q[index] for index, field in enumerate(Q_FIELDS)})
                    writer.writerow(row)

                time_values = np.asarray(durations, dtype=np.float64) / 1_000_000.0
                summaries.append(
                    {
                        "robot": robot_name,
                        "method": method,
                        "trials": args.trials,
                        "solver_successes": solver_successes,
                        "experiment_successes": successes,
                        "failures": args.trials - successes,
                        "mean_success_matrix_error": float(np.mean(errors))
                        if errors
                        else math.nan,
                        "mean_time_ms": float(np.mean(time_values)),
                        "median_time_ms": float(np.median(time_values)),
                        "p95_time_ms": float(np.percentile(time_values, 95)),
                        "p99_time_ms": float(np.percentile(time_values, 99)),
                    }
                )

    with args.summary_output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)


if __name__ == "__main__":
    main()
