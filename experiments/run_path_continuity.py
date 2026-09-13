"""Run an offline Cartesian-path IK continuity experiment on Yaskawa GP12."""

import argparse
import csv
import math
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
Q_FIELDS = [f"q{i}_rad" for i in range(1, 7)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline Yaskawa Cartesian-path branch continuity study."
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--line-paths", type=int, default=10)
    parser.add_argument("--arc-paths", type=int, default=10)
    parser.add_argument("--points", type=int, default=100, help="Points per path")
    parser.add_argument("--line-length-mm", type=float, default=80.0)
    parser.add_argument("--arc-radius-mm", type=float, default=40.0)
    parser.add_argument("--arc-angle-deg", type=float, default=120.0)
    parser.add_argument(
        "--trials-output", type=Path, default=PROCESSED_DIR / "path_trials.csv"
    )
    parser.add_argument(
        "--summary-output", type=Path, default=PROCESSED_DIR / "path_summary.csv"
    )
    return parser.parse_args()


def random_reference(robot_dh, rng: np.random.Generator) -> np.ndarray:
    lower = np.array([item.theta_min for item in robot_dh[:6]], dtype=float)
    upper = np.array([item.theta_max for item in robot_dh[:6]], dtype=float)
    q_deg = rng.uniform(lower + 0.15 * (upper - lower), upper - 0.15 * (upper - lower))
    # Keep reference paths away from the wrist singularity studied separately.
    if abs(q_deg[4]) < 20.0:
        q_deg[4] = 20.0 if q_deg[4] >= 0.0 else -20.0
    return np.radians(q_deg)


def orthonormal_pair(rng: np.random.Generator):
    first = rng.normal(size=3)
    first /= np.linalg.norm(first)
    second = rng.normal(size=3)
    second -= first * np.dot(first, second)
    second /= np.linalg.norm(second)
    return first, second


def build_path(kind, reference_pose, points, scale, rng, arc_angle_deg):
    first, second = orthonormal_pair(rng)
    poses = []
    if kind == "line":
        parameters = np.linspace(-0.5, 0.5, points)
        offsets = [parameter * scale * first for parameter in parameters]
    else:
        half_angle = math.radians(arc_angle_deg) / 2.0
        parameters = np.linspace(-half_angle, half_angle, points)
        offsets = [
            scale * ((math.cos(t) - 1.0) * first + math.sin(t) * second)
            for t in parameters
        ]
    for offset in offsets:
        pose = reference_pose.copy()
        pose[:3, 3] += offset
        poses.append(pose)
    return poses


def wrapped_delta(q: np.ndarray, reference: np.ndarray) -> np.ndarray:
    difference = q - reference
    return np.arctan2(np.sin(difference), np.cos(difference))


def within_limits(q_rad: np.ndarray, robot_dh) -> bool:
    q_deg = np.degrees(q_rad)
    return all(
        robot_dh[i].theta_min - 1e-9 <= q_deg[i] <= robot_dh[i].theta_max + 1e-9
        for i in range(6)
    )


def pose_metrics(target: np.ndarray, actual: np.ndarray):
    matrix_error = float(np.mean(np.abs(target - actual)))
    position_error = float(np.linalg.norm(target[:3, 3] - actual[:3, 3]))
    relative_rotation = target[:3, :3].T @ actual[:3, :3]
    cosine = float(np.clip((np.trace(relative_rotation) - 1.0) / 2.0, -1.0, 1.0))
    return matrix_error, position_error, math.degrees(math.acos(cosine))


def main() -> None:
    args = parse_args()
    if args.line_paths < 10 or args.arc_paths < 10:
        raise ValueError("At least 10 line paths and 10 arc paths are required")
    if args.points < 2:
        raise ValueError("--points must be at least 2")
    if args.line_length_mm <= 0.0 or args.arc_radius_mm <= 0.0:
        raise ValueError("Path scales must be positive")
    if not 0.0 < args.arc_angle_deg < 360.0:
        raise ValueError("--arc-angle-deg must lie between 0 and 360")

    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    template = load_robot(str(PROJECT_ROOT / "robots" / "fanuc_lr_mate_200id.json"))
    robot_dh = load_robot(str(PROJECT_ROOT / "robots" / "yaskawa_gp12.json"))
    transformed = transform_dh(template, robot_dh)
    args.trials_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)

    trial_fields = [
        "mode",
        "path_id",
        "path_type",
        "point",
        "x_mm",
        "y_mm",
        "z_mm",
        "fk_x_mm",
        "fk_y_mm",
        "fk_z_mm",
        "valid_candidate_count",
        "selected_branch",
        *Q_FIELDS,
        "matrix_mean_error",
        "position_error_mm",
        "rotation_error_deg",
        "wrapped_step_distance_rad",
        "point_success",
    ]
    summaries = []
    with args.trials_output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=trial_fields)
        writer.writeheader()
        path_specs = [("line", args.line_length_mm)] * args.line_paths + [
            ("arc", args.arc_radius_mm)
        ] * args.arc_paths
        for path_id, (kind, scale) in enumerate(path_specs):
            reference_q = random_reference(robot_dh, rng)
            reference_pose = np.asarray(
                forward_kinematics(robot_dh, np.degrees(reference_q)), dtype=float
            )
            poses = build_path(
                kind,
                reference_pose,
                args.points,
                scale,
                rng,
                args.arc_angle_deg,
            )
            previous = reference_q
            successes = 0
            branch_switches = 0
            previous_branch = ""
            distances = []

            for point, target in enumerate(poses):
                solutions = solve_ik(
                    target, template, robot_dh, transformed_dh=transformed
                )
                candidates = []
                for branch in CONFIGS:
                    q_rad = np.asarray(solutions.get(branch, [math.nan] * 6), dtype=float)
                    if q_rad.shape != (6,) or not np.all(np.isfinite(q_rad)):
                        continue
                    if not within_limits(q_rad, robot_dh):
                        continue
                    actual = np.asarray(
                        forward_kinematics(robot_dh, np.degrees(q_rad)), dtype=float
                    )
                    error, position_error, rotation_error = pose_metrics(target, actual)
                    if error < TOLERANCE:
                        distance = float(np.linalg.norm(wrapped_delta(q_rad, previous)))
                        candidates.append(
                            (
                                distance,
                                branch,
                                q_rad,
                                error,
                                position_error,
                                rotation_error,
                                actual,
                            )
                        )

                output = {
                    "mode": "offline",
                    "path_id": path_id,
                    "path_type": kind,
                    "point": point,
                    "x_mm": target[0, 3],
                    "y_mm": target[1, 3],
                    "z_mm": target[2, 3],
                    "fk_x_mm": math.nan,
                    "fk_y_mm": math.nan,
                    "fk_z_mm": math.nan,
                    "valid_candidate_count": len(candidates),
                    "selected_branch": "",
                    "matrix_mean_error": math.nan,
                    "position_error_mm": math.nan,
                    "rotation_error_deg": math.nan,
                    "wrapped_step_distance_rad": math.nan,
                    "point_success": False,
                }
                output.update({field: math.nan for field in Q_FIELDS})
                if candidates:
                    (
                        distance,
                        branch,
                        selected,
                        error,
                        position_error,
                        rotation_error,
                        actual,
                    ) = min(candidates, key=lambda item: item[0])
                    output.update(
                        {
                            "selected_branch": branch,
                            "matrix_mean_error": error,
                            "position_error_mm": position_error,
                            "rotation_error_deg": rotation_error,
                            "fk_x_mm": actual[0, 3],
                            "fk_y_mm": actual[1, 3],
                            "fk_z_mm": actual[2, 3],
                            "wrapped_step_distance_rad": distance,
                            "point_success": True,
                        }
                    )
                    output.update({field: selected[i] for i, field in enumerate(Q_FIELDS)})
                    successes += 1
                    distances.append(distance)
                    if previous_branch and branch != previous_branch:
                        branch_switches += 1
                    previous = selected
                    previous_branch = branch
                writer.writerow(output)

            summaries.append(
                {
                    "mode": "offline",
                    "path_id": path_id,
                    "path_type": kind,
                    "points": args.points,
                    "successful_points": successes,
                    "complete_path": successes == args.points,
                    "branch_switches": branch_switches,
                    "mean_wrapped_step_rad": float(np.mean(distances))
                    if distances
                    else math.nan,
                    "max_wrapped_step_rad": max(distances) if distances else math.nan,
                }
            )

    summary_fields = [
        "mode",
        "path_id",
        "path_type",
        "points",
        "successful_points",
        "complete_path",
        "branch_switches",
        "mean_wrapped_step_rad",
        "max_wrapped_step_rad",
    ]
    with args.summary_output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summaries)


if __name__ == "__main__":
    main()
