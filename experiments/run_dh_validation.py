"""Validate FK preservation for transformed DH parameters on five robots."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from dh_transform_ik import load_robot, transform_dh
from experiments.common import (
    DEFAULT_SEED,
    JOINT_COUNT,
    PROCESSED_DIR,
    ROBOT_REGISTRY,
    TEMPLATE_ROBOT,
    forward_kinematics,
    git_commit,
    map_arbitrary_to_standard_deg,
    pose_errors,
    sample_joint_angles,
    transformed_forward_kinematics,
    write_csv,
)

ERROR_FIELDS = [
    "position_error_mm",
    "rotation_error_deg",
    "matrix_mean_abs_error",
    "matrix_max_abs_error",
]


def summary_values(values: Sequence[float], prefix: str) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        f"{prefix}_mean": float(np.mean(array)),
        f"{prefix}_median": float(np.median(array)),
        f"{prefix}_p95": float(np.percentile(array, 95)),
        f"{prefix}_max": float(np.max(array)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    if args.trials <= 0:
        parser.error("--trials must be a positive integer")
    return args


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    commit = git_commit()
    template_dh = load_robot(str(TEMPLATE_ROBOT.source_path))
    validation_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for spec in ROBOT_REGISTRY:
        robot_dh = load_robot(str(spec.source_path))
        transformed_dh = transform_dh(template_dh, robot_dh)
        metrics: Dict[str, List[float]] = {field: [] for field in ERROR_FIELDS}

        for trial in range(1, args.trials + 1):
            arbitrary_joints = sample_joint_angles(robot_dh, rng)
            standard_joints = map_arbitrary_to_standard_deg(
                arbitrary_joints, transformed_dh
            )
            original_pose = forward_kinematics(robot_dh, arbitrary_joints)
            transformed_pose = transformed_forward_kinematics(
                transformed_dh, standard_joints
            )
            errors = pose_errors(original_pose, transformed_pose).to_dict()
            for field in ERROR_FIELDS:
                metrics[field].append(errors[field])

            row: Dict[str, Any] = {
                "robot_id": spec.robot_id,
                "display_name": spec.display_name,
                "trial": trial,
                "seed": args.seed,
                "source_file": spec.source_file,
                "source_commit": commit,
                **errors,
            }
            row.update(
                {f"q_arbitrary_{i + 1}_deg": q for i, q in enumerate(arbitrary_joints)}
            )
            row.update(
                {f"q_standard_{i + 1}_deg": q for i, q in enumerate(standard_joints)}
            )
            validation_rows.append(row)

        summary: Dict[str, Any] = {
            "robot_id": spec.robot_id,
            "display_name": spec.display_name,
            "trials": args.trials,
            "seed": args.seed,
            "source_file": spec.source_file,
            "source_commit": commit,
            "template_robot_id": TEMPLATE_ROBOT.robot_id,
            "template_source_file": TEMPLATE_ROBOT.source_file,
        }
        for field in ERROR_FIELDS:
            summary.update(summary_values(metrics[field], field))
        summary_rows.append(summary)

    joint_fields = [f"q_arbitrary_{i}_deg" for i in range(1, JOINT_COUNT + 1)]
    joint_fields += [f"q_standard_{i}_deg" for i in range(1, JOINT_COUNT + 1)]
    validation_fields = [
        "robot_id",
        "display_name",
        "trial",
        "seed",
        "source_file",
        "source_commit",
    ] + joint_fields + ERROR_FIELDS
    summary_fields = [
        "robot_id",
        "display_name",
        "trials",
        "seed",
        "source_file",
        "source_commit",
        "template_robot_id",
        "template_source_file",
    ]
    for field in ERROR_FIELDS:
        summary_fields.extend(
            f"{field}_{stat}" for stat in ("mean", "median", "p95", "max")
        )

    write_csv(
        PROCESSED_DIR / "dh_validation.csv", validation_fields, validation_rows
    )
    write_csv(
        PROCESSED_DIR / "dh_validation_summary.csv", summary_fields, summary_rows
    )


if __name__ == "__main__":
    main()
