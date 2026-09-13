"""Export robot provenance and original/transformed DH tables for chapter 2."""

from __future__ import annotations

import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from dh_transform_ik import DHParam, __version__, load_robot, transform_dh
from experiments.common import (
    DEFAULT_SEED,
    JOINT_COUNT,
    JOINT_LIMIT_MARGIN,
    METADATA_DIR,
    PROCESSED_DIR,
    ROBOT_REGISTRY,
    TEMPLATE_ROBOT,
    RobotSpec,
    file_sha256,
    git_commit,
    git_worktree_dirty,
    write_csv,
    write_json,
)

TRANSFORM_SOURCE = PROJECT_ROOT / "dh_transform_ik" / "transform.py"

PROVENANCE_FIELDS = [
    "robot_id",
    "display_name",
    "manufacturer",
    "model",
    "is_template",
    "source_file",
    "source_commit",
]
DH_FIELDS = [
    "row_index",
    "frame_role",
    "alpha_deg",
    "a_mm",
    "theta_offset_deg",
    "d_mm",
    "theta_min_deg",
    "theta_max_deg",
    "is_co_orient",
    "zero_offset_deg",
]


def provenance(spec: RobotSpec, commit: Optional[str]) -> Dict[str, Any]:
    return {
        "robot_id": spec.robot_id,
        "display_name": spec.display_name,
        "manufacturer": spec.manufacturer,
        "model": spec.model,
        "is_template": spec.is_template,
        "source_file": spec.source_file,
        "source_commit": commit,
    }


def original_role(index: int) -> str:
    return f"joint_{index + 1}" if index < JOINT_COUNT else "end_effector"


def transformed_role(index: int) -> str:
    if index == 0:
        return "base_transform"
    if index == JOINT_COUNT + 1:
        return "end_transform"
    return f"joint_{index}"


def dh_values(dh: DHParam, index: int, role: str) -> Dict[str, Any]:
    return {
        "row_index": index,
        "frame_role": role,
        "alpha_deg": dh.alpha,
        "a_mm": dh.a,
        "theta_offset_deg": dh.theta_offset,
        "d_mm": dh.d,
        "theta_min_deg": dh.theta_min,
        "theta_max_deg": dh.theta_max,
        "is_co_orient": dh.is_co_orient,
        "zero_offset_deg": dh.zero_offset,
    }


def main() -> None:
    commit = git_commit()
    template_dh = load_robot(str(TEMPLATE_ROBOT.source_path))
    registry_rows: List[Dict[str, Any]] = []
    original_rows: List[Dict[str, Any]] = []
    transformed_rows: List[Dict[str, Any]] = []

    for spec in ROBOT_REGISTRY:
        source = provenance(spec, commit)
        robot_dh = load_robot(str(spec.source_path))
        transformed_dh = transform_dh(template_dh, robot_dh)
        registry_rows.append(
            {
                **source,
                "source_sha256": file_sha256(spec.source_path),
                "joint_count": JOINT_COUNT,
                "original_dh_row_count": len(robot_dh),
                "transformed_dh_row_count": len(transformed_dh),
                "kinematic_convention": "modified Denavit-Hartenberg",
                "angle_unit": "degree",
                "length_unit": "millimetre",
            }
        )
        for index, dh in enumerate(robot_dh):
            original_rows.append(
                {**source, **dh_values(dh, index, original_role(index))}
            )
        for index, dh in enumerate(transformed_dh):
            transformed_rows.append(
                {
                    **source,
                    "template_robot_id": TEMPLATE_ROBOT.robot_id,
                    "template_source_file": TEMPLATE_ROBOT.source_file,
                    "template_source_commit": commit,
                    "algorithm_source_file": TRANSFORM_SOURCE.relative_to(PROJECT_ROOT).as_posix(),
                    "algorithm_source_sha256": file_sha256(TRANSFORM_SOURCE),
                    **dh_values(dh, index, transformed_role(index)),
                }
            )

    registry_fields = PROVENANCE_FIELDS + [
        "source_sha256",
        "joint_count",
        "original_dh_row_count",
        "transformed_dh_row_count",
        "kinematic_convention",
        "angle_unit",
        "length_unit",
    ]
    transformed_fields = PROVENANCE_FIELDS + [
        "template_robot_id",
        "template_source_file",
        "template_source_commit",
        "algorithm_source_file",
        "algorithm_source_sha256",
    ] + DH_FIELDS
    write_csv(METADATA_DIR / "robot_registry.csv", registry_fields, registry_rows)
    write_json(
        METADATA_DIR / "environment.json",
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": commit,
            "git_worktree_dirty": git_worktree_dirty(),
            "transform_source_sha256": file_sha256(TRANSFORM_SOURCE),
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "numpy_version": np.__version__,
            "dh_transform_ik_version": __version__,
            "default_seed": DEFAULT_SEED,
            "joint_limit_margin_fraction": JOINT_LIMIT_MARGIN,
            "project_root": str(PROJECT_ROOT),
        },
    )
    write_csv(
        PROCESSED_DIR / "dh_original.csv",
        PROVENANCE_FIELDS + DH_FIELDS,
        original_rows,
    )
    write_csv(
        PROCESSED_DIR / "dh_transformed.csv", transformed_fields, transformed_rows
    )


if __name__ == "__main__":
    main()
