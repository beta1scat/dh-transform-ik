"""Shared paths and numerical helpers for the chapter 2 experiments."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dh_transform_ik import (  # noqa: E402
    DHParam,
    forward_kinematics_chain,
    forward_kinematics_mdh,
)

ROBOTS_DIR = PROJECT_ROOT / "robots"
DATA_DIR = PROJECT_ROOT / "data" / "chapter2"
METADATA_DIR = DATA_DIR / "metadata"
PROCESSED_DIR = DATA_DIR / "processed"

DEFAULT_SEED = 42
JOINT_LIMIT_MARGIN = 0.05
JOINT_COUNT = 6


@dataclass(frozen=True)
class RobotSpec:
    robot_id: str
    manufacturer: str
    model: str
    display_name: str
    filename: str
    is_template: bool = False

    @property
    def source_path(self) -> Path:
        return ROBOTS_DIR / self.filename

    @property
    def source_file(self) -> str:
        return self.source_path.relative_to(PROJECT_ROOT).as_posix()


ROBOT_REGISTRY: Sequence[RobotSpec] = (
    RobotSpec(
        "fanuc_lr_mate_200id",
        "FANUC",
        "LR Mate 200iD",
        "FANUC LR Mate 200iD",
        "fanuc_lr_mate_200id.json",
        True,
    ),
    RobotSpec(
        "kuka_kr50_r2500",
        "KUKA",
        "KR 50 R2500",
        "KUKA KR 50 R2500",
        "kuka_kr50_r2500.json",
    ),
    RobotSpec(
        "abb_irb2600_20_165",
        "ABB",
        "IRB 2600-20/1.65",
        "ABB IRB 2600-20/1.65",
        "abb_irb2600_20_165.json",
    ),
    RobotSpec(
        "yaskawa_gp12",
        "Yaskawa",
        "GP12",
        "Yaskawa GP12",
        "yaskawa_gp12.json",
    ),
    RobotSpec(
        "estun_er8_720mi",
        "ESTUN",
        "ER8-720-MI",
        "ESTUN ER8-720-MI",
        "estun_er8_720mi.json",
    ),
)
TEMPLATE_ROBOT = ROBOT_REGISTRY[0]


@dataclass(frozen=True)
class PoseErrors:
    position_error_mm: float
    rotation_error_deg: float
    matrix_mean_abs_error: float
    matrix_max_abs_error: float

    def to_dict(self) -> Mapping[str, float]:
        return {
            "position_error_mm": self.position_error_mm,
            "rotation_error_deg": self.rotation_error_deg,
            "matrix_mean_abs_error": self.matrix_mean_abs_error,
            "matrix_max_abs_error": self.matrix_max_abs_error,
        }


def sample_joint_angles(
    dh_list: Sequence[DHParam], rng: random.Random
) -> List[float]:
    """Sample six joints in degrees, excluding 5% at each limit boundary."""
    angles: List[float] = []
    for dh in dh_list[:JOINT_COUNT]:
        margin = (dh.theta_max - dh.theta_min) * JOINT_LIMIT_MARGIN
        angles.append(rng.uniform(dh.theta_min + margin, dh.theta_max - margin))
    return angles


def forward_kinematics(
    dh_list: Sequence[DHParam], joint_angles_deg: Sequence[float]
) -> np.ndarray:
    """Compute the original MDH chain pose in millimetres."""
    return forward_kinematics_chain(list(dh_list), list(joint_angles_deg))


def map_arbitrary_to_standard_deg(
    arbitrary_joints_deg: Sequence[float], transformed_dh: Sequence[DHParam]
) -> List[float]:
    """Apply the exact inverse of the joint mapping used by the IK solver."""
    standard: List[float] = []
    for index, arbitrary_deg in enumerate(arbitrary_joints_deg):
        dh = transformed_dh[index + 1]
        if dh.is_co_orient:
            standard.append(arbitrary_deg - dh.zero_offset)
        else:
            standard.append(-arbitrary_deg - dh.zero_offset)
    return standard


def transformed_forward_kinematics(
    transformed_dh: Sequence[DHParam], standard_joints_deg: Sequence[float]
) -> np.ndarray:
    """Compute base * six mapped joint transforms * end transform."""
    if len(transformed_dh) != JOINT_COUNT + 2:
        raise ValueError("Transformed DH data must contain base, six joints, and end")

    pose = np.eye(4, dtype=np.float64)
    for index, dh in enumerate(transformed_dh):
        theta = standard_joints_deg[index - 1] if 1 <= index <= JOINT_COUNT else 0.0
        pose = pose @ forward_kinematics_mdh(
            dh.alpha, dh.a, dh.theta_offset, dh.d, theta
        )
    return pose


def pose_errors(first: np.ndarray, second: np.ndarray) -> PoseErrors:
    """Measure translation, geodesic rotation, and elementwise matrix errors."""
    delta = np.abs(first - second)
    position_error = float(np.linalg.norm(first[:3, 3] - second[:3, 3]))
    relative_rotation = first[:3, :3].T @ second[:3, :3]
    cosine = float(np.clip((np.trace(relative_rotation) - 1.0) / 2.0, -1.0, 1.0))
    return PoseErrors(
        position_error_mm=position_error,
        rotation_error_deg=math.degrees(math.acos(cosine)),
        matrix_mean_abs_error=float(np.mean(delta)),
        matrix_max_abs_error=float(np.max(delta)),
    )


def git_commit() -> Optional[str]:
    """Return the repository HEAD, or None when Git metadata is unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = result.stdout.strip()
    return commit or None


def git_worktree_dirty() -> Optional[bool]:
    """Report whether tracked or untracked files differ from HEAD."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return bool(result.stdout.strip())


def file_sha256(path: Path) -> str:
    """Return a source file checksum for provenance records."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(
    path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
