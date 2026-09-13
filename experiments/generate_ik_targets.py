"""Generate deterministic FK targets for the analytical IK experiments."""

import argparse
import csv
from pathlib import Path

import numpy as np

try:
    from common import DEFAULT_SEED, PROCESSED_DIR, PROJECT_ROOT, forward_kinematics
except ImportError:  # Support ``python -m experiments.generate_ik_targets``.
    from experiments.common import (
        DEFAULT_SEED,
        PROCESSED_DIR,
        PROJECT_ROOT,
        forward_kinematics,
    )

from dh_transform_ik import load_robot


ROBOT_FILES = (
    ("abb_irb2600_20_165", "abb_irb2600_20_165.json"),
    ("estun_er8_720mi", "estun_er8_720mi.json"),
    ("kuka_kr50_r2500", "kuka_kr50_r2500.json"),
    ("yaskawa_gp12", "yaskawa_gp12.json"),
)
Q_FIELDS = [f"source_q{i}_deg" for i in range(1, 7)]
T_FIELDS = [f"target_{row}{col}" for row in range(4) for col in range(4)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate fixed FK-derived IK targets for four non-template robots."
    )
    parser.add_argument("--trials", type=int, default=10_000, help="Targets per robot")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--margin", type=float, default=0.05, help="Joint-limit margin")
    parser.add_argument(
        "--output", type=Path, default=PROCESSED_DIR / "ik_targets.csv"
    )
    return parser.parse_args()


def sample_joints(robot_dh, rng: np.random.Generator, margin: float) -> np.ndarray:
    lower = np.array([item.theta_min for item in robot_dh[:6]], dtype=float)
    upper = np.array([item.theta_max for item in robot_dh[:6]], dtype=float)
    span = upper - lower
    return rng.uniform(lower + margin * span, upper - margin * span)


def main() -> None:
    args = parse_args()
    if args.trials < 1:
        raise ValueError("--trials must be positive")
    if not 0.0 <= args.margin < 0.5:
        raise ValueError("--margin must be in [0, 0.5)")

    rng = np.random.default_rng(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["trial", "robot", *Q_FIELDS, *T_FIELDS]
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for robot_name, filename in ROBOT_FILES:
            robot_dh = load_robot(str(PROJECT_ROOT / "robots" / filename))
            for trial in range(args.trials):
                q_deg = sample_joints(robot_dh, rng, args.margin)
                target = np.asarray(forward_kinematics(robot_dh, q_deg), dtype=float)
                row = {"trial": trial, "robot": robot_name}
                row.update({name: value for name, value in zip(Q_FIELDS, q_deg)})
                row.update({name: value for name, value in zip(T_FIELDS, target.flat)})
                writer.writerow(row)


if __name__ == "__main__":
    main()
