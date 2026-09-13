"""Measure complete online analytical IK latency using the fixed target set."""

from __future__ import annotations

import argparse
import csv
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from common import PROCESSED_DIR, PROJECT_ROOT
except ImportError:
    from experiments.common import PROCESSED_DIR, PROJECT_ROOT

from dh_transform_ik import load_robot, solve_ik, transform_dh


ROBOT_FILES = {
    "abb_irb2600_20_165": "abb_irb2600_20_165.json",
    "estun_er8_720mi": "estun_er8_720mi.json",
    "kuka_kr50_r2500": "kuka_kr50_r2500.json",
    "yaskawa_gp12": "yaskawa_gp12.json",
}
T_FIELDS = [f"target_{row}{column}" for row in range(4) for column in range(4)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=1_000)
    parser.add_argument("--trials", type=int, default=10_000, help="Calls per robot")
    parser.add_argument("--input", type=Path, default=PROCESSED_DIR / "ik_targets.csv")
    parser.add_argument(
        "--trials-output", type=Path, default=PROCESSED_DIR / "ik_timing_trials.csv"
    )
    parser.add_argument(
        "--summary-output", type=Path, default=PROCESSED_DIR / "ik_timing_summary.csv"
    )
    args = parser.parse_args()
    if args.warmup < 0 or args.trials <= 0:
        parser.error("--warmup must be nonnegative and --trials must be positive")
    return args


def main() -> None:
    args = parse_args()
    targets = defaultdict(list)
    with args.input.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            robot = row["robot"]
            if robot in ROBOT_FILES and len(targets[robot]) < args.trials:
                targets[robot].append(
                    np.array([float(row[field]) for field in T_FIELDS]).reshape(4, 4)
                )

    template = load_robot(str(PROJECT_ROOT / "robots" / "fanuc_lr_mate_200id.json"))
    rows = []
    summaries = []
    for robot, filename in ROBOT_FILES.items():
        robot_targets = targets[robot]
        if len(robot_targets) < args.trials:
            raise ValueError(f"Only {len(robot_targets)} targets available for {robot}")
        robot_dh = load_robot(str(PROJECT_ROOT / "robots" / filename))
        transformed = transform_dh(template, robot_dh)
        for index in range(args.warmup):
            solve_ik(
                robot_targets[index % len(robot_targets)],
                template,
                robot_dh,
                transformed_dh=transformed,
            )

        elapsed = []
        for trial, target in enumerate(robot_targets[: args.trials]):
            start = time.perf_counter_ns()
            solve_ik(target, template, robot_dh, transformed_dh=transformed)
            duration = time.perf_counter_ns() - start
            elapsed.append(duration)
            rows.append(
                {
                    "robot": robot,
                    "trial": trial,
                    "scope": "base_end_transform_plus_8_branch_ik_plus_joint_mapping",
                    "time_ns": duration,
                }
            )

        values = np.asarray(elapsed, dtype=np.float64)
        summaries.append(
            {
                "robot": robot,
                "warmup_calls": args.warmup,
                "measured_calls": args.trials,
                "scope": "base_end_transform_plus_8_branch_ik_plus_joint_mapping",
                "mean_us": float(np.mean(values) / 1_000.0),
                "median_us": float(np.median(values) / 1_000.0),
                "p90_us": float(np.percentile(values, 90) / 1_000.0),
                "p95_us": float(np.percentile(values, 95) / 1_000.0),
                "p99_us": float(np.percentile(values, 99) / 1_000.0),
                "min_us": float(np.min(values) / 1_000.0),
                "max_us": float(np.max(values) / 1_000.0),
            }
        )

    args.trials_output.parent.mkdir(parents=True, exist_ok=True)
    with args.trials_output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["robot", "trial", "scope", "time_ns"]
        )
        writer.writeheader()
        writer.writerows(rows)
    with args.summary_output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)


if __name__ == "__main__":
    main()
