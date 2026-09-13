"""Summarize legacy numerical IK records under explicit failure protocols."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from common import PROCESSED_DIR
except ImportError:
    from experiments.common import PROCESSED_DIR


TOLERANCE = 1e-3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=PROCESSED_DIR / "ik_numerical_legacy_trials.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROCESSED_DIR / "ik_numerical_protocol_summary.csv",
    )
    return parser.parse_args()


def as_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def main() -> None:
    args = parse_args()
    groups = defaultdict(
        lambda: {
            "trials": 0,
            "solver_successes": 0,
            "finite": 0,
            "threshold_successes": 0,
            "combined_successes": 0,
            "all_errors": [],
            "threshold_errors": [],
            "unified_errors": [],
            "unified_threshold_successes": 0,
            "unified_combined_successes": 0,
            "times_ms": [],
        }
    )
    with args.input.open("r", newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            item = groups[(row["robot"], row["method"])]
            item["trials"] += 1
            solver_success = as_bool(row["solver_success"])
            item["solver_successes"] += int(solver_success)
            item["times_ms"].append(float(row["time_ns"]) / 1_000_000.0)
            try:
                error = float(row["legacy_matrix_mean_error_m"])
            except (TypeError, ValueError):
                error = math.nan
            finite = math.isfinite(error)
            threshold_success = finite and error <= TOLERANCE
            combined_success = solver_success and threshold_success
            item["finite"] += int(finite)
            item["threshold_successes"] += int(threshold_success)
            item["combined_successes"] += int(combined_success)
            if finite:
                item["all_errors"].append(error)
            if threshold_success:
                item["threshold_errors"].append(error)
            try:
                unified_error = float(row["matrix_mean_error"])
            except (TypeError, ValueError):
                unified_error = math.nan
            if math.isfinite(unified_error):
                item["unified_errors"].append(unified_error)
                unified_threshold_success = unified_error <= TOLERANCE
                item["unified_threshold_successes"] += int(unified_threshold_success)
                item["unified_combined_successes"] += int(
                    solver_success and unified_threshold_success
                )

    rows = []
    for (robot, method), item in groups.items():
        all_errors = item["all_errors"]
        threshold_errors = item["threshold_errors"]
        unified_errors = item["unified_errors"]
        times = np.asarray(item["times_ms"], dtype=np.float64)
        rows.append(
            {
                "robot": robot,
                "method": method,
                "trials": item["trials"],
                "finite_results": item["finite"],
                "solver_successes": item["solver_successes"],
                "threshold_successes": item["threshold_successes"],
                "threshold_failures": item["trials"] - item["threshold_successes"],
                "combined_successes": item["combined_successes"],
                "combined_failures": item["trials"] - item["combined_successes"],
                "mean_all_finite_legacy_me": float(np.mean(all_errors))
                if all_errors
                else math.nan,
                "mean_threshold_success_legacy_me": float(np.mean(threshold_errors))
                if threshold_errors
                else math.nan,
                "mean_all_finite_unified_me": float(np.mean(unified_errors))
                if unified_errors
                else math.nan,
                "unified_threshold_successes": item["unified_threshold_successes"],
                "unified_threshold_failures": item["trials"]
                - item["unified_threshold_successes"],
                "unified_combined_successes": item["unified_combined_successes"],
                "unified_combined_failures": item["trials"]
                - item["unified_combined_successes"],
                "mean_time_ms": float(np.mean(times)),
                "median_time_ms": float(np.median(times)),
                "p95_time_ms": float(np.percentile(times, 95)),
                "p99_time_ms": float(np.percentile(times, 99)),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
