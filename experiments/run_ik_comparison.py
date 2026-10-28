"""Run analytical and multi-start numerical IK under one acceptance protocol.

Run from the project root with ``python -m experiments.run_ik_comparison``.
Output directories must be new, so previous measurements are never overwritten.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import math
import os
import platform
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
from spatialmath import SE3

from dh_transform_ik import CONFIGS, load_robot, solve_ik, transform_dh
from experiments.common import (
    PROJECT_ROOT, PROCESSED_DIR, file_sha256, forward_kinematics,
    git_commit, git_worktree_dirty, write_csv, write_json,
)
from experiments.ik_comparison_protocol import (
    evaluate_solution, initial_guesses, joint_limits_rad,
)
from experiments.ik_robot_models import ROBOT_FILES, METHODS, T_FIELDS, build_rtb_robot

METHOD_ORDER = ("analytical", *METHODS)
METRIC_FIELDS = (
    "finite", "within_limits", "experiment_success", "matrix_mean_error",
    "position_error_mm", "rotation_error_deg", "failure_reason",
    *(f"q{i}_rad" for i in range(1, 7)),
)
ATTEMPT_FIELDS = (
    "robot", "trial", "method", "attempt", "branch", "solver_success",
    "iterations", "solver_reason", "time_ns", "cumulative_time_ns",
    *(f"q0_{i}_rad" for i in range(1, 7)), *METRIC_FIELDS,
)
TARGET_FIELDS = (
    "robot", "trial", "method", "initial_seed", "attempts_used",
    "accepted_solver_success", "time_ns", *METRIC_FIELDS,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PROCESSED_DIR / "ik_targets.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=10_000, help="Targets per robot")
    parser.add_argument("--attempts", type=int, default=10, help="Maximum starts per numerical method")
    parser.add_argument("--seed", type=int, default=142, help="Initial-guess seed, independent of target generation")
    parser.add_argument("--warmup", type=int, default=100, help="Untimed calls per method and robot")
    parser.add_argument("--me-tolerance", type=float, default=1e-3)
    args = parser.parse_args()
    if args.trials < 1 or args.attempts < 1 or args.warmup < 0 or args.seed < 0:
        parser.error("trials/attempts must be positive; warmup/seed must be nonnegative")
    if not math.isfinite(args.me_tolerance) or args.me_tolerance <= 0:
        parser.error("me-tolerance must be finite and positive")
    return args


def load_targets(path, count):
    targets = defaultdict(list)
    identifiers = set()
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            robot = row["robot"]
            if robot not in ROBOT_FILES or len(targets[robot]) >= count:
                continue
            trial = int(row["trial"])
            if trial < 0 or (robot, trial) in identifiers:
                raise ValueError(f"Invalid or duplicate target identifier: {robot}/{trial}")
            identifiers.add((robot, trial))
            target = np.array([float(row[key]) for key in T_FIELDS]).reshape(4, 4)
            if not np.all(np.isfinite(target)):
                raise ValueError(f"Nonfinite target: {robot}/{trial}")
            targets[robot].append((trial, target))
    for robot in ROBOT_FILES:
        if len(targets[robot]) != count:
            raise ValueError(f"Expected {count} targets for {robot}, got {len(targets[robot])}")
    return targets


def numerical_options(model):
    """Pin starts and disable internal random search where the API exposes it."""
    options = {}
    for method in METHODS:
        signature = inspect.signature(getattr(model, method))
        if "q0" not in signature.parameters:
            raise RuntimeError(f"{method} has no q0 parameter; use the documented RTB 1.0.3 environment")
        kwargs = {}
        if "search" in signature.parameters:
            kwargs["search"] = False
        if "qlim" in signature.parameters:
            kwargs["qlim"] = True
        options[method] = kwargs
    return options


def numerical_call(model, method, target, q0, options, target_mm, robot_dh, lower, upper, tolerance):
    start = time.perf_counter_ns()
    solver_success = ""
    iterations = ""
    reason = ""
    try:
        solution = getattr(model, method)(target, q0=q0.copy(), **options)
        q = getattr(solution, "q", np.full(6, math.nan))
        raw_success = getattr(solution, "success", None)
        solver_success = bool(raw_success) if raw_success is not None else ""
        iterations = getattr(solution, "iterations", "")
        reason = str(getattr(solution, "reason", ""))
        checked = evaluate_solution(q, target_mm, robot_dh, lower, upper, tolerance)
    except Exception as error:
        checked = evaluate_solution(np.full(6, math.nan), target_mm, robot_dh, lower, upper, tolerance)
        checked["failure_reason"] = "solver_exception"
        reason = f"{type(error).__name__}: {error}"
    return checked, solver_success, iterations, reason, time.perf_counter_ns() - start



def analytical_call(target_mm, template, robot_dh, transformed, lower, upper, tolerance):
    start = time.perf_counter_ns()
    reason = ""
    try:
        branches = solve_ik(target_mm, template, robot_dh, transformed_dh=transformed)
        checked = [(branch, evaluate_solution(branches[branch], target_mm, robot_dh, lower, upper, tolerance)) for branch in CONFIGS]
    except Exception as error:
        reason = f"{type(error).__name__}: {error}"
        checked = []
        for branch in CONFIGS:
            result = evaluate_solution(np.full(6, math.nan), target_mm, robot_dh, lower, upper, tolerance)
            result["failure_reason"] = "solver_exception"
            checked.append((branch, result))
    valid = [pair for pair in checked if pair[1]["experiment_success"]]
    best = min(valid or checked, key=lambda pair: pair[1]["matrix_mean_error"] if math.isfinite(pair[1]["matrix_mean_error"]) else math.inf)[1]
    return checked, best, reason, time.perf_counter_ns() - start


def summary_rows(groups):
    rows = []
    for robot in ROBOT_FILES:
        for method in METHOD_ORDER:
            values = groups[(robot, method)]
            successful = [row for row in values if row["experiment_success"]]
            times = np.array([row["time_ns"] for row in values], dtype=float) / 1e6
            rows.append({
                "robot": robot, "method": method, "trials": len(values),
                "successes": len(successful), "failures": len(values) - len(successful),
                "success_rate": len(successful) / len(values),
                "mean_success_position_error_mm": float(np.mean([r["position_error_mm"] for r in successful])) if successful else math.nan,
                "mean_success_rotation_error_deg": float(np.mean([r["rotation_error_deg"] for r in successful])) if successful else math.nan,
                "accepted_solver_successes": sum(r["accepted_solver_success"] is True for r in successful),
                "accepted_solver_false": sum(r["accepted_solver_success"] is False for r in successful),
                "mean_attempts": float(np.mean([r["attempts_used"] for r in values])),
                "last_attempt_exceptions": sum(r["failure_reason"] == "solver_exception" for r in values),
                "mean_time_ms": float(np.mean(times)),
                "median_time_ms": float(np.median(times)),
                "p95_time_ms": float(np.percentile(times, 95)),
            })
    return rows


def main():
    args = parse_args()
    targets = load_targets(args.input, args.trials)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    source_paths = [
        Path(__file__), Path(__file__).with_name("ik_comparison_protocol.py"),
        Path(__file__).with_name("common.py"), Path(__file__).with_name("ik_robot_models.py"),
        *(PROJECT_ROOT / "dh_transform_ik" / name for name in ("dh.py", "transform.py", "ik.py")),
        *(PROJECT_ROOT / "robots" / name for name in (*ROBOT_FILES.values(), "fanuc_lr_mate_200id.json")),
    ]
    packages = {}
    for package in ("numpy", "scipy", "roboticstoolbox-python", "spatialmath-python"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    manifest = {
        "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "input_sha256": file_sha256(args.input), "git_commit": git_commit(), "git_worktree_dirty": git_worktree_dirty(),
        "source_sha256": {path.relative_to(PROJECT_ROOT).as_posix(): file_sha256(path) for path in source_paths},
        "python": sys.version, "platform": platform.platform(), "packages": packages,
        "hardware": {"machine": platform.machine(), "processor": platform.processor(), "logical_cpu_count": os.cpu_count()},
        "thread_environment": {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
        "acceptance": "finite + periodic-equivalent actual joint limits + ME_mm <= tolerance; solver status diagnostic only",
        "timing": "solve + restart loop + normalization + original-MDH FK acceptance; excludes file IO, model setup, q0 generation and target conversion",
        "initial_guesses": "midpoint, then uniform actual limits; SeedSequence([seed, robot_index, trial])",
        "robot_order": list(ROBOT_FILES), "methods": METHOD_ORDER,
    }
    write_json(args.output_dir / "manifest.json", manifest)
    template = load_robot(str(PROJECT_ROOT / "robots" / "fanuc_lr_mate_200id.json"))
    groups = defaultdict(list)
    try:
        with (args.output_dir / "attempts.csv").open("w", newline="", encoding="utf-8") as attempt_file, (args.output_dir / "targets.csv").open("w", newline="", encoding="utf-8") as target_file, (args.output_dir / "initial_guesses.csv").open("w", newline="", encoding="utf-8") as initial_file:
            attempt_writer = csv.DictWriter(attempt_file, fieldnames=ATTEMPT_FIELDS)
            target_writer = csv.DictWriter(target_file, fieldnames=TARGET_FIELDS)
            initial_writer = csv.DictWriter(initial_file, fieldnames=("robot", "trial", "attempt", *(f"q0_{i}_rad" for i in range(1, 7))))
            for writer in (attempt_writer, target_writer, initial_writer):
                writer.writeheader()
            for robot_index, (robot, filename) in enumerate(ROBOT_FILES.items()):
                robot_dh = load_robot(str(PROJECT_ROOT / "robots" / filename))
                lower, upper = joint_limits_rad(robot_dh)
                model = build_rtb_robot(robot_dh)
                for link, lo, hi in zip(model.links, lower, upper):
                    link.qlim = np.array([lo, hi])
                transformed = transform_dh(template, robot_dh)
                options = numerical_options(model)
                manifest.setdefault("solver_options", {})[robot] = {
                    method: {"explicit_options": options[method], "signature": str(inspect.signature(getattr(model, method)))}
                    for method in METHODS
                }
                write_json(args.output_dir / "manifest.json", manifest)
                # Unit/model preflight before collecting measurements.
                midpoint = (lower + upper) / 2.0
                reference = forward_kinematics(robot_dh, np.degrees(midpoint))
                toolbox_pose = model.fkine(midpoint).A.copy()
                toolbox_pose[:3, 3] *= 1000.0
                if not np.allclose(reference, toolbox_pose, rtol=0, atol=1e-8):
                    raise ValueError(f"Original and toolbox MDH chains disagree for {robot}")
                for index in range(args.warmup):
                    _, target_mm = targets[robot][index % args.trials]
                    target_m = target_mm.copy()
                    target_m[:3, 3] /= 1000.0
                    target = SE3(target_m)
                    analytical_call(target_mm, template, robot_dh, transformed, lower, upper, args.me_tolerance)
                    for method in METHODS:
                        numerical_call(model, method, target, midpoint, options[method], target_mm, robot_dh, lower, upper, args.me_tolerance)
                for target_index, (trial, target_mm) in enumerate(targets[robot]):
                    guesses = initial_guesses(args.seed, robot_index, trial, args.attempts, lower, upper)
                    for attempt, q0 in enumerate(guesses, 1):
                        initial_writer.writerow({"robot": robot, "trial": trial, "attempt": attempt, **{f"q0_{i + 1}_rad": float(q) for i, q in enumerate(q0)}})
                    checked_branches, best, analytical_reason, duration = analytical_call(target_mm, template, robot_dh, transformed, lower, upper, args.me_tolerance)
                    row = {"robot": robot, "trial": trial, "method": "analytical", "initial_seed": "", "attempts_used": 1, "accepted_solver_success": "", "time_ns": duration, **best}
                    target_writer.writerow(row)
                    groups[(robot, "analytical")].append(row)
                    for branch, checked in checked_branches:
                        attempt_writer.writerow({"robot": robot, "trial": trial, "method": "analytical", "branch": branch, "solver_reason": analytical_reason, **checked})
                    target_m = target_mm.copy()
                    target_m[:3, 3] /= 1000.0
                    target = SE3(target_m)
                    for method in METHODS:
                        attempt_rows = []
                        start = time.perf_counter_ns()
                        for attempt, q0 in enumerate(guesses, 1):
                            checked, status, iterations, reason, elapsed = numerical_call(model, method, target, q0, options[method], target_mm, robot_dh, lower, upper, args.me_tolerance)
                            cumulative = time.perf_counter_ns() - start
                            attempt_rows.append({"robot": robot, "trial": trial, "method": method, "attempt": attempt, "solver_success": status, "iterations": iterations, "solver_reason": reason, "time_ns": elapsed, "cumulative_time_ns": cumulative, **{f"q0_{i + 1}_rad": float(q) for i, q in enumerate(q0)}, **checked})
                            if checked["experiment_success"]:
                                break
                        duration = time.perf_counter_ns() - start
                        row = {"robot": robot, "trial": trial, "method": method, "initial_seed": args.seed, "attempts_used": attempt, "accepted_solver_success": status if checked["experiment_success"] else "", "time_ns": duration, **checked}
                        attempt_writer.writerows(attempt_rows)
                        target_writer.writerow(row)
                        groups[(robot, method)].append(row)
                    if (target_index + 1) % 100 == 0 or target_index + 1 == args.trials:
                        attempt_file.flush()
                        target_file.flush()
                        initial_file.flush()
                        print(f"{robot}: {target_index + 1}/{args.trials} targets", flush=True)
        summaries = summary_rows(groups)
        write_csv(args.output_dir / "summary.csv", list(summaries[0]), summaries)
    except BaseException as error:
        manifest.update(status="interrupted_or_failed", error=f"{type(error).__name__}: {error}")
        write_json(args.output_dir / "manifest.json", manifest)
        raise
    manifest.update(status="complete", finished_utc=datetime.now(timezone.utc).isoformat())
    write_json(args.output_dir / "manifest.json", manifest)
    print(f"Completed: {args.output_dir}")


if __name__ == "__main__":
    main()
