"""Paper-style FK/IK benchmark using the shared analytical implementation.

The analytical path enumerates branches; the numerical baseline makes one
Robotics Toolbox LM call. Timing covers the solver call only. See README.md.
"""
from __future__ import annotations

import argparse
import math
import platform
import random
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np

from dh_transform_ik import CONFIGS, load_robot, solve_ik, transform_dh
from experiments.common import (
    PROJECT_ROOT, ROBOT_REGISTRY, TEMPLATE_ROBOT, file_sha256, forward_kinematics,
    git_commit, pose_errors, sample_joint_angles, write_csv, write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analytical-trials", type=int, default=10_000)
    parser.add_argument("--numerical-trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--analytical-only", action="store_true")
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "data" / "paper" / "benchmark")
    args = parser.parse_args()
    if args.analytical_trials < 1 or args.numerical_trials < 1:
        parser.error("trial counts must be positive")
    if not 0 <= args.margin < 0.5 or args.seed < 0:
        parser.error("margin must be in [0, 0.5); seed must be nonnegative")
    return args


def main():
    args = parse_args()
    # Optional dependencies are imported only when the baseline is requested.
    if not args.analytical_only:
        from experiments.common.rtb_models import build_rtb_robot
        from spatialmath import SE3
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    template = load_robot(str(TEMPLATE_ROBOT.source_path))
    sources = [Path(__file__), PROJECT_ROOT / "experiments/common/__init__.py",
               PROJECT_ROOT / "experiments/common/rtb_models.py",
               *(PROJECT_ROOT / "dh_transform_ik" / name for name in ("dh.py", "ik.py", "transform.py")),
               *(spec.source_path for spec in ROBOT_REGISTRY)]
    manifest = {
        "status": "running", "git_commit": git_commit(),
        "arguments": {key: str(value) if isinstance(value, Path) else value
                      for key, value in vars(args).items()},
        "python": platform.python_version(), "platform": platform.platform(),
        "numpy": np.__version__,
        "source_sha256": {p.relative_to(PROJECT_ROOT).as_posix(): file_sha256(p) for p in sources},
        "timing": "solver call only; excludes FK acceptance and file IO",
        "acceptance": "finite + original-MDH FK ME_mm <= 1e-3; LM also requires success=True; no limit rejection",
    }
    if not args.analytical_only:
        manifest["packages"] = {name: version(name) for name in
                                ("roboticstoolbox-python", "spatialmath-python", "scipy")}
    write_json(args.output_dir / "manifest.json", manifest)
    records, target_records, summaries = [], [], []
    for spec in ROBOT_REGISTRY:
        robot = load_robot(str(spec.source_path))
        transformed = transform_dh(template, robot)
        count = args.analytical_trials if args.analytical_only else max(args.analytical_trials, args.numerical_trials)
        targets = []
        for trial in range(count):
            source_q = sample_joint_angles(robot, rng, args.margin)
            target = forward_kinematics(robot, source_q)
            targets.append(target)
            target_records.append({"robot": spec.robot_id, "trial": trial,
                                   **{f"source_q{i+1}_deg": value for i, value in enumerate(source_q)},
                                   **{f"target_{r}{c}": target[r, c] for r in range(4) for c in range(4)}})
        methods = [("analytical", args.analytical_trials)]
        if not args.analytical_only:
            methods.append(("ikine_LM", args.numerical_trials))
            model = build_rtb_robot(robot)
        for method, trials in methods:
            group = []
            for trial, target in enumerate(targets[:trials]):
                target_m = target.copy()
                target_m[:3, 3] /= 1000.0
                if method == "ikine_LM":
                    toolbox_target = SE3(target_m)
                accepted, error, branch, message = False, math.nan, "", ""
                start = time.perf_counter_ns()
                try:
                    if method == "analytical":
                        solutions = solve_ik(target, template, robot, transformed_dh=transformed)
                        solver_success = ""
                    else:
                        solution = model.ikine_LM(toolbox_target)
                        solver_success = bool(solution.success)
                except Exception as exc:
                    message = f"{type(exc).__name__}: {exc}"
                    solver_success = False
                duration = time.perf_counter_ns() - start
                if not message:
                    candidates = ((name, solutions[name]) for name in CONFIGS) if method == "analytical" else [("", solution.q)]
                    for name, q in candidates:
                        q = np.asarray(q, dtype=float)
                        if q.shape != (6,) or not np.all(np.isfinite(q)):
                            continue
                        actual = forward_kinematics(robot, np.degrees(q))
                        candidate_error = pose_errors(target, actual).matrix_mean_abs_error
                        if candidate_error <= 1e-3 and (method == "analytical" or solver_success):
                            accepted, error, branch = True, candidate_error, name
                            break
                row = {"robot": spec.robot_id, "trial": trial, "method": method,
                       "pose_success": accepted, "matrix_mean_error_mm": error,
                       "solver_success": solver_success, "selected_branch": branch,
                       "time_ns": duration, "exception": message}
                records.append(row)
                group.append(row)
            valid = [row for row in group if row["pose_success"]]
            summary = {"robot": spec.robot_id, "method": method, "trials": trials,
                       "successes": len(valid), "failures": trials - len(valid),
                       "pose_success_rate": len(valid) / trials,
                       "mean_success_matrix_error_mm": float(np.mean([row["matrix_mean_error_mm"] for row in valid])) if valid else math.nan,
                       "mean_solver_time_us": float(np.mean([row["time_ns"] for row in group])) / 1000.0}
            summaries.append(summary)
            print(f"{spec.display_name}: {method} {len(valid)}/{trials}, "
                  f"mean solver time {summary['mean_solver_time_us']:.3f} us")
    for filename, rows in (("targets.csv", target_records), ("trials.csv", records), ("summary.csv", summaries)):
        write_csv(args.output_dir / filename, list(rows[0]), rows)
    manifest["status"] = "complete"
    write_json(args.output_dir / "manifest.json", manifest)


if __name__ == "__main__":
    main()
