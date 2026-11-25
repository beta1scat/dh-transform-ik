"""Export two chapter 2 LaTeX tables from complete unified IK runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from experiments.common import file_sha256, write_csv, write_json

ROBOTS = (
    ("kuka_kr50_r2500", "Kuka KR 50 R2500"),
    ("abb_irb2600_20_165", "ABB IRB2600-20/1.65"),
    ("yaskawa_gp12", "Yaskawa GP12"),
    ("estun_er8_720mi", "Estun ER8-720-MI"),
)
METHODS = ("analytical", "ikine_min", "ikine_LM", "ikine_LMS")
HEADERS = r"机器人 & 本文方法 & \texttt{ikine\_min} & \texttt{ikine\_LM} & \texttt{ikine\_LMS} \\"


def scientific(value):
    if not math.isfinite(value):
        return r"\text{--}"
    if value == 0:
        return "0"
    coefficient, exponent = f"{value:.2e}".split("e")
    return rf"{coefficient}\times10^{{{int(exponent)}}}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    experiments = []
    seeds = set()
    reference = None
    for directory in args.runs:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("status") != "complete":
            raise ValueError(f"Incomplete run: {directory}")
        settings = manifest["arguments"]
        fingerprint = {
            "input": manifest["input_sha256"], "sources": manifest["source_sha256"],
            "packages": manifest["packages"], "platform": manifest["platform"],
            "python": manifest["python"], "hardware": manifest.get("hardware"),
            "thread_environment": manifest.get("thread_environment"),
            "acceptance": manifest["acceptance"], "timing": manifest["timing"],
            "solver_options": manifest["solver_options"],
            "protocol": {key: settings[key] for key in ("trials", "attempts", "warmup", "me_tolerance")},
        }
        if reference is not None and fingerprint != reference:
            raise ValueError("Run inputs, code, environment or protocol differ; export separately")
        reference = fingerprint
        if settings["seed"] in seeds:
            raise ValueError("Duplicate initial-guess seed; export repetitions separately")
        seeds.add(settings["seed"])
        with (directory / "summary.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        records = {(row["robot"], row["method"]): row for row in rows}
        expected = {(robot, method) for robot, _ in ROBOTS for method in METHODS}
        if set(records) != expected or len(rows) != len(expected):
            raise ValueError(f"Missing or duplicate method/robot summaries: {directory}")
        for row in rows:
            if int(row["trials"]) != settings["trials"]:
                raise ValueError(f"Unexpected target count: {directory}")
            if int(row["successes"]) + int(row["failures"]) != settings["trials"]:
                raise ValueError(f"Inconsistent successes/failures: {directory}")
        experiments.append(records)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    count = len(experiments)
    trials = reference["protocol"]["trials"]
    attempts = reference["protocol"]["attempts"]
    aggregated = []
    cells = {}
    for robot, _ in ROBOTS:
        for method in METHODS:
            records = [run[(robot, method)] for run in experiments]
            def mean(field):
                return float(np.mean([float(row[field]) for row in records]))
            failures = np.array([int(row["failures"]) for row in records])
            row = {
                "robot": robot, "method": method, "targets_per_run": trials,
                "runs": count, "mean_failures": float(np.mean(failures)),
                "std_failures": float(np.std(failures, ddof=1)) if count > 1 else 0.0,
                "mean_success_position_error_mm": mean("mean_success_position_error_mm"),
                "mean_success_rotation_error_deg": mean("mean_success_rotation_error_deg"),
                "mean_time_ms": mean("mean_time_ms"),
                "mean_attempts": mean("mean_attempts"),
            }
            aggregated.append(row)
            cells[(robot, method)] = row
    repeat_note = (
        f"每种机器人{trials:,}个相同可达目标，数值法最多{attempts}次尝试；"
        if count == 1 else
        f"每种机器人{trials:,}个相同可达目标，{count}个初值种子，数值法最多{attempts}次尝试；"
    )
    accuracy = [
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{逆运动学成功样本回代误差与目标失败次数（" + repeat_note +
        (r"单元格为$e_p/e_R\;(N_f)$，单位分别为mm与度）}" if count == 1 else r"误差为各运行成功样本均值的平均，$N_f$为各运行失败数的均值$\pm$样本标准差；误差单位分别为mm与度）}"),
        r"\label{tab:ik_accuracy}", r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{lcccc}", r"\toprule", HEADERS, r"\midrule",
    ]
    timing = [
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{逆运动学目标级平均总耗时（" + repeat_note + r"包含数值重启及解的验收，全部目标均纳入计时，单位为ms）}",
        r"\label{tab:ik_efficiency}", r"\begin{tabular}{lcccc}",
        r"\toprule", HEADERS, r"\midrule",
    ]
    for robot, name in ROBOTS:
        error_cells, time_cells = [], []
        for method in METHODS:
            row = cells[(robot, method)]
            failure = str(int(row["mean_failures"])) if count == 1 else f'{row["mean_failures"]:.1f}\\pm{row["std_failures"]:.1f}'
            error_cells.append(f'${scientific(row["mean_success_position_error_mm"])}/{scientific(row["mean_success_rotation_error_deg"])}\\;({failure})$')
            time_cells.append(f'${row["mean_time_ms"]:.3f}$')
        accuracy.append(name + " & " + " & ".join(error_cells) + r" \\")
        timing.append(name + " & " + " & ".join(time_cells) + r" \\")
    accuracy += [r"\bottomrule", r"\end{tabular}", "}", r"\end{table}"]
    timing += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (args.output_dir / "ik_accuracy_table.tex").write_text("\n".join(accuracy) + "\n", encoding="utf-8")
    (args.output_dir / "ik_timing_table.tex").write_text("\n".join(timing) + "\n", encoding="utf-8")
    write_csv(args.output_dir / "table_values.csv", list(aggregated[0]), aggregated)
    write_json(args.output_dir / "table_manifest.json", {
        "run_directories": [str(path.resolve()) for path in args.runs],
        "initial_seeds": sorted(seeds), "protocol": reference,
        "summary_sha256": {str(path): file_sha256(path / "summary.csv") for path in args.runs},
        "error_aggregation": "mean of run-wise success-only means; no-success runs yield NaN, not zero",
        "failure_aggregation": "per-run failures, mean and sample SD for multiple seeds",
    })
    print(f"Wrote both LaTeX tables and provenance to {args.output_dir}")


if __name__ == "__main__":
    main()
