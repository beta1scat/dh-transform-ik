"""Plot representative offline line and arc trajectories for chapter 2."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

try:
    from common import PROCESSED_DIR, PROJECT_ROOT
except ImportError:
    from experiments.common import PROCESSED_DIR, PROJECT_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PROCESSED_DIR / "path_trials.csv")
    parser.add_argument("--line-id", type=int, default=2)
    parser.add_argument("--arc-id", type=int, default=18)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT.parents[1]
        / "figures"
        / "chapter2"
        / "2_6a_fig_offline_path.pdf",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    song = FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc", size=14.4)
    song_title = FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc", size=14.4)
    times = FontProperties(fname=r"C:\Windows\Fonts\times.ttf", size=14.4)
    times_italic = FontProperties(fname=r"C:\Windows\Fonts\timesi.ttf", size=14.4)
    times_panel = FontProperties(fname=r"C:\Windows\Fonts\times.ttf", size=14.4)
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 14.4,
            "axes.titlesize": 14.4,
            "axes.labelsize": 14.4,
            "xtick.labelsize": 14.4,
            "ytick.labelsize": 14.4,
            "legend.fontsize": 14.4,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "savefig.dpi": 600,
        }
    )
    paths = defaultdict(list)
    with args.input.open("r", newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["point_success"].lower() == "true":
                paths[int(row["path_id"])].append(row)

    selected = [(args.line_id, "Line"), (args.arc_id, "Arc")]
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
    joint_labels = ["q₁", "q₂", "q₃", "q₄", "q₅", "q₆"]
    fig = plt.figure(figsize=(9.2, 8.4))
    grid = fig.add_gridspec(2, 2, height_ratios=(0.9, 1.43))
    joint_handles = None
    trajectory_handles = None
    for column, (path_id, title) in enumerate(selected):
        rows = paths[path_id]
        if not rows:
            raise ValueError(f"No successful points found for path {path_id}")
        points = np.array([int(row["point"]) for row in rows])
        joints = np.array(
            [[float(row[f"q{i}_rad"]) for i in range(1, 7)] for row in rows]
        )
        unwrapped = np.unwrap(joints, axis=0)
        joint_changes = np.degrees(unwrapped - unwrapped[len(unwrapped) // 2])
        target = np.array(
            [[float(row[f"{axis}_mm"]) for axis in ("x", "y", "z")] for row in rows]
        )
        actual = np.array(
            [
                [float(row[f"fk_{axis}_mm"]) for axis in ("x", "y", "z")]
                for row in rows
            ]
        )

        joint_ax = fig.add_subplot(grid[0, column])
        for index in range(6):
            joint_ax.plot(
                points,
                joint_changes[:, index],
                color=colors[index],
                linewidth=1.4,
                label=joint_labels[index],
            )
        path_name = "直线路径" if title == "Line" else "圆弧路径"
        panel = "(a)" if column == 0 else "(b)"
        joint_ax.set_title(
            f"{path_name}关节角变化", fontproperties=song_title, pad=8
        )
        joint_ax.text(
            -0.09,
            1.05,
            panel,
            transform=joint_ax.transAxes,
            fontproperties=times_panel,
            clip_on=False,
        )
        joint_ax.set_xlabel("路径点序号", fontproperties=song)
        joint_ax.set_ylabel("相对关节角（°）", fontproperties=song)
        joint_ax.grid(alpha=0.25, linewidth=0.5)
        if joint_handles is None:
            joint_handles, _ = joint_ax.get_legend_handles_labels()

        trajectory_ax = fig.add_subplot(grid[1, column], projection="3d")
        trajectory_ax.plot(
            target[:, 0],
            target[:, 1],
            target[:, 2],
            color="#222222",
            linewidth=2.2,
            alpha=0.85,
            label="目标路径",
        )
        trajectory_ax.plot(
            actual[:, 0],
            actual[:, 1],
            actual[:, 2],
            color="#D55E00",
            linestyle="--",
            linewidth=1.5,
            label="正运动学回代路径",
        )
        trajectory_ax.scatter(*target[0], color="#009E73", s=28, label="起点")
        trajectory_ax.scatter(*target[-1], color="#CC79A7", s=28, label="终点")
        all_points = np.vstack((target, actual))
        center = (all_points.min(axis=0) + all_points.max(axis=0)) / 2.0
        span = max(float(np.ptp(all_points, axis=0).max()), 1.0) * 1.15
        trajectory_ax.set_xlim(center[0] - span / 2.0, center[0] + span / 2.0)
        trajectory_ax.set_ylim(center[1] - span / 2.0, center[1] + span / 2.0)
        trajectory_ax.set_zlim(center[2] - span / 2.0, center[2] + span / 2.0)
        trajectory_ax.set_box_aspect((1, 1, 1))
        panel = "(c)" if column == 0 else "(d)"
        trajectory_ax.set_title(
            f"{path_name}笛卡尔轨迹", fontproperties=song_title, pad=8
        )
        trajectory_ax.text2D(
            -0.09,
            1.05,
            panel,
            transform=trajectory_ax.transAxes,
            fontproperties=times_panel,
            clip_on=False,
        )
        trajectory_ax.set_xlabel("X/mm", fontproperties=times, labelpad=8)
        trajectory_ax.set_ylabel("Y/mm", fontproperties=times, labelpad=10)
        trajectory_ax.zaxis.set_rotate_label(False)
        trajectory_ax.set_zlabel(
            "Z/mm", fontproperties=times, rotation=90, labelpad=14
        )
        if trajectory_handles is None:
            trajectory_handles, _ = trajectory_ax.get_legend_handles_labels()

    fig.legend(
        joint_handles,
        joint_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=6,
        frameon=False,
        prop=times_italic,
    )
    fig.legend(
        trajectory_handles,
        ["目标路径", "正运动学回代路径", "起点", "终点"],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=4,
        frameon=False,
        prop=song,
    )
    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.10,
        top=0.90,
        wspace=0.22,
        hspace=0.50,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=600, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
