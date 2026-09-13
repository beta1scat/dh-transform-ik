"""Plot the Kuka target-position distribution with thesis fonts."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea

try:
    from common import PROCESSED_DIR, PROJECT_ROOT
except ImportError:
    from experiments.common import PROCESSED_DIR, PROJECT_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PROCESSED_DIR / "ik_targets.csv")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT.parents[1]
        / "figures"
        / "chapter2"
        / "2_5_fig_workspace_samples.pdf",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    song = FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc")
    times = FontProperties(fname=r"C:\Windows\Fonts\times.ttf")
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "savefig.dpi": 600,
        }
    )
    positions = []
    with args.input.open("r", newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["robot"] == "kuka_kr50_r2500":
                positions.append(
                    [float(row["target_03"]), float(row["target_13"]), float(row["target_23"])]
                )
    points = np.asarray(positions, dtype=np.float64)
    if points.shape != (10_000, 3):
        raise ValueError(f"Expected 10000 Kuka targets, got {points.shape[0]}")

    figure = plt.figure(figsize=(6.8, 5.4))
    axis = figure.add_subplot(111, projection="3d")
    axis.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        s=3.0,
        alpha=0.58,
        color="#0072B2",
        edgecolors="none",
        rasterized=True,
    )
    title = HPacker(
        children=[
            TextArea("KUKA", textprops={"fontproperties": times, "fontsize": 12}),
            TextArea(
                "机器人目标位置分布",
                textprops={"fontproperties": song, "fontsize": 12},
            ),
        ],
        align="baseline",
        pad=0,
        sep=0,
    )
    axis.add_artist(
        AnchoredOffsetbox(
            loc="upper center",
            child=title,
            frameon=False,
            bbox_to_anchor=(0.5, 1.02),
            bbox_transform=axis.transAxes,
            borderpad=0,
        )
    )
    axis.set_xlabel(r"$X/\mathrm{mm}$")
    axis.set_ylabel(r"$Y/\mathrm{mm}$")
    axis.set_zlabel(r"$Z/\mathrm{mm}$")
    axis.grid(alpha=0.25, linewidth=0.5)
    figure.subplots_adjust(left=0.02, right=0.90, bottom=0.04, top=0.92)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=600, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
