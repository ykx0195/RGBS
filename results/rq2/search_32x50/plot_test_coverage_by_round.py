"""Plot Test coverage from saved 32-round coordinate-beam traces."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


HERE = Path(__file__).resolve().parent
data = json.loads((HERE / "test_coverage_by_round.json").read_text(encoding="utf-8"))

series = [
    ("abcd", "ABCD", "#6750a4"),
    ("banking77", "Banking77", "#2066a5"),
    ("dogo", "MultiDoGO", "#16816d"),
]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 200,
})
fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
fig.subplots_adjust(left=0.09, right=0.97, bottom=0.10, top=0.84, hspace=0.36)

with (HERE / "test_coverage_by_round.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow(["dataset", "round", "test_covered", "test_total", "test_coverage"])
    for key, label, color in series:
        run = data[key]
        points = run["points"]
        for point in points:
            writer.writerow([key, point["round"], point["covered"],
                             point["total"], f'{point["coverage"]:.10f}'])

for ax, (key, label, color) in zip(axes, series):
    run = data[key]
    points = run["points"]
    rounds = [point["round"] for point in points]
    values = [point["coverage"] for point in points]
    final = run["actual_32_step_final"]
    ax.plot(rounds, values, color=color, linewidth=2.2, marker="o", markersize=3,
            label="Saved search trajectory")
    ax.scatter([26, 28], [values[26], values[28]], color=color, s=43,
               edgecolors="white", linewidths=0.8, zorder=5)
    ax.scatter([32], [final["coverage"]], color="#de6431", marker="D", s=57,
               edgecolors="white", linewidths=0.8, zorder=6,
               label="Final 32-round global selection")
    ax.set_title(f"{label}  (Test n={points[0]['total']})", loc="left", weight="bold")
    ax.set_ylabel("Test coverage")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=1))
    ax.grid(axis="y", color="#d9e0e7", alpha=0.8)
    ax.set_xlim(-0.5, 33.5)
    low = min(values + [final["coverage"]])
    high = max(values + [final["coverage"]])
    pad = max(0.004, (high - low) * 0.12)
    ax.set_ylim(low - pad, high + pad)
    ax.annotate(f"{final['covered']}/{final['total']}",
                xy=(32, final["coverage"]), xytext=(-70, 12), textcoords="offset points",
                color="#aa421d", fontsize=9)

axes[-1].set_xticks(list(range(0, 33, 4)))
axes[-1].set_xlabel("Search rounds per sentence position (beam width 50)")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.93),
           ncol=2, frameon=False)
fig.suptitle("Test coverage by search round", fontsize=14, weight="bold", y=0.985)
fig.text(0.5, 0.035,
         "Trajectory cutoffs retain the 32-round predecessor choices; they are not independent reruns.",
         ha="center", va="top", fontsize=9, color="#55616c")
fig.savefig(HERE / "test_coverage_by_round.png", bbox_inches="tight")
fig.savefig(HERE / "test_coverage_by_round.svg", bbox_inches="tight")
print(HERE / "test_coverage_by_round.png")
