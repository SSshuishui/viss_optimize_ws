"""Generate Figure 4 from the final STS sensitivity-sweep summaries."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.5,
        "lines.markersize": 4.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "text.usetex": False,
    }
)

WORKSPACE = Path(__file__).resolve().parents[1]
DATA_ROOT = WORKSPACE / "results/sensitivity"


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


segment_profiles = [
    row
    for row in read_csv(DATA_ROOT / "results_segment_sweep/profile_summary.csv")
    if "_rep/" in row["profile"]
]
segment_profiles.sort(key=lambda row: int(row["segments"]))

segments = np.array([int(row["segments"]) for row in segment_profiles])
segment_total = np.array([float(row["segment_wall_s"]) for row in segment_profiles])
plane_rmse = np.array(
    [float(row["mean_fit_rmse_wavelength"]) for row in segment_profiles]
)

segment_accuracy = {
    int(row["label"].removeprefix("segment_s")): row
    for row in read_csv(DATA_ROOT / "results_segment_sweep/accuracy_summary.csv")
}

grid_profiles = [
    row
    for row in read_csv(DATA_ROOT / "result_grid_sweep/profile_summary.csv")
    if "_rep/" in row["profile"]
]
grid_profile_by_scale = {
    float(row["grid_cell_scale_requested"]): row for row in grid_profiles
}
# The unit-scale run is shared with the 10-segment sweep.
grid_profile_by_scale[1.0] = next(
    row for row in segment_profiles if int(row["segments"]) == 10
)

grid_accuracy_by_scale = {
    float(row["label"].removeprefix("grid_")): row
    for row in read_csv(DATA_ROOT / "result_grid_sweep/accuracy_summary.csv")
}
grid_accuracy_by_scale[1.0] = segment_accuracy[10]

grid_scale = np.array(sorted(grid_profile_by_scale))
grid_recon = np.array(
    [float(grid_profile_by_scale[scale]["reconstruction_stage_wall_s"]) for scale in grid_scale]
)
grid_nrmse = np.array(
    [
        float(grid_accuracy_by_scale[scale]["affine_nrmse_by_reference_range"])
        for scale in grid_scale
    ]
)

if not np.array_equal(segments, np.array([5, 10, 20, 40])):
    raise ValueError(f"Unexpected segment sweep: {segments.tolist()}")
if not np.allclose(grid_scale, np.array([0.75, 1.00, 1.25, 1.50])):
    raise ValueError(f"Unexpected grid sweep: {grid_scale.tolist()}")

color_error = "#2B6F8E"
color_time = "#C15A37"
color_default = "#202020"
color_grid = "#D9D9D9"


def style_axis(ax):
    ax.grid(axis="y", color=color_grid, linewidth=0.55, alpha=0.75, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#4A4A4A")
    ax.spines["bottom"].set_color("#4A4A4A")
    ax.tick_params(colors="#333333")


def plot_sweep(ax, x, y, default_x, color, marker, ylabel, title):
    ax.plot(
        x,
        y,
        color=color,
        marker=marker,
        markerfacecolor="white",
        markeredgecolor=color,
        markeredgewidth=1.2,
        zorder=3,
    )
    default_index = int(np.flatnonzero(np.isclose(x, default_x))[0])
    ax.scatter(
        [x[default_index]],
        [y[default_index]],
        s=54,
        facecolor="white",
        edgecolor=color_default,
        linewidth=1.1,
        zorder=4,
    )
    ax.scatter(
        [x[default_index]],
        [y[default_index]],
        s=18,
        facecolor=color,
        edgecolor="none",
        zorder=5,
    )
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="semibold", pad=5)
    style_axis(ax)


fig, axes = plt.subplots(2, 2, figsize=(7.15, 4.25), sharex="col")

plot_sweep(
    axes[0, 0],
    segments,
    plane_rmse,
    10,
    color_error,
    "o",
    "Plane-fit RMSE (wavelength)",
    "(a) Segmentation residual",
)
plot_sweep(
    axes[1, 0],
    segments,
    segment_total,
    10,
    color_time,
    "s",
    "Compute-loop time (s)",
    "(c) Segmentation cost",
)
plot_sweep(
    axes[0, 1],
    grid_scale,
    100.0 * grid_nrmse,
    1.0,
    color_error,
    "o",
    "Affine NRMSE (%)",
    "(b) Grid consistency",
)
plot_sweep(
    axes[1, 1],
    grid_scale,
    grid_recon,
    1.0,
    color_time,
    "s",
    "Reconstruction time (s)",
    "(d) Grid cost",
)

axes[0, 0].set_ylim(0.039, 0.101)
axes[1, 0].set_ylim(54, 76)
axes[0, 1].set_ylim(0.07, 0.23)
axes[1, 1].set_ylim(11.5, 26.5)
axes[0, 1].yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

axes[1, 0].set_xlabel("Temporal segments")
axes[1, 1].set_xlabel("Grid-cell scale")
axes[1, 0].set_xticks(segments)
axes[1, 1].set_xticks(grid_scale)
axes[1, 1].xaxis.set_major_formatter(FormatStrFormatter("%.2f"))

fig.subplots_adjust(
    left=0.105,
    right=0.985,
    bottom=0.13,
    top=0.92,
    wspace=0.34,
    hspace=0.38,
)

outputs = [WORKSPACE / "figures/sensitivity_tradeoff.png"]
for output in outputs:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.03)
plt.close(fig)
print("saved:")
for output in outputs:
    print(f"  {output}")
