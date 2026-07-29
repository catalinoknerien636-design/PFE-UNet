from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
TABLE_PATH = ROOT / "source_tables" / "table_S14_gate_placement.csv"
RUN_PATH = ROOT / "gate_placement" / "results" / "multiscale_run_results.csv"
OUTPUT_DIR = ROOT / "figures"
OUTPUT_STEM = "Figure_Gate_Placement_Ablation"

VARIANT_LABELS = {
    "M0": "M0\nNo gate\n(n=1)",
    "M1": "M1\nL1\n(n=3)",
    "M3": "M3\nL5\n(n=3)",
    "M4": "M4\nL1-L3\n(n=3)",
    "M5": "M5\nL3-L5\n(n=3)",
    "M6": "M6\nL1-L5\n(n=5)",
}

COLORS = {
    "M0": "#8C8C8C",
    "M1": "#4477AA",
    "M3": "#66A61E",
    "M4": "#AA3377",
    "M5": "#CCBB44",
    "M6": "#228833",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.2,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def finish_axis(ax: plt.Axes, panel: str, ylabel: str) -> None:
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#D5D5D5", linewidth=0.5, alpha=0.8)
    ax.set_axisbelow(True)
    ax.text(
        0.0,
        1.04,
        panel,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top",
    )


def draw_metric(
    ax: plt.Axes,
    variants: list[str],
    summary: dict[str, dict[str, str]],
    runs: list[dict[str, str]],
    mean_key: str,
    sd_key: str,
    run_key: str,
) -> None:
    x = np.arange(len(variants))
    rng = np.random.default_rng(20260728)

    for index, variant in enumerate(variants):
        mean = float(summary[variant][mean_key])
        n = int(summary[variant]["N seeds"])
        color = COLORS[variant]

        if n > 1:
            sd = float(summary[variant][sd_key])
            ax.errorbar(
                index,
                mean,
                yerr=sd,
                fmt="o",
                markersize=6,
                color=color,
                markeredgecolor="#222222",
                markeredgewidth=0.5,
                ecolor=color,
                elinewidth=1.25,
                capsize=3,
                capthick=1.0,
                zorder=4,
            )
        else:
            ax.scatter(
                index,
                mean,
                s=34,
                color=color,
                edgecolor="#222222",
                linewidth=0.5,
                zorder=4,
            )

        variant_rows = [row for row in runs if row["Variant"] == variant]
        jitter = rng.uniform(-0.10, 0.10, size=len(variant_rows))
        ax.scatter(
            index + jitter,
            [float(row[run_key]) for row in variant_rows],
            s=14,
            color="#202020",
            alpha=0.8,
            edgecolor="white",
            linewidth=0.35,
            zorder=5,
        )

    ax.set_xticks(x, [VARIANT_LABELS[variant] for variant in variants])
    ax.tick_params(axis="x", length=0, pad=5)
    ax.set_xlim(-0.45, len(variants) - 0.55)


def main() -> None:
    setup_style()
    summary_rows = read_csv(TABLE_PATH)
    run_rows = read_csv(RUN_PATH)
    summary = {row["Variant"]: row for row in summary_rows}
    variants = ["M0", "M1", "M3", "M4", "M5", "M6"]

    fig, axes = plt.subplots(1, 3, figsize=(7.35, 2.75))

    draw_metric(
        axes[0],
        variants,
        summary,
        run_rows,
        "Best validation IoU mean",
        "Best validation IoU SD",
        "Best validation IoU",
    )
    draw_metric(
        axes[1],
        variants,
        summary,
        run_rows,
        "Test IoU mean",
        "Test IoU SD",
        "Test IoU",
    )
    draw_metric(
        axes[2],
        variants,
        summary,
        run_rows,
        "Image-averaged MAE mean",
        "Image-averaged MAE SD",
        "Image-averaged MAE",
    )

    finish_axis(axes[0], "a", "Validation IoU")
    finish_axis(axes[1], "b", "Test IoU")
    finish_axis(axes[2], "c", "Image-averaged MAE")
    axes[0].set_ylim(0.35, 0.46)
    axes[1].set_ylim(0.36, 0.52)
    axes[2].set_ylim(0.028, 0.044)

    fig.tight_layout(w_pad=1.5)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for extension in ("svg", "pdf"):
        fig.savefig(
            OUTPUT_DIR / f"{OUTPUT_STEM}.{extension}",
            bbox_inches="tight",
            facecolor="white",
        )
    fig.savefig(
        OUTPUT_DIR / f"{OUTPUT_STEM}.png",
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        OUTPUT_DIR / f"{OUTPUT_STEM}.tiff",
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
