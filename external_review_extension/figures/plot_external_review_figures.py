from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EXTENSION_ROOT = Path(__file__).resolve().parent.parent
FIGURE_DIR = EXTENSION_ROOT / "figures"
MULTISCALE_ROOT = EXTENSION_ROOT / "gate_placement" / "results"
REG_ROOT = EXTENSION_ROOT / "registration" / "results"
QUALITY_ROOT = EXTENSION_ROOT / "dolp_quality" / "results"
TABLE_DIR = EXTENSION_ROOT / "source_tables"
STATS_DIR = EXTENSION_ROOT / "registration" / "results"

COLORS = {
    "direct": "#4477AA",
    "pfe_multiply": "#228833",
    "pfe_1x1": "#EE7733",
}
LABELS = {
    "direct": "Direct RGB+DoLP",
    "pfe_multiply": "PFE multiply",
    "pfe_1x1": "PFE 1x1",
}
MARKERS = {"direct": "o", "pfe_multiply": "s", "pfe_1x1": "^"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.5,
            "lines.markersize": 4.5,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def panel_label(ax, label: str) -> None:
    ax.text(
        -0.16,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top",
    )


def finish_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#D0D0D0", linewidth=0.5, alpha=0.7)


def export(fig, stem: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for extension in ("svg", "pdf"):
        fig.savefig(
            FIGURE_DIR / f"{stem}.{extension}",
            bbox_inches="tight",
            facecolor="white",
        )
    fig.savefig(
        FIGURE_DIR / f"{stem}.png",
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def plot_multiscale() -> None:
    summary = read_csv(TABLE_DIR / "table_S14_gate_placement.csv")
    runs = read_csv(MULTISCALE_ROOT / "multiscale_run_results.csv")
    variants = [row["Variant"] for row in summary]
    descriptions = [row["Description"] for row in summary]
    x = np.arange(len(variants))
    iou_mean = np.array([float(row["Test IoU mean"]) for row in summary])
    iou_sd = np.array([float(row["Test IoU SD"]) for row in summary])
    mae_mean = np.array([float(row["Image-averaged MAE mean"]) for row in summary])
    mae_sd = np.array([float(row["Image-averaged MAE SD"]) for row in summary])
    gflops = np.array([float(row["Effective GFLOPs"]) for row in summary])

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.25))
    bar_colors = ["#BBBBBB", "#4477AA", "#66CCEE", "#AA3377", "#CCBB44", "#228833", "#EE7733"]
    variant_colors = bar_colors[: len(variants)]
    axes[0].bar(x, iou_mean, yerr=iou_sd, color=variant_colors, edgecolor="#333333", linewidth=0.5, capsize=2)
    axes[1].bar(x, mae_mean, yerr=mae_sd, color=variant_colors, edgecolor="#333333", linewidth=0.5, capsize=2)
    rng = np.random.default_rng(20260723)
    for index, variant in enumerate(variants):
        variant_rows = [row for row in runs if row["Variant"] == variant]
        jitter = rng.uniform(-0.09, 0.09, size=len(variant_rows))
        axes[0].scatter(
            index + jitter,
            [float(row["Test IoU"]) for row in variant_rows],
            s=13,
            color="#111111",
            edgecolor="white",
            linewidth=0.35,
            zorder=3,
        )
        axes[1].scatter(
            index + jitter,
            [float(row["Image-averaged MAE"]) for row in variant_rows],
            s=13,
            color="#111111",
            edgecolor="white",
            linewidth=0.35,
            zorder=3,
        )
    axes[2].scatter(
        gflops,
        iou_mean,
        s=38,
        c=variant_colors,
        edgecolor="#222222",
        linewidth=0.6,
    )
    for index, variant in enumerate(variants):
        axes[2].annotate(
            variant,
            (gflops[index], iou_mean[index]),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=7,
        )

    for ax in axes[:2]:
        ax.set_xticks(x, [f"{variant}\n{description}" for variant, description in zip(variants, descriptions)], rotation=35, ha="right")
    axes[0].set_ylabel("Test IoU")
    axes[1].set_ylabel("Image-averaged MAE")
    axes[2].set_xlabel("Effective GFLOPs")
    axes[2].set_ylabel("Test IoU mean")
    for ax, label in zip(axes, ("a", "b", "c")):
        finish_axes(ax)
        panel_label(ax, label)
    fig.tight_layout(w_pad=1.7)
    export(fig, "Figure_Multiscale_Ablation")


def plot_registration() -> None:
    translation = read_csv(REG_ROOT / "translation_summary.csv")
    rotation = read_csv(STATS_DIR / "rotation_absolute_angle_summary.csv")
    tolerance = read_csv(TABLE_DIR / "table_S15_registration_sensitivity.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.7))
    axes = axes.ravel()

    for method in COLORS:
        rows = sorted(
            [row for row in translation if row["method"] == method],
            key=lambda row: int(row["translation_pixels"]),
        )
        x = np.array([int(row["translation_pixels"]) for row in rows])
        iou = np.array([float(row["iou_mean_over_directions_and_seeds"]) for row in rows])
        iou_sd = np.array([float(row["iou_seed_sd"]) for row in rows])
        mae = np.array([float(row["mae_mean_over_directions_and_seeds"]) for row in rows])
        mae_sd = np.array([float(row["mae_seed_sd"]) for row in rows])
        axes[0].errorbar(x, iou, yerr=iou_sd, color=COLORS[method], marker=MARKERS[method], capsize=2, label=LABELS[method])
        axes[1].errorbar(x, mae, yerr=mae_sd, color=COLORS[method], marker=MARKERS[method], capsize=2, label=LABELS[method])

        rotation_rows = sorted(
            [row for row in rotation if row["method"] == method],
            key=lambda row: float(row["absolute_rotation_degrees"]),
        )
        angle = np.array([float(row["absolute_rotation_degrees"]) for row in rotation_rows])
        rot_iou = np.array([float(row["iou_mean_over_signs_and_seeds"]) for row in rotation_rows])
        rot_sd = np.array([float(row["iou_seed_sd"]) for row in rotation_rows])
        axes[2].errorbar(angle, rot_iou, yerr=rot_sd, color=COLORS[method], marker=MARKERS[method], capsize=2, label=LABELS[method])

    axes[0].set(xlabel="DoLP translation (pixels)", ylabel="Test IoU")
    axes[1].set(xlabel="DoLP translation (pixels)", ylabel="Image-averaged MAE")
    axes[2].set(xlabel="Absolute DoLP rotation (degrees)", ylabel="Test IoU")
    axes[0].legend(frameon=False, ncol=1)

    methods = list(COLORS)
    t2_shift = [
        float(
            next(
                row["maximum_level_meeting_criterion"]
                for row in tolerance
                if row["method"] == method
                and row["perturbation"] == "translation"
                and row["threshold"] == "T2_absolute_0p01"
            )
        )
        for method in methods
    ]
    t2_rotation = [
        float(
            next(
                row["maximum_level_meeting_criterion"]
                for row in tolerance
                if row["method"] == method
                and row["perturbation"] == "rotation"
                and row["threshold"] == "T2_absolute_0p01"
            )
        )
        for method in methods
    ]
    positions = np.arange(len(methods))
    width = 0.34
    axes[3].bar(positions - width / 2, t2_shift, width, color="#4477AA", label="Translation (px)")
    axes[3].bar(positions + width / 2, t2_rotation, width, color="#EE7733", label="Rotation (deg)")
    axes[3].set_xticks(positions, [LABELS[method] for method in methods], rotation=20, ha="right")
    axes[3].set_ylabel("Maximum T2 level")
    axes[3].legend(frameon=False)
    for ax, label in zip(axes, ("a", "b", "c", "d")):
        finish_axes(ax)
        panel_label(ax, label)
    fig.tight_layout(h_pad=2.0, w_pad=1.6)
    export(fig, "Figure_Registration_Sensitivity")


def plot_quality() -> None:
    rows = read_csv(TABLE_DIR / "table_dolp_quality_summary_45_rows.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.7))
    axes = axes.ravel()
    kinds = [
        ("blur", "Gaussian blur sigma", "Test IoU"),
        ("noise", "Gaussian noise std", "Test IoU"),
        ("quantization", "Quantization bits", "Test IoU"),
    ]
    for axis, (kind, xlabel, ylabel) in zip(axes[:3], kinds):
        for method in COLORS:
            subset = sorted(
                [
                    row
                    for row in rows
                    if row["perturbation_type"] == kind and row["method"] == method
                ],
                key=lambda row: float(row["level"]),
            )
            x = np.array([float(row["level"]) for row in subset])
            y = np.array([float(row["iou_mean"]) for row in subset])
            yerr = np.array([float(row["iou_seed_sd"]) for row in subset])
            axis.errorbar(
                x,
                y,
                yerr=yerr,
                color=COLORS[method],
                marker=MARKERS[method],
                capsize=2,
                label=LABELS[method],
            )
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
    axes[0].legend(frameon=False)

    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        level = float(row["level"])
        kind = row["perturbation_type"]
        is_clean = (kind in {"blur", "noise"} and level == 0.0) or (
            kind == "quantization" and level == 8.0
        )
        if not is_clean:
            grouped[row["method"]].append(float(row["absolute_iou_drop_mean"]))
    methods = list(COLORS)
    values = [float(np.mean(grouped[method])) for method in methods]
    axes[3].bar(
        np.arange(len(methods)),
        values,
        color=[COLORS[method] for method in methods],
        edgecolor="#333333",
        linewidth=0.5,
    )
    axes[3].axhline(0, color="#333333", linewidth=0.7)
    axes[3].set_xticks(
        np.arange(len(methods)),
        [LABELS[method] for method in methods],
        rotation=20,
        ha="right",
    )
    axes[3].set_ylabel("Mean absolute IoU drop")
    axes[3].set_xlabel("All non-clean quality levels")

    for ax, label in zip(axes, ("a", "b", "c", "d")):
        finish_axes(ax)
        panel_label(ax, label)
    fig.tight_layout(h_pad=2.0, w_pad=1.6)
    export(fig, "Figure_DoLP_Quality_Sensitivity")


def main() -> None:
    setup_style()
    plot_multiscale()
    plot_registration()
    plot_quality()
    print(f"Figures saved to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
