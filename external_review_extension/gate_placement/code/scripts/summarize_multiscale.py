from __future__ import annotations

import csv
import json
import os
import statistics
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
EXTENSION_ROOT = EXPERIMENT_ROOT.parent
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
PROJECT_ROOT = Path(os.environ.get("PFE_SOURCE_PROJECT_ROOT", REPOSITORY_ROOT)).resolve()
EVIDENCE_ROOT = Path(os.environ.get("PFE_EVIDENCE_ROOT", REPOSITORY_ROOT)).resolve()
TABLE_DIR = EXTENSION_ROOT / "source_tables"
STATS_DIR = EXPERIMENT_ROOT / "results"
REPORT_DIR = EXPERIMENT_ROOT / "reports"

VARIANTS = {
    "M0": ("No active gate", ()),
    "M1": ("L1 only", (1,)),
    "M3": ("L5 only", (5,)),
    "M4": ("Shallow", (1, 2, 3)),
    "M5": ("Deep", (3, 4, 5)),
    "M6": ("All five levels", (1, 2, 3, 4, 5)),
}
PARTIAL_VARIANTS = ("M1", "M3", "M4", "M5")
CORE_SEEDS = (42, 2026, 3407)
ALL_SEEDS = (42, 2026, 3407, 7819, 12011)
VERIFIED_M0_SEEDS = (42,)

M6_SOURCES = {
    42: {
        "run": "grouped_pfe_unet_multiply_directml_50ep",
        "history_metrics": PROJECT_ROOT / "outputs" / "metrics",
        "test_metrics": PROJECT_ROOT / "outputs" / "metrics",
        "cod_metrics": EVIDENCE_ROOT / "metrics",
        "checkpoint": PROJECT_ROOT / "checkpoints",
        "predictions": PROJECT_ROOT / "outputs" / "predictions",
        "config": PROJECT_ROOT / "outputs" / "metrics",
    },
    2026: {
        "run": "grouped_pfe_unet_multiply_directml_50ep_seed2026",
        "history_metrics": PROJECT_ROOT / "outputs" / "metrics",
        "test_metrics": PROJECT_ROOT / "outputs" / "metrics",
        "cod_metrics": EVIDENCE_ROOT / "metrics",
        "checkpoint": PROJECT_ROOT / "checkpoints",
        "predictions": PROJECT_ROOT / "outputs" / "predictions",
        "config": PROJECT_ROOT / "outputs" / "metrics",
    },
    3407: {
        "run": "grouped_pfe_unet_multiply_directml_50ep_seed3407",
        "history_metrics": PROJECT_ROOT / "outputs" / "metrics",
        "test_metrics": PROJECT_ROOT / "outputs" / "metrics",
        "cod_metrics": EVIDENCE_ROOT / "metrics",
        "checkpoint": PROJECT_ROOT / "checkpoints",
        "predictions": PROJECT_ROOT / "outputs" / "predictions",
        "config": PROJECT_ROOT / "outputs" / "metrics",
    },
    7819: {
        "run": "ext_pfe_dolp_multiply_seed7819",
        "history_metrics": EVIDENCE_ROOT / "metrics",
        "test_metrics": EVIDENCE_ROOT / "metrics",
        "cod_metrics": EVIDENCE_ROOT / "metrics",
        "checkpoint": EVIDENCE_ROOT / "checkpoints",
        "predictions": EVIDENCE_ROOT / "predictions",
        "config": EVIDENCE_ROOT / "metrics",
    },
    12011: {
        "run": "ext_pfe_dolp_multiply_seed12011",
        "history_metrics": EVIDENCE_ROOT / "metrics",
        "test_metrics": EVIDENCE_ROOT / "metrics",
        "cod_metrics": EVIDENCE_ROOT / "metrics",
        "checkpoint": EVIDENCE_ROOT / "checkpoints",
        "predictions": EVIDENCE_ROOT / "predictions",
        "config": EVIDENCE_ROOT / "metrics",
    },
}


def local_run_name(variant: str, seed: int) -> str:
    levels = "_".join(f"L{level}" for level in VARIANTS[variant][1]) or "none"
    return f"multiscale_{variant}_{levels}_seed{seed}"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_one(path: Path) -> dict[str, str]:
    rows = read_rows(path)
    if len(rows) != 1:
        raise ValueError(f"Expected one row in {path}, found {len(rows)}")
    return rows[0]


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def best_history(path: Path) -> tuple[int, float]:
    rows = read_rows(path)
    if len(rows) != 50:
        raise ValueError(f"Expected 50 epochs in {path}, found {len(rows)}")
    best = max(rows, key=lambda row: float(row["val_iou"]))
    return int(best["epoch"]), float(best["val_iou"])


def complexity_map() -> dict[str, dict[str, str]]:
    rows = read_rows(EXPERIMENT_ROOT / "model_complexity.csv")
    return {row["variant"]: row for row in rows}


def local_paths(variant: str, seed: int) -> dict[str, Path | str]:
    name = local_run_name(variant, seed)
    metrics = EXPERIMENT_ROOT / "metrics" / name
    checkpoint = (
        PROJECT_ROOT
        / "checkpoints"
        / "grouped_pfe_unet_none_directml_50ep"
        / "best.pt"
        if variant == "M0" and seed == 42
        else EXPERIMENT_ROOT / "checkpoints" / name / "best.pt"
    )
    return {
        "run": name,
        "history": metrics / "history.csv",
        "test": metrics / "test_metrics.csv",
        "cod": metrics / "cod_metrics.csv",
        "config": metrics / "config.json",
        "checkpoint": checkpoint,
        "predictions": EXPERIMENT_ROOT / "predictions" / name,
        "per_image": metrics / "cod_per_sample.csv",
    }


def m6_paths(seed: int) -> dict[str, Path | str]:
    source = M6_SOURCES[seed]
    name = str(source["run"])
    return {
        "run": name,
        "history": Path(source["history_metrics"]) / name / "history.csv",
        "test": Path(source["test_metrics"]) / name / "test_metrics.csv",
        "cod": Path(source["cod_metrics"]) / name / "cod_metrics.csv",
        "config": Path(source["config"]) / name / "config.json",
        "checkpoint": Path(source["checkpoint"]) / name / "best.pt",
        "predictions": Path(source["predictions"]) / name,
        "per_image": Path(source["cod_metrics"]) / name / "cod_per_sample.csv",
    }


def collect_one(variant: str, seed: int, complexity: dict[str, dict[str, str]]) -> dict:
    paths = m6_paths(seed) if variant == "M6" else local_paths(variant, seed)
    required = ("history", "test", "cod", "config", "checkpoint", "predictions", "per_image")
    missing = [key for key in required if not Path(paths[key]).exists()]
    if missing:
        raise FileNotFoundError(f"{variant} seed {seed} missing {missing}")
    if len(list(Path(paths["predictions"]).glob("*.png"))) != 230:
        raise ValueError(f"{variant} seed {seed} does not have 230 predictions")
    per_image_rows = read_rows(Path(paths["per_image"]))
    if len(per_image_rows) != 230:
        raise ValueError(f"{variant} seed {seed} does not have 230 per-image metric rows")

    best_epoch, best_val_iou = best_history(Path(paths["history"]))
    test = read_one(Path(paths["test"]))
    cod = read_one(Path(paths["cod"]))
    if per_image_rows and "mae_image" in per_image_rows[0]:
        image_averaged_mae = statistics.mean(
            float(row["mae_image"]) for row in per_image_rows
        )
    elif "mae_image_mean" in test:
        image_averaged_mae = float(test["mae_image_mean"])
    else:
        image_averaged_mae = float(test["mae"])
    levels = VARIANTS[variant][1]
    row = {
        "Method": "PFE-UNet",
        "Variant": variant,
        "Description": VARIANTS[variant][0],
        "Active gate levels": ",".join(f"L{level}" for level in levels),
        "Seed": seed,
        "Best epoch": best_epoch,
        "Best validation IoU": best_val_iou,
        "Test IoU": float(test["iou"]),
        "F1": float(test["f1"]),
        "Precision": float(test["precision"]),
        "Recall": float(test["recall"]),
        "Image-averaged MAE": image_averaged_mae,
        "S-measure": float(cod["s_measure"]),
        "Mean E-measure": float(cod["e_measure_mean"]),
        "Weighted F-measure": float(cod["weighted_f_measure"]),
        "Params": int(float(complexity[variant]["total_params"])),
        "Effective GFLOPs": float(complexity[variant]["conv_gflops"]),
        "Checkpoint": str(paths["checkpoint"]),
        "Config": str(paths["config"]),
        "Prediction path": str(paths["predictions"]),
        "Per-image metrics": str(paths["per_image"]),
        "Run name": str(paths["run"]),
        "Verified": "yes",
    }
    return row


def mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def summary_rows(run_rows: list[dict]) -> list[dict]:
    summaries = []
    for variant in VARIANTS:
        rows = [row for row in run_rows if row["Variant"] == variant]
        iou_mean, iou_sd = mean_sd([float(row["Test IoU"]) for row in rows])
        mae_mean, mae_sd = mean_sd([float(row["Image-averaged MAE"]) for row in rows])
        val_mean, val_sd = mean_sd([float(row["Best validation IoU"]) for row in rows])
        summaries.append(
            {
                "Variant": variant,
                "Description": VARIANTS[variant][0],
                "Active gate levels": ",".join(
                    f"L{level}" for level in VARIANTS[variant][1]
                ),
                "Seeds": ";".join(str(row["Seed"]) for row in rows),
                "N seeds": len(rows),
                "Seed-count basis": (
                    "existing reused results"
                    if variant in {"M0", "M6"}
                    else "three matched new-training seeds"
                ),
                "Best validation IoU mean": val_mean,
                "Best validation IoU SD": val_sd,
                "Test IoU mean": iou_mean,
                "Test IoU SD": iou_sd,
                "Image-averaged MAE mean": mae_mean,
                "Image-averaged MAE SD": mae_sd,
                "Params": rows[0]["Params"],
                "Effective GFLOPs": rows[0]["Effective GFLOPs"],
            }
        )
    return summaries


def matched_rows(run_rows: list[dict]) -> list[dict]:
    lookup = {(row["Variant"], int(row["Seed"])): row for row in run_rows}
    rows = []
    for variant in ("M0", *PARTIAL_VARIANTS):
        common_seeds = sorted(
            set(seed for v, seed in lookup if v == variant)
            & set(seed for v, seed in lookup if v == "M6")
        )
        for seed in common_seeds:
            candidate = lookup[(variant, seed)]
            all_five = lookup[("M6", seed)]
            rows.append(
                {
                    "Comparison": f"M6 - {variant}",
                    "Seed": seed,
                    "IoU difference": float(all_five["Test IoU"])
                    - float(candidate["Test IoU"]),
                    "MAE difference": float(all_five["Image-averaged MAE"])
                    - float(candidate["Image-averaged MAE"]),
                }
            )
    return rows


def update_traceability(run_rows: list[dict]) -> None:
    path = REPORT_DIR / "RESULT_TRACEABILITY.csv"
    existing = read_rows(path) if path.exists() else []
    fieldnames = (
        list(existing[0])
        if existing
        else [
            "experiment",
            "method",
            "variant",
            "active_gate_levels",
            "perturbation_type",
            "perturbation_level",
            "direction",
            "seed",
            "noise_realization",
            "best_epoch",
            "validation_iou",
            "test_iou",
            "f1",
            "precision",
            "recall",
            "mae",
            "config_path",
            "checkpoint_path",
            "prediction_path",
            "metrics_path",
            "script",
            "verified",
            "notes",
        ]
    )
    retained = [row for row in existing if row.get("experiment") != "multiscale_gate_ablation"]
    for row in run_rows:
        retained.append(
            {
                "experiment": "multiscale_gate_ablation",
                "method": row["Method"],
                "variant": row["Variant"],
                "active_gate_levels": row["Active gate levels"],
                "perturbation_type": "",
                "perturbation_level": "",
                "direction": "",
                "seed": row["Seed"],
                "noise_realization": "",
                "best_epoch": row["Best epoch"],
                "validation_iou": row["Best validation IoU"],
                "test_iou": row["Test IoU"],
                "f1": row["F1"],
                "precision": row["Precision"],
                "recall": row["Recall"],
                "mae": row["Image-averaged MAE"],
                "config_path": row["Config"],
                "checkpoint_path": row["Checkpoint"],
                "prediction_path": row["Prediction path"],
                "metrics_path": row["Per-image metrics"],
                "script": str(Path(__file__).resolve()),
                "verified": "yes",
                "notes": "Checkpoint selected by validation IoU; test not used for model or variant selection.",
            }
        )
    write_csv(path, fieldnames, retained)


def write_report(summaries: list[dict], matched: list[dict], selected: str) -> None:
    lookup = {row["Variant"]: row for row in summaries}
    all_five = lookup["M6"]
    best_partial = lookup[selected]
    no_gate = lookup["M0"]

    if float(all_five["Test IoU mean"]) > float(best_partial["Test IoU mean"]):
        status = "A if M6 is also consistently favorable in matched seeds; otherwise B"
    elif abs(float(all_five["Test IoU mean"]) - float(best_partial["Test IoU mean"])) <= 0.005:
        status = "B: multi-scale is useful but partially redundant"
    else:
        status = "C/D: the Multi-Scale title and placement claim require reframing"

    report = f"""# 多尺度门控位置消融报告

## 预注册与选择规则

M0-M6 的组合和前三个匹配种子在训练前固定。最佳部分尺度变体只依据 M1-M5 三个种子的 best validation IoU 均值选择，未读取测试指标。依据该规则选中 `{selected}`，随后 `{selected}` 与 M0 扩展到种子 7819 和 12011；M6 复用既有五种子正式结果。

## 核心汇总

- M0 无门控：Test IoU = {float(no_gate['Test IoU mean']):.4f} +/- {float(no_gate['Test IoU SD']):.4f}，image-averaged MAE = {float(no_gate['Image-averaged MAE mean']):.4f} +/- {float(no_gate['Image-averaged MAE SD']):.4f}。
- 最佳部分尺度 `{selected}`：Test IoU = {float(best_partial['Test IoU mean']):.4f} +/- {float(best_partial['Test IoU SD']):.4f}，image-averaged MAE = {float(best_partial['Image-averaged MAE mean']):.4f} +/- {float(best_partial['Image-averaged MAE SD']):.4f}。
- M6 五尺度：Test IoU = {float(all_five['Test IoU mean']):.4f} +/- {float(all_five['Test IoU SD']):.4f}，image-averaged MAE = {float(all_five['Image-averaged MAE mean']):.4f} +/- {float(all_five['Image-averaged MAE SD']):.4f}。

## 解释状态

自动描述性判定：**{status}**。

最终正文措辞必须结合逐 seed 差异，不得把五尺度均值优势写成“每一层均不可缺少”。单层、浅层、深层、五尺度的边界与语义解释只能作为与结果一致的机制性观察，不应由大量 image-level p 值制造显著性。

## 结果文件

- `04_multiscale_ablation/multiscale_run_results.csv`
- `07_tables/table_multiscale_gate_ablation.csv`
- `09_statistics/multiscale_matched_seed_differences.csv`
- `04_multiscale_ablation/origin_multiscale_iou.csv`
- `04_multiscale_ablation/origin_multiscale_mae.csv`
"""
    (REPORT_DIR / "MULTISCALE_GATE_ABLATION_REPORT_CN.md").write_text(
        report, encoding="utf-8"
    )


def write_scoped_report(summaries: list[dict], selected: str) -> None:
    lookup = {row["Variant"]: row for row in summaries}
    all_five = lookup["M6"]
    best_partial = lookup[selected]
    no_gate = lookup["M0"]

    if float(all_five["Test IoU mean"]) > float(best_partial["Test IoU mean"]):
        status = "A if M6 is also consistently favorable in matched seeds; otherwise B"
    elif (
        abs(float(all_five["Test IoU mean"]) - float(best_partial["Test IoU mean"]))
        <= 0.005
    ):
        status = "B: multi-scale is useful but partially redundant"
    else:
        status = "C/D: the multi-scale title and placement claim require reframing"

    report = f"""# 多尺度门控消融报告

## 冻结范围

本轮新增训练严格限定为 4 种 partial variant（M1、M3、M4、M5）乘以 3 个匹配种子（42、2026、3407），共 12 个 run。M0 no-gate 与 M6 all-five 只允许复用已有正式结果，不参与新增训练。L3 only、L1+L5、L2+L3+L4、leave-one-level-out 以及验证后扩种子均未运行。

## 统计口径

- M6 all-five：复用 5 个正式种子。
- M1、M3、M4、M5：各 3 个匹配种子，以 mean +/- SD 和结构趋势为主。
- M0 no-gate：当前本机仅核验到 {int(no_gate['N seeds'])} 个可追溯正式种子。不得在找到其余正式来源前写成 5-seed 统计，也不得用重训结果替代作者指定的既有结果。
- 不把不同种子数量当作完全对等的大样本统计，不据此宣称统计优越性。

## 描述性结果

- M0 no-gate：Test IoU = {float(no_gate['Test IoU mean']):.4f} +/- {float(no_gate['Test IoU SD']):.4f}，image-averaged MAE = {float(no_gate['Image-averaged MAE mean']):.4f} +/- {float(no_gate['Image-averaged MAE SD']):.4f}。
- 最佳三种子 partial variant `{selected}`：Test IoU = {float(best_partial['Test IoU mean']):.4f} +/- {float(best_partial['Test IoU SD']):.4f}，image-averaged MAE = {float(best_partial['Image-averaged MAE mean']):.4f} +/- {float(best_partial['Image-averaged MAE SD']):.4f}。
- M6 all-five：Test IoU = {float(all_five['Test IoU mean']):.4f} +/- {float(all_five['Test IoU SD']):.4f}，image-averaged MAE = {float(all_five['Image-averaged MAE mean']):.4f} +/- {float(all_five['Image-averaged MAE SD']):.4f}。

## 解释边界

自动描述性判断：**{status}**。

最终正文必须基于 matched three-seed 的结构趋势解释 partial variants，并单独标注 M0/M6 的复用来源与实际 N。不得把 all-five 的均值优势写成“每一层均不可缺少”，也不得用大量 image-level p 值制造显著性。
"""
    (REPORT_DIR / "MULTISCALE_GATE_ABLATION_REPORT_CN.md").write_text(
        report, encoding="utf-8"
    )


def main() -> None:
    selection = json.loads(
        (EXPERIMENT_ROOT / "configs" / "partial_variant_selection.json").read_text(
            encoding="utf-8"
        )
    )
    selected = str(selection["selected_variant"])
    complexity = complexity_map()
    run_rows = []
    for seed in VERIFIED_M0_SEEDS:
        run_rows.append(collect_one("M0", seed, complexity))
    for variant in PARTIAL_VARIANTS:
        for seed in CORE_SEEDS:
            run_rows.append(collect_one(variant, seed, complexity))
    for seed in ALL_SEEDS:
        run_rows.append(collect_one("M6", seed, complexity))

    run_fields = list(run_rows[0])
    write_csv(EXPERIMENT_ROOT / "multiscale_run_results.csv", run_fields, run_rows)
    summaries = summary_rows(run_rows)
    write_csv(
        TABLE_DIR / "table_multiscale_gate_ablation.csv",
        list(summaries[0]),
        summaries,
    )
    matched = matched_rows(run_rows)
    write_csv(
        STATS_DIR / "multiscale_matched_seed_differences.csv",
        list(matched[0]),
        matched,
    )

    iou_origin = [
        {
            "Variant": row["Variant"],
            "Description": row["Description"],
            "Active gate levels": row["Active gate levels"],
            "Seed": row["Seed"],
            "Test IoU": row["Test IoU"],
        }
        for row in run_rows
    ]
    mae_origin = [
        {
            "Variant": row["Variant"],
            "Description": row["Description"],
            "Active gate levels": row["Active gate levels"],
            "Seed": row["Seed"],
            "Image-averaged MAE": row["Image-averaged MAE"],
        }
        for row in run_rows
    ]
    write_csv(
        EXPERIMENT_ROOT / "origin_multiscale_iou.csv",
        list(iou_origin[0]),
        iou_origin,
    )
    write_csv(
        EXPERIMENT_ROOT / "origin_multiscale_mae.csv",
        list(mae_origin[0]),
        mae_origin,
    )
    update_traceability(run_rows)
    write_scoped_report(summaries, selected)
    print(f"Summarized {len(run_rows)} verified multiscale runs; selected={selected}.")


if __name__ == "__main__":
    main()
