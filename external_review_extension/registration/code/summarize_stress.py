from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path


EXTENSION_ROOT = Path(__file__).resolve().parents[2]
REG_ROOT = EXTENSION_ROOT / "registration"
QUALITY_ROOT = EXTENSION_ROOT / "dolp_quality"
TABLE_DIR = EXTENSION_ROOT / "source_tables"
STATS_DIR = REG_ROOT / "results"
REPORT_DIR = EXTENSION_ROOT / "reports"
METHODS = ("direct", "pfe_multiply", "pfe_1x1")
SEEDS = (42, 2026, 3407, 7819, 12011)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def load_suites() -> tuple[list[dict], list[dict]]:
    registration: list[dict] = []
    quality: list[dict] = []
    for method in METHODS:
        for seed in SEEDS:
            reg_path = REG_ROOT / "runs" / f"{method}_seed{seed}" / "registration_suite_summary.csv"
            quality_path = (
                QUALITY_ROOT / "runs" / f"{method}_seed{seed}" / "quality_suite_summary.csv"
            )
            reg_rows = read_csv(reg_path)
            quality_rows = read_csv(quality_path)
            if len(reg_rows) != 27:
                raise ValueError(f"Expected 27 registration conditions: {reg_path}")
            if len(quality_rows) != 24:
                raise ValueError(f"Expected 24 quality conditions: {quality_path}")
            registration.extend(reg_rows)
            quality.extend(quality_rows)
    return registration, quality


def add_degradation(rows: list[dict], clean_map: dict[tuple[str, int], dict]) -> list[dict]:
    enriched = []
    for row in rows:
        method = row["method"]
        seed = int(row["seed"])
        clean = clean_map[(method, seed)]
        clean_iou = float(clean["iou"])
        absolute_drop = clean_iou - float(row["iou"])
        enriched.append(
            {
                **row,
                "clean_iou": clean_iou,
                "clean_mae": float(clean["mae_image_mean"]),
                "absolute_iou_drop": absolute_drop,
                "relative_iou_drop": absolute_drop / clean_iou if clean_iou else 0.0,
                "mae_increase": float(row["mae_image_mean"])
                - float(clean["mae_image_mean"]),
            }
        )
    return enriched


def registration_outputs(rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    clean = {
        (row["method"], int(row["seed"])): row
        for row in rows
        if row["perturbation_type"] == "clean"
    }
    translation = [row for row in rows if row["perturbation_type"] == "translation"]
    rotation = [row for row in rows if row["perturbation_type"] == "rotation"]
    for clean_row in clean.values():
        translation.append(
            {
                **clean_row,
                "perturbation_type": "translation",
                "perturbation_level": "0",
                "direction": "none",
            }
        )
        rotation.append(
            {
                **clean_row,
                "perturbation_type": "rotation",
                "perturbation_level": "0",
                "direction": "",
            }
        )
    translation = add_degradation(translation, clean)
    rotation = add_degradation(rotation, clean)

    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in translation:
        grouped[(row["method"], int(float(row["perturbation_level"])))].append(row)
    summary = []
    for (method, pixels), group in sorted(grouped.items()):
        seed_means_iou = []
        seed_means_mae = []
        seed_drops = []
        direction_sds = []
        for seed in SEEDS:
            seed_rows = [row for row in group if int(row["seed"]) == seed]
            ious = [float(row["iou"]) for row in seed_rows]
            maes = [float(row["mae_image_mean"]) for row in seed_rows]
            drops = [float(row["absolute_iou_drop"]) for row in seed_rows]
            seed_means_iou.append(statistics.mean(ious))
            seed_means_mae.append(statistics.mean(maes))
            seed_drops.append(statistics.mean(drops))
            direction_sds.append(statistics.stdev(ious) if len(ious) > 1 else 0.0)
        iou_mean, iou_sd = mean_sd(seed_means_iou)
        mae_mean, mae_sd = mean_sd(seed_means_mae)
        drop_mean, drop_sd = mean_sd(seed_drops)
        summary.append(
            {
                "method": method,
                "translation_pixels": pixels,
                "iou_mean_over_directions_and_seeds": iou_mean,
                "iou_seed_sd": iou_sd,
                "mean_direction_sd": statistics.mean(direction_sds),
                "mae_mean_over_directions_and_seeds": mae_mean,
                "mae_seed_sd": mae_sd,
                "absolute_iou_drop_mean": drop_mean,
                "absolute_iou_drop_seed_sd": drop_sd,
            }
        )

    tolerance = []
    for method in METHODS:
        method_rows = [row for row in summary if row["method"] == method]
        clean_iou = next(
            float(row["iou_mean_over_directions_and_seeds"])
            for row in method_rows
            if int(row["translation_pixels"]) == 0
        )
        thresholds = {
            "T1_relative_2pct": lambda row: float(row["absolute_iou_drop_mean"])
            <= 0.02 * clean_iou,
            "T2_absolute_0p01": lambda row: float(row["absolute_iou_drop_mean"]) <= 0.01,
            "T3_absolute_0p02": lambda row: float(row["absolute_iou_drop_mean"]) <= 0.02,
        }
        for threshold, predicate in thresholds.items():
            eligible = [
                int(row["translation_pixels"])
                for row in method_rows
                if predicate(row)
            ]
            tolerance.append(
                {
                    "method": method,
                    "perturbation": "translation",
                    "threshold": threshold,
                    "maximum_level_meeting_criterion": max(eligible) if eligible else 0,
                    "unit": "pixels",
                    "note": "Descriptive threshold predefined by this study; not an industry standard.",
                }
            )

    abs_rotation: dict[tuple[str, float], list[dict]] = defaultdict(list)
    for row in rotation:
        abs_rotation[(row["method"], abs(float(row["perturbation_level"])))].append(row)
    rotation_summary = []
    for (method, angle), group in sorted(abs_rotation.items()):
        seed_ious = []
        seed_drops = []
        for seed in SEEDS:
            seed_rows = [row for row in group if int(row["seed"]) == seed]
            seed_ious.append(statistics.mean(float(row["iou"]) for row in seed_rows))
            seed_drops.append(
                statistics.mean(float(row["absolute_iou_drop"]) for row in seed_rows)
            )
        iou_mean, iou_sd = mean_sd(seed_ious)
        drop_mean, drop_sd = mean_sd(seed_drops)
        rotation_summary.append(
            {
                "method": method,
                "absolute_rotation_degrees": angle,
                "iou_mean_over_signs_and_seeds": iou_mean,
                "iou_seed_sd": iou_sd,
                "absolute_iou_drop_mean": drop_mean,
                "absolute_iou_drop_seed_sd": drop_sd,
            }
        )
    for method in METHODS:
        method_rows = [row for row in rotation_summary if row["method"] == method]
        clean_iou = next(
            float(row["iou_mean_over_signs_and_seeds"])
            for row in method_rows
            if float(row["absolute_rotation_degrees"]) == 0.0
        )
        thresholds = {
            "T1_relative_2pct": lambda row: float(row["absolute_iou_drop_mean"])
            <= 0.02 * clean_iou,
            "T2_absolute_0p01": lambda row: float(row["absolute_iou_drop_mean"]) <= 0.01,
            "T3_absolute_0p02": lambda row: float(row["absolute_iou_drop_mean"]) <= 0.02,
        }
        for threshold, predicate in thresholds.items():
            eligible = [
                float(row["absolute_rotation_degrees"])
                for row in method_rows
                if predicate(row)
            ]
            tolerance.append(
                {
                    "method": method,
                    "perturbation": "rotation",
                    "threshold": threshold,
                    "maximum_level_meeting_criterion": max(eligible) if eligible else 0.0,
                    "unit": "degrees",
                    "note": "Descriptive threshold predefined by this study; not an industry standard.",
                }
            )

    write_csv(REG_ROOT / "translation_per_direction.csv", translation)
    write_csv(REG_ROOT / "translation_summary.csv", summary)
    write_csv(REG_ROOT / "rotation_results.csv", rotation)
    write_csv(REG_ROOT / "origin_registration_translation.csv", summary)
    write_csv(REG_ROOT / "origin_registration_rotation.csv", rotation_summary)
    write_csv(TABLE_DIR / "table_registration_sensitivity.csv", tolerance)
    write_csv(STATS_DIR / "rotation_absolute_angle_summary.csv", rotation_summary)
    return translation, rotation, tolerance


def quality_outputs(rows: list[dict]) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    clean = {}
    for row in rows:
        if row["perturbation_type"] == "blur" and float(row["perturbation_level"]) == 0.0:
            clean[(row["method"], int(row["seed"]))] = row
    blur = add_degradation(
        [row for row in rows if row["perturbation_type"] == "blur"], clean
    )
    noise = add_degradation(
        [row for row in rows if row["perturbation_type"] == "noise"], clean
    )
    quantization = add_degradation(
        [row for row in rows if row["perturbation_type"] == "quantization"], clean
    )
    write_csv(QUALITY_ROOT / "dolp_blur_results.csv", blur)
    write_csv(QUALITY_ROOT / "dolp_noise_results.csv", noise)
    write_csv(QUALITY_ROOT / "dolp_quantization_results.csv", quantization)

    summaries = []
    for kind, kind_rows in (
        ("blur", blur),
        ("noise", noise),
        ("quantization", quantization),
    ):
        grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for row in kind_rows:
            grouped[(row["method"], row["perturbation_level"])].append(row)
        for (method, level), group in sorted(
            grouped.items(), key=lambda item: (item[0][0], float(item[0][1]))
        ):
            seed_values: dict[int, list[dict]] = defaultdict(list)
            for row in group:
                seed_values[int(row["seed"])].append(row)
            seed_iou = [
                statistics.mean(float(row["iou"]) for row in seed_values[seed])
                for seed in SEEDS
            ]
            seed_mae = [
                statistics.mean(
                    float(row["mae_image_mean"]) for row in seed_values[seed]
                )
                for seed in SEEDS
            ]
            seed_drop = [
                statistics.mean(
                    float(row["absolute_iou_drop"]) for row in seed_values[seed]
                )
                for seed in SEEDS
            ]
            iou_mean, iou_sd = mean_sd(seed_iou)
            mae_mean, mae_sd = mean_sd(seed_mae)
            drop_mean, drop_sd = mean_sd(seed_drop)
            summaries.append(
                {
                    "perturbation_type": kind,
                    "method": method,
                    "level": level,
                    "iou_mean": iou_mean,
                    "iou_seed_sd": iou_sd,
                    "mae_mean": mae_mean,
                    "mae_seed_sd": mae_sd,
                    "absolute_iou_drop_mean": drop_mean,
                    "absolute_iou_drop_seed_sd": drop_sd,
                }
            )
    write_csv(TABLE_DIR / "table_dolp_quality_summary.csv", summaries)
    write_csv(QUALITY_ROOT / "origin_dolp_quality.csv", summaries)
    return blur, noise, quantization, summaries


def update_traceability(registration: list[dict], quality: list[dict]) -> None:
    path = REPORT_DIR / "RESULT_TRACEABILITY.csv"
    existing = read_csv(path)
    fieldnames = list(existing[0])
    retained = [
        row
        for row in existing
        if row.get("experiment") not in {"registration_stress", "dolp_quality_stress"}
    ]
    for experiment, rows in (
        ("registration_stress", registration),
        ("dolp_quality_stress", quality),
    ):
        for row in rows:
            retained.append(
                {
                    "experiment": experiment,
                    "method": row["method"],
                    "variant": row["method_label"],
                    "active_gate_levels": "",
                    "perturbation_type": row["perturbation_type"],
                    "perturbation_level": row["perturbation_level"],
                    "direction": row["direction"],
                    "seed": row["seed"],
                    "noise_realization": row["noise_realization"],
                    "best_epoch": row["best_epoch"],
                    "validation_iou": row["validation_iou"],
                    "test_iou": row["iou"],
                    "f1": row["f1"],
                    "precision": row["precision"],
                    "recall": row["recall"],
                    "mae": row["mae_image_mean"],
                    "config_path": row["source_config_path"],
                    "checkpoint_path": row["checkpoint_path"],
                    "prediction_path": row["prediction_path"],
                    "metrics_path": row["per_image_metrics_path"],
                    "script": str(Path(__file__).resolve()),
                    "verified": "yes",
                    "notes": "Synthetic test-time perturbation; checkpoint remains validation-selected.",
                }
            )
    write_csv(path, retained)


def lookup_summary(
    rows: list[dict],
    kind: str,
    method: str,
    level: float,
) -> dict:
    return next(
        row
        for row in rows
        if row["perturbation_type"] == kind
        and row["method"] == method
        and abs(float(row["level"]) - level) < 1e-12
    )


def write_report(
    translation: list[dict],
    rotation: list[dict],
    tolerance: list[dict],
    quality_summary: list[dict],
) -> None:
    pfe_translation_t2 = next(
        row
        for row in tolerance
        if row["method"] == "pfe_multiply"
        and row["perturbation"] == "translation"
        and row["threshold"] == "T2_absolute_0p01"
    )
    pfe_rotation_t2 = next(
        row
        for row in tolerance
        if row["method"] == "pfe_multiply"
        and row["perturbation"] == "rotation"
        and row["threshold"] == "T2_absolute_0p01"
    )
    pfe_blur = lookup_summary(quality_summary, "blur", "pfe_multiply", 0.5)
    pfe_noise = lookup_summary(quality_summary, "noise", "pfe_multiply", 0.01)
    pfe_quant = lookup_summary(quality_summary, "quantization", "pfe_multiply", 6.0)
    direct_noise = lookup_summary(quality_summary, "noise", "direct", 0.01)

    mild_max = max(
        float(pfe_blur["absolute_iou_drop_mean"]),
        float(pfe_noise["absolute_iou_drop_mean"]),
        float(pfe_quant["absolute_iou_drop_mean"]),
    )
    if mild_max <= 0.01:
        quality_status = "A: mild DoLP degradation is descriptively stable"
    elif float(pfe_noise["absolute_iou_drop_mean"]) > float(
        pfe_blur["absolute_iou_drop_mean"]
    ):
        quality_status = "B: noise-sensitive but comparatively blur-tolerant"
    elif float(pfe_blur["absolute_iou_drop_mean"]) > float(
        pfe_noise["absolute_iou_drop_mean"]
    ):
        quality_status = "C: blur-sensitive but comparatively noise-tolerant"
    else:
        quality_status = "D: broad sensitivity requires a stronger limitation"

    pfe_more_noise_sensitive = float(pfe_noise["absolute_iou_drop_mean"]) > float(
        direct_noise["absolute_iou_drop_mean"]
    ) + 0.005
    translation_limit = float(pfe_translation_t2["maximum_level_meeting_criterion"])
    should_train_aug = translation_limit < 2 or pfe_more_noise_sensitive
    recommendation = (
        "建议先在 validation 上预注册增强范围，再单独训练 registration-augmented PFE；本队列不自动启动。"
        if should_train_aug
        else "当前描述性结果不足以证明必须新增 registration-augmented 训练。"
    )

    report = f"""# 配准误差与 DoLP 质量退化报告

## 协议

Direct RGB+DoLP、PFE multiply 和 PFE 1x1 均使用五个既有 validation-best 正式 checkpoint。所有扰动只作用于 352 x 352 的 DoLP，RGB 与 GT 不变。平移和旋转使用双线性插值与 reflection padding；噪声样本在方法与种子之间固定复用。这里的阈值是本研究预先定义的描述性工程阈值，不是行业标准，也不等同于真实硬件测试。

## 配准容限

- PFE multiply 在 T2（IoU 绝对下降不超过 0.01）下满足判据的最大测试平移：{pfe_translation_t2['maximum_level_meeting_criterion']} px。
- PFE multiply 在 T2 下满足判据的最大测试绝对旋转：{pfe_rotation_t2['maximum_level_meeting_criterion']} degree。
- 32 px 与 +/-4 degree 未纳入核心结论；本轮使用预注册的 0/1/2/4/8/16 px 和 0/+/-0.5/+/-1/+/-2 degree。

## DoLP 质量

- sigma=0.5 模糊的 PFE IoU 平均下降：{float(pfe_blur['absolute_iou_drop_mean']):.4f}。
- std=0.01 噪声的 PFE IoU 平均下降：{float(pfe_noise['absolute_iou_drop_mean']):.4f}。
- 6-bit 量化的 PFE IoU 平均下降：{float(pfe_quant['absolute_iou_drop_mean']):.4f}。
- 自动描述性分类：**{quality_status}**。

## 是否需要增强训练

{recommendation}

没有执行低 S0 感知噪声实验：当前没有经 validation 或物理标定预先确定的 alpha、epsilon 与噪声上限，强行设置会把 test 曲线用于参数选择。

## 论文边界

这些结果只能支持“在合成的渐进式 DoLP 失配和质量退化下观察到的敏感性”。不得写成真实传感器鲁棒性、真实标定容限或硬件部署证明。
"""
    (REPORT_DIR / "REGISTRATION_AND_DOLP_QUALITY_REPORT_CN.md").write_text(
        report, encoding="utf-8"
    )


def main() -> None:
    registration_rows, quality_rows = load_suites()
    translation, rotation, tolerance = registration_outputs(registration_rows)
    blur, noise, quantization, quality_summary = quality_outputs(quality_rows)
    update_traceability(
        [*translation, *rotation],
        [*blur, *noise, *quantization],
    )
    write_report(translation, rotation, tolerance, quality_summary)
    print(
        f"Stress summary complete: registration={len(registration_rows)}, "
        f"quality={len(quality_rows)}."
    )


if __name__ == "__main__":
    main()
