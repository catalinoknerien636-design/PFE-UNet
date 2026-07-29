from __future__ import annotations

import argparse
import csv
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from py_sod_metrics import Emeasure, Fmeasure, MAE, Smeasure, WeightedFmeasure
from tqdm import tqdm


CODE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from scripts.compute_cod_metrics import (
    confusion_curve,
    enhanced_alignment_from_counts,
    f_measure_curve,
    sample_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute frozen COD and per-image metrics.")
    parser.add_argument("--pred_dir", type=Path, required=True)
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument(
        "--gt_dir",
        type=Path,
        default=REPOSITORY_ROOT / "data" / "PCOD_1200" / "full-dataset" / "test" / "test-gt",
    )
    return parser.parse_args()


def load_u8(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


def resize(pred: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if pred.shape == shape:
        return pred
    return cv2.resize(pred, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def binary_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    binary = pred >= 0.5
    truth = gt > 0.5
    tp = float(np.logical_and(binary, truth).sum())
    fp = float(np.logical_and(binary, ~truth).sum())
    fn = float(np.logical_and(~binary, truth).sum())
    tn = float(np.logical_and(~binary, ~truth).sum())
    eps = 1e-7
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    return {
        "iou": iou,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def main() -> None:
    args = parse_args()
    gt_paths = sorted(args.gt_dir.glob("*.png"))
    if len(gt_paths) != 230:
        raise ValueError(f"Expected 230 official test masks, found {len(gt_paths)}")
    pred_paths = {path.stem: path for path in args.pred_dir.glob("*.png")}
    gt_stems = {path.stem for path in gt_paths}
    if set(pred_paths) != gt_stems:
        missing = sorted(gt_stems - set(pred_paths))[:5]
        extra = sorted(set(pred_paths) - gt_stems)[:5]
        raise ValueError(f"Prediction/GT stem mismatch; missing={missing}, extra={extra}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fm = Fmeasure()
    mae = MAE()
    sm = Smeasure()
    em = Emeasure()
    wfm = WeightedFmeasure()
    per_image: list[dict] = []

    for gt_path in tqdm(gt_paths, desc=f"COD:{args.run_name}"):
        gt_u8 = load_u8(gt_path)
        pred_u8 = resize(load_u8(pred_paths[gt_path.stem]), gt_u8.shape)
        fm.step(pred_u8, gt_u8)
        mae.step(pred_u8, gt_u8)
        sm.step(pred_u8, gt_u8)
        em.step(pred_u8, gt_u8)
        wfm.step(pred_u8, gt_u8)

        pred = pred_u8.astype(np.float64) / 255.0
        gt = gt_u8.astype(np.float64) / 255.0
        gt_bool = gt > 0.5
        cod = sample_metrics(pred, gt_bool)
        tp, fp, fn, tn = confusion_curve(pred, gt_bool)
        e_curve = enhanced_alignment_from_counts(tp, fp, fn, tn)
        f_curve = f_measure_curve(tp, fp, fn)
        per_image.append(
            {
                "run": args.run_name,
                "stem": gt_path.stem,
                **binary_metrics(pred, gt),
                "mae_image": float(np.abs(pred - gt_bool.astype(np.float64)).mean()),
                "s_measure": cod["s_alpha"],
                "e_measure_adaptive": cod["e_phi_adaptive"],
                "e_measure_mean": float(np.mean(e_curve)),
                "e_measure_max": cod["e_phi_max"],
                "weighted_f_measure": cod["f_beta_w"],
                "f_measure_mean": float(np.mean(f_curve)),
                "f_measure_max": cod["f_beta_max"],
            }
        )

    fm_result = fm.get_results()["fm"]
    em_result = em.get_results()["em"]
    aggregate = {
        "run": args.run_name,
        "num_images": len(gt_paths),
        "s_measure": float(sm.get_results()["sm"]),
        "weighted_f_measure": float(wfm.get_results()["wfm"]),
        "mae_image_mean": float(mae.get_results()["mae"]),
        "e_measure_adaptive": float(em_result["adp"]),
        "e_measure_mean": float(np.mean(em_result["curve"])),
        "e_measure_max": float(np.max(em_result["curve"])),
        "f_measure_adaptive": float(fm_result["adp"]),
        "f_measure_mean": float(np.mean(fm_result["curve"])),
        "f_measure_max": float(np.max(fm_result["curve"])),
        "metric_package": "pysodmetrics==1.6.2",
    }
    write_csv(args.out_dir / "cod_metrics.csv", [aggregate])
    write_csv(args.out_dir / "cod_per_sample.csv", per_image)
    print(f"Saved COD metrics for {args.run_name}: {args.out_dir}")


if __name__ == "__main__":
    main()
