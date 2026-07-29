from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import convolve, distance_transform_edt
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
EPS = np.finfo(np.float64).eps
BETA2 = 0.3
SM_ALPHA = 0.5
GAUSSIAN = cv2.getGaussianKernel(7, 5)
GAUSSIAN = GAUSSIAN @ GAUSSIAN.T


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute COD-standard metrics from formal grouped predictions."
    )
    parser.add_argument(
        "--summary_csv",
        type=Path,
        default=ROOT / "outputs" / "metrics" / "formal_grouped_summary.csv",
    )
    parser.add_argument(
        "--predictions_root",
        type=Path,
        default=ROOT / "outputs" / "predictions",
    )
    parser.add_argument(
        "--gt_dir",
        type=Path,
        default=ROOT / "datasets" / "PCOD_1200" / "full-dataset" / "test" / "test-gt",
    )
    parser.add_argument(
        "--out_csv",
        type=Path,
        default=ROOT / "outputs" / "metrics" / "formal_grouped_cod_summary.csv",
    )
    parser.add_argument(
        "--per_sample_csv",
        type=Path,
        default=ROOT / "outputs" / "metrics" / "formal_grouped_cod_per_sample.csv",
    )
    return parser.parse_args()


def load_gray(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float64) / 255.0


def load_gt(path: Path) -> np.ndarray:
    return load_gray(path) > 0.5


def object_score(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    mean = values.mean()
    return float(2.0 * mean / (mean * mean + 1.0 + values.std(ddof=0) + EPS))


def ssim(pred: np.ndarray, gt: np.ndarray) -> float:
    gt_mean = gt.mean()
    pred_mean = pred.mean()
    gt_var = ((gt - gt_mean) ** 2).mean()
    pred_var = ((pred - pred_mean) ** 2).mean()
    covariance = ((pred - pred_mean) * (gt - gt_mean)).mean()
    alpha = 4.0 * pred_mean * gt_mean * covariance
    beta = (pred_mean * pred_mean + gt_mean * gt_mean) * (pred_var + gt_var)
    if alpha != 0.0:
        return float(alpha / (beta + EPS))
    if alpha == 0.0 and beta == 0.0:
        return 1.0
    return 0.0


def centroid(gt: np.ndarray) -> tuple[int, int]:
    height, width = gt.shape
    if gt.sum() == 0:
        return width // 2, height // 2
    ys, xs = np.nonzero(gt)
    return int(np.round(xs.mean())), int(np.round(ys.mean()))


def region_score(pred: np.ndarray, gt: np.ndarray) -> float:
    height, width = gt.shape
    x, y = centroid(gt)
    x = min(max(x, 1), width - 1)
    y = min(max(y, 1), height - 1)
    areas = [
        (slice(0, y), slice(0, x), x * y),
        (slice(0, y), slice(x, width), (width - x) * y),
        (slice(y, height), slice(0, x), x * (height - y)),
        (slice(y, height), slice(x, width), (width - x) * (height - y)),
    ]
    total = float(height * width)
    return float(
        sum(ssim(pred[ys, xs], gt[ys, xs]) * area / total for ys, xs, area in areas)
    )


def s_measure(pred: np.ndarray, gt: np.ndarray) -> float:
    gt_mean = gt.mean()
    if gt_mean == 0.0:
        return float(1.0 - pred.mean())
    if gt_mean == 1.0:
        return float(pred.mean())
    object_term = gt_mean * object_score(pred[gt]) + (1.0 - gt_mean) * object_score(1.0 - pred[~gt])
    return max(0.0, float(SM_ALPHA * object_term + (1.0 - SM_ALPHA) * region_score(pred, gt)))


def confusion_curve(pred: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, ...]:
    bins = np.clip((pred * 255.0).astype(np.int32), 0, 255)
    positive = np.bincount(bins[gt].ravel(), minlength=256).astype(np.float64)
    negative = np.bincount(bins[~gt].ravel(), minlength=256).astype(np.float64)
    tp = np.cumsum(positive[::-1])[::-1]
    fp = np.cumsum(negative[::-1])[::-1]
    fn = positive.sum() - tp
    tn = negative.sum() - fp
    return tp, fp, fn, tn


def f_measure_curve(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> np.ndarray:
    precision = tp / (tp + fp + EPS)
    recall = tp / (tp + fn + EPS)
    return (1.0 + BETA2) * precision * recall / (BETA2 * precision + recall + EPS)


def enhanced_alignment_from_counts(
    tp: np.ndarray,
    fp: np.ndarray,
    fn: np.ndarray,
    tn: np.ndarray,
) -> np.ndarray:
    total = tp + fp + fn + tn
    pred_mean = (tp + fp) / total
    gt_mean = (tp + fn) / total
    score = np.zeros_like(tp, dtype=np.float64)
    for count, pred_value, gt_value in (
        (tp, 1.0, 1.0),
        (fp, 1.0, 0.0),
        (fn, 0.0, 1.0),
        (tn, 0.0, 0.0),
    ):
        alignment = (
            2.0 * (pred_value - pred_mean) * (gt_value - gt_mean)
            / ((pred_value - pred_mean) ** 2 + (gt_value - gt_mean) ** 2 + EPS)
        )
        enhanced = ((alignment + 1.0) ** 2) / 4.0
        score += count * enhanced
    return score / total


def e_measure_binary(binary: np.ndarray, gt: np.ndarray) -> float:
    tp = np.asarray([np.logical_and(binary, gt).sum()], dtype=np.float64)
    fp = np.asarray([np.logical_and(binary, ~gt).sum()], dtype=np.float64)
    fn = np.asarray([np.logical_and(~binary, gt).sum()], dtype=np.float64)
    tn = np.asarray([np.logical_and(~binary, ~gt).sum()], dtype=np.float64)
    return float(enhanced_alignment_from_counts(tp, fp, fn, tn)[0])


def weighted_f_measure(pred: np.ndarray, gt: np.ndarray) -> float:
    if not gt.any():
        return 0.0
    error = np.abs(pred - gt.astype(np.float64))
    distance, indices = distance_transform_edt(~gt, return_indices=True)
    extended_error = error.copy()
    background = ~gt
    extended_error[background] = error[indices[0][background], indices[1][background]]
    smoothed_error = convolve(extended_error, GAUSSIAN, mode="constant", cval=0.0)
    min_error = error.copy()
    replace = gt & (smoothed_error < error)
    min_error[replace] = smoothed_error[replace]
    weight = np.ones_like(pred)
    weight[background] = 2.0 - np.exp(np.log(0.5) / 5.0 * distance[background])
    weighted_error = min_error * weight
    true_positive = gt.sum() - weighted_error[gt].sum()
    false_positive = weighted_error[background].sum()
    recall = 1.0 - weighted_error[gt].mean()
    precision = true_positive / (true_positive + false_positive + EPS)
    return float(2.0 * recall * precision / (recall + precision + EPS))


def sample_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    if pred.shape != gt.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
    tp, fp, fn, tn = confusion_curve(pred, gt)
    f_curve = f_measure_curve(tp, fp, fn)
    e_curve = enhanced_alignment_from_counts(tp, fp, fn, tn)
    adaptive = pred >= min(2.0 * pred.mean(), 1.0)
    return {
        "s_alpha": s_measure(pred, gt),
        "e_phi_adaptive": e_measure_binary(adaptive, gt),
        "e_phi_max": float(e_curve.max()),
        "f_beta_max": float(f_curve.max()),
        "f_beta_w": weighted_f_measure(pred, gt),
        "mae": float(np.abs(pred - gt.astype(np.float64)).mean()),
    }


def formal_runs(summary_csv: Path) -> list[str]:
    with summary_csv.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    runs = [row["experiment"] for row in rows]
    invalid = [run for run in runs if not run.startswith("grouped_")]
    if invalid:
        raise ValueError(f"Only grouped_* formal runs are allowed: {invalid}")
    return runs


def write_csv(path: Path, rows: list[dict[str, str | float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    gt_paths = sorted(args.gt_dir.glob("*.png"))
    if not gt_paths:
        raise FileNotFoundError(f"No GT files found in: {args.gt_dir}")
    gt_by_stem = {path.stem: path for path in gt_paths}
    summary_rows: list[dict[str, str | float]] = []
    per_sample_rows: list[dict[str, str | float]] = []
    for run in formal_runs(args.summary_csv):
        pred_dir = args.predictions_root / run
        pred_by_stem = {path.stem: path for path in sorted(pred_dir.glob("*.png"))}
        if set(pred_by_stem) != set(gt_by_stem):
            raise ValueError(f"{run}: prediction stems do not match official test GT stems")
        run_rows = []
        for stem, gt_path in tqdm(gt_by_stem.items(), desc=run):
            metrics = sample_metrics(load_gray(pred_by_stem[stem]), load_gt(gt_path))
            row = {"experiment": run, "stem": stem, **metrics}
            run_rows.append(row)
            per_sample_rows.append(row)
        summary_rows.append(
            {
                "experiment": run,
                "num_samples": len(run_rows),
                **{
                    key: float(np.mean([float(row[key]) for row in run_rows]))
                    for key in run_rows[0]
                    if key not in {"experiment", "stem"}
                },
            }
        )
    write_csv(args.out_csv, summary_rows)
    write_csv(args.per_sample_csv, per_sample_rows)
    print(f"Saved summary: {args.out_csv}")
    print(f"Saved per-sample metrics: {args.per_sample_csv}")


if __name__ == "__main__":
    main()
