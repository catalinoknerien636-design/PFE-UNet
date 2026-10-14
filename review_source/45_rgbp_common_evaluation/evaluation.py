"""CPU common-grid metrics and paired-seed summaries; no prediction generation."""
from fractions import Fraction
import importlib.util
import itertools
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location('boundary45', ROOT / '06_experiment_preparation/boundary_metrics.py')
boundary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(boundary)
GRID = (352, 352)
SEEDS = (42, 2026, 3407)
VARIANTS = ('official_dateblock_reference', 'association_quarantine_sensitivity_v2')
EPS = 1e-7
BOUNDARY_KEYS = ('boundary_f1_1px', 'boundary_f1_2px', 'boundary_f1_4px', 'boundary_iou')


def prepare_grid(probability, native_mask, *, raw_grid):
    p, y = np.asarray(probability), np.asarray(native_mask)
    if tuple(raw_grid) not in (GRID, (416, 416)) or p.shape != tuple(raw_grid):
        raise ValueError('Only explicitly declared 352 or 416 raw grids admitted')
    if p.dtype != np.float32 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Raw finite FP32 probabilities required; no logits or PNG')
    if y.dtype != np.uint8 or y.ndim != 2 or not y.size:
        raise ValueError('Native mask must be nonempty uint8 HW')
    u = np.unique(y)
    if not (np.isin(u, [0, 1]).all() or np.isin(u, [0, 255]).all()):
        raise ValueError('Strict binary mask required')
    if p.shape != GRID:
        p = cv2.resize(p, GRID[::-1], interpolation=cv2.INTER_LINEAR)
    # Preserve the frozen RGBP loader's binarize-then-OpenCV-nearest convention.
    y = cv2.resize((y > 0).astype(np.float32), GRID[::-1], interpolation=cv2.INTER_NEAREST)
    return np.ascontiguousarray(p), np.ascontiguousarray(y)


def image_metrics(sample_id, probability, native_mask, *, raw_grid):
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError('Nonempty sample identifier required')
    p, y = prepare_grid(probability, native_mask, raw_grid=raw_grid)
    prediction, target = p > .5, y > .5
    result = boundary.pair_metrics(prediction, target, tolerances=(1, 2, 4), band_ratio=.02)
    # Use the same CPU FP32 reduction as train_rgbp.Metrics, not a FP64 substitute.
    mae = float((torch.from_numpy(p) - torch.from_numpy(y)).abs().mean())
    result.update(sample_id=sample_id, mae=mae, evaluated_pixels=352*352)
    return result


def aggregate(rows):
    if not rows or len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('Nonempty unique image rows required')
    for r in rows:
        if any(type(r[k]) is not int or r[k] < 0 for k in ('tp', 'fp', 'fn', 'tn')):
            raise ValueError('Nonnegative integer confusion counts required')
        if sum(r[k] for k in ('tp', 'fp', 'fn', 'tn')) != 352*352:
            raise ValueError('Confusion counts do not match common grid')
        if any(not np.isfinite(r[k]) or not 0 <= r[k] <= 1 for k in ('mae', *BOUNDARY_KEYS)):
            raise ValueError('Invalid metric scalar')
    tp, fp, fn, tn = (sum(r[k] for r in rows) for k in ('tp', 'fp', 'fn', 'tn'))
    inter, union = (sum(r[k] for r in rows) for k in ('boundary_intersection', 'boundary_union'))
    return {'images': len(rows), 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'iou': tp/(tp+fp+fn+EPS), 'f1': 2*tp/(2*tp+fp+fn+EPS),
        'precision': tp/(tp+fp+EPS), 'recall': tp/(tp+fn+EPS),
        'image_mae': sum(r['mae'] for r in rows)/len(rows),
        **{k+'_image_mean': sum(r[k] for r in rows)/len(rows) for k in BOUNDARY_KEYS},
        'boundary_iou_pooled': inter/union if union else 1.,
        'both_empty_images': sum(r['pred_empty'] and r['gt_empty'] for r in rows),
        'exactly_one_empty_images': sum(r['pred_empty'] != r['gt_empty'] for r in rows)}


def complete_report(rows, expected_ids, strata):
    if len(expected_ids) != len(set(expected_ids)) or set(strata) != set(expected_ids):
        raise ValueError('Invalid frozen membership/strata')
    if [r['sample_id'] for r in rows] != list(expected_ids):
        raise ValueError('Missing, extra, duplicated or reordered test records')
    names = ('train', 'validation', 'none_detected')
    if set(strata.values()) - set(names):
        raise ValueError('Unknown frozen descriptive stratum')
    return {'all_test': aggregate(rows), 'association_strata': {
        name: aggregate(part) if (part := [r for r in rows if strata[r['sample_id']] == name]) else None
        for name in names}, 'statistical_unit': 'training_seed', 'scene_independence_established': False}


def paired_three_seed(pfe, direct):
    if set(pfe) != set(SEEDS) or set(direct) != set(SEEDS):
        raise ValueError('Exactly the same three formal seed IDs required')
    differences = [float(pfe[s]) - float(direct[s]) for s in SEEDS]
    if not np.isfinite(differences).all():
        raise ValueError('Finite paired scalar values required')
    exact = [Fraction.from_float(d) for d in differences]
    observed = abs(sum(exact))
    extreme = sum(abs(sum(sign*d for sign,d in zip(signs, exact))) >= observed
                  for signs in itertools.product((-1, 1), repeat=3))
    return {'seeds': list(SEEDS), 'pfe_minus_direct': differences,
        'mean': float(np.mean(differences)), 'sample_sd': float(np.std(differences, ddof=1)),
        'minimum': min(differences), 'maximum': max(differences), 'n': 3,
        'two_sided_non_mid_p': extreme/8, 'enumerations': 8,
        'assumption': 'joint sign exchangeability; not implied by equal seed labels'}


def holm_two_primary(raw):
    if set(raw) != set(VARIANTS) or any(not np.isfinite(v) or not 0 <= v <= 1 for v in raw.values()):
        raise ValueError('Exactly the two prespecified primary IoU p values required')
    ordered = sorted(raw, key=lambda k: (raw[k], k))
    first = min(1., 2*raw[ordered[0]])
    return {ordered[0]: first, ordered[1]: max(first, raw[ordered[1]])}
