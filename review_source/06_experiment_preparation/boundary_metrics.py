"""Strict binary-mask metrics for the separate archived-PNG analysis.

Boundary-band definition: Cheng et al., Boundary IoU, CVPR 2021.
This module does not load models, normalize predictions, or choose thresholds.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def binary_mask(mask):
    array = np.asarray(mask)
    if array.ndim != 2 or not array.size:
        raise ValueError('Expected a nonempty 2-D binary mask')
    if not np.all((array == 0) | (array == 1)):
        raise ValueError('Mask must contain only 0/1; threshold explicitly upstream')
    return array.astype(bool, copy=False)


def boundary_band(mask, width=1):
    mask = binary_mask(mask)
    if not isinstance(width, (int, np.integer)) or width < 1:
        raise ValueError('Boundary width must be a positive integer')
    eroded = ndimage.binary_erosion(mask, structure=np.ones((3, 3), bool),
                                    iterations=int(width), border_value=0)
    return mask & ~eroded


def pair_metrics(prediction, truth, tolerances=(1, 2, 4), band_ratio=0.02):
    prediction, truth = binary_mask(prediction), binary_mask(truth)
    if prediction.shape != truth.shape:
        raise ValueError('Prediction and truth grids must agree')
    tolerances = tuple(tolerances)
    if not tolerances or any(not np.isfinite(t) or t < 0 for t in tolerances):
        raise ValueError('Boundary tolerances must be finite and nonnegative')
    if not np.isfinite(band_ratio) or not 0 < band_ratio <= 1:
        raise ValueError('Boundary band ratio must be in (0, 1]')
    pb, tb = boundary_band(prediction), boundary_band(truth)
    npb, ntb = int(pb.sum()), int(tb.sum())
    out = {'tp': int((prediction & truth).sum()),
           'fp': int((prediction & ~truth).sum()),
           'fn': int((~prediction & truth).sum()),
           'tn': int((~prediction & ~truth).sum()),
           'pred_boundary_pixels': npb, 'gt_boundary_pixels': ntb,
           'pred_empty': int(not prediction.any()), 'gt_empty': int(not truth.any())}
    if npb and ntb:
        p_dist = ndimage.distance_transform_edt(~tb)[pb]
        t_dist = ndimage.distance_transform_edt(~pb)[tb]
    for tol in tolerances:
        key = f'{tol:g}px'
        if not npb and not ntb:
            precision = recall = f1 = 1.0
        elif not npb or not ntb:
            precision = recall = f1 = 0.0
        else:
            precision, recall = float(np.mean(p_dist <= tol)), float(np.mean(t_dist <= tol))
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out.update({f'boundary_precision_{key}': precision,
                    f'boundary_recall_{key}': recall, f'boundary_f1_{key}': f1})
    width = max(1, int(round(band_ratio * np.hypot(*truth.shape))))
    pband, tband = boundary_band(prediction, width), boundary_band(truth, width)
    intersection, union = int((pband & tband).sum()), int((pband | tband).sum())
    out.update(boundary_band_width=width, boundary_intersection=intersection,
               boundary_union=union, boundary_iou=intersection / union if union else 1.0)
    return out
