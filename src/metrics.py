"""
metrics.py
----------
Evaluation metrics for image reconstruction.

  - PSNR              : peak signal-to-noise ratio, in dB, for [0, 1]
                         normalised images.
  - SSIM              : structural similarity, via scikit-image.
  - peak_position_err : Euclidean pixel offset between the brightest
                         pixel of the reconstruction and the truth.
                         A small offset means morphology is preserved,
                         which is what an astrophysicist actually cares
                         about; PSNR alone can be misleading on sparse
                         sources.
"""

from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity


def psnr(truth: np.ndarray, pred: np.ndarray, data_range: float = 1.0
         ) -> float:
    mse = float(np.mean((truth - pred) ** 2))
    if mse < 1e-12:
        return float("inf")
    return float(10.0 * np.log10(data_range ** 2 / mse))


def ssim(truth: np.ndarray, pred: np.ndarray, data_range: float = 1.0
         ) -> float:
    return float(structural_similarity(truth, pred, data_range=data_range))


def peak_position_err(truth: np.ndarray, pred: np.ndarray) -> float:
    py, px = np.unravel_index(np.argmax(pred), pred.shape)
    ty, tx = np.unravel_index(np.argmax(truth), truth.shape)
    return float(np.hypot(py - ty, px - tx))


def evaluate_batch(truth: np.ndarray, pred: np.ndarray) -> dict:
    """Average PSNR/SSIM/peak-position-error over a batch of images.

    Both inputs may be of shape (N, H, W) or (H, W).
    """
    if truth.ndim == 2:
        truth = truth[None]
        pred = pred[None]
    psnrs, ssims, peaks = [], [], []
    for t, p in zip(truth, pred):
        p_clip = np.clip(p, 0.0, 1.0)
        psnrs.append(psnr(t, p_clip))
        ssims.append(ssim(t, p_clip))
        peaks.append(peak_position_err(t, p_clip))
    return {"psnr_db": float(np.mean(psnrs)),
            "ssim":    float(np.mean(ssims)),
            "peak_px": float(np.mean(peaks))}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    truth = rng.uniform(0, 1, size=(4, 64, 64)).astype(np.float32)
    pred = truth + 0.02 * rng.standard_normal(truth.shape).astype(np.float32)
    print(evaluate_batch(truth, pred))
