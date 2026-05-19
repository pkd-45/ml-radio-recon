"""
dataset.py
----------
PyTorch ``Dataset`` that produces (dirty_image, true_image) pairs on
the fly from a directory or HDF5 file of true galaxy images.

For each sample a fresh uv-mask is drawn from a distribution of
realistic ALMA-like configurations (random declination, hour-angle
range, antenna seed) so that the network learns reconstruction over a
*family* of point-spread functions rather than a fixed one. This is
the key generalisation property that any imaging network for a real
telescope must have.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from uv_simulator import make_uv_mask, dirty_image
from anomaly import inject_anomaly

# Lighter uv-synthesis settings (~2–3× faster forward model than defaults).
_FAST_MASK_TEMPLATE = [
    dict(n_ant=20, declination_deg=-30.0, hour_angle_range_h=(-1.0, 1.0),
         n_times=24, seed=0),
    dict(n_ant=24, declination_deg=-45.0, hour_angle_range_h=(-1.5, 1.5),
         n_times=32, seed=1),
    dict(n_ant=16, declination_deg=-15.0, hour_angle_range_h=(-0.75, 0.75),
         n_times=20, seed=2),
]


def _normalise(image: np.ndarray) -> np.ndarray:
    """Per-image min-max normalisation to [0, 1]."""
    lo, hi = float(image.min()), float(image.max())
    if hi - lo < 1e-8:
        return np.zeros_like(image)
    return (image - lo) / (hi - lo)


class SparseAperturePairs(Dataset):
    """Pairs of (dirty, truth) images with on-the-fly uv-sampling.

    Parameters
    ----------
    images
        Either a path to an ``.npy`` / ``.h5`` file or an in-memory
        array of shape ``(N, H, W)`` with non-negative pixel values.
    augment
        If True, apply random flips and 90-degree rotations to the
        truth image *before* the forward model (so dirty and truth
        remain consistent).
    noise_sigma_range
        Tuple ``(lo, hi)`` of per-sample additive complex Gaussian
        visibility noise levels (relative to the truth FFT peak).
    mask_kwargs_choices
        List of dicts of keyword arguments forwarded to
        :func:`uv_simulator.make_uv_mask`. One is drawn at random per
        sample. Pass a single-element list for fixed coverage.
    fast
        If True, use fewer hour-angle samples when building uv-masks
        (faster forward model, still multiple configurations).
    anomaly_frac
        Probability per sample of injecting one out-of-distribution
        feature into the truth image. Use 0.0 for training data (the
        network must see normal galaxies only); use e.g. 0.5 for an
        evaluation set when measuring anomaly detection.
    anomaly_kinds
        Which anomaly types to draw from; see :func:`anomaly.inject_anomaly`.
    seed
        Base RNG seed; per-index seeds are derived from it.
    """

    def __init__(self,
                 images: str | Path | np.ndarray,
                 augment: bool = True,
                 noise_sigma_range: tuple[float, float] = (0.0, 0.02),
                 mask_kwargs_choices: list[dict] | None = None,
                 fast: bool = False,
                 anomaly_frac: float = 0.0,
                 anomaly_kinds: tuple[str, ...] = ("streak", "point", "ring"),
                 seed: int = 1234):
        if isinstance(images, (str, Path)):
            p = Path(images)
            if p.suffix == ".npy":
                self.images = np.load(p, mmap_mode="r")
            elif p.suffix in (".h5", ".hdf5"):
                with h5py.File(p, "r") as f:
                    self.images = f["images"][:]
            else:
                raise ValueError(f"unrecognised file format: {p.suffix}")
        else:
            self.images = np.asarray(images)
        if self.images.ndim != 3:
            raise ValueError("images must be (N, H, W)")

        self.augment = augment
        self.noise_sigma_range = noise_sigma_range
        self.anomaly_frac = anomaly_frac
        self.anomaly_kinds = tuple(anomaly_kinds)
        self.seed = seed
        n = self.images.shape[1]
        if mask_kwargs_choices is not None:
            self.mask_kwargs_choices = mask_kwargs_choices
        elif fast:
            self.mask_kwargs_choices = [
                dict(n_pix=n, **kw) for kw in _FAST_MASK_TEMPLATE
            ]
        else:
            self.mask_kwargs_choices = [
                dict(n_pix=n, n_ant=24, declination_deg=-30.0,
                     hour_angle_range_h=(-1.5, 1.5), n_times=60, seed=0),
                dict(n_pix=n, n_ant=32, declination_deg=-45.0,
                     hour_angle_range_h=(-2.0, 2.0), n_times=80, seed=1),
                dict(n_pix=n, n_ant=20, declination_deg=-15.0,
                     hour_angle_range_h=(-1.0, 1.0), n_times=40, seed=2),
            ]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> dict:
        rng = np.random.default_rng(self.seed + idx)
        truth = np.asarray(self.images[idx], dtype=np.float32)
        truth = _normalise(truth)

        if self.augment:
            if rng.random() < 0.5:
                truth = truth[:, ::-1].copy()
            if rng.random() < 0.5:
                truth = truth[::-1, :].copy()
            k = int(rng.integers(0, 4))
            if k:
                truth = np.rot90(truth, k=k).copy()

        # optionally inject one out-of-distribution feature (eval only)
        anomaly_label = 0
        anomaly_kind = ""
        if self.anomaly_frac > 0.0 and rng.random() < self.anomaly_frac:
            kind = self.anomaly_kinds[
                int(rng.integers(0, len(self.anomaly_kinds)))]
            truth = inject_anomaly(truth, kind=kind, rng=rng)
            anomaly_label = 1
            anomaly_kind = kind

        kw = dict(self.mask_kwargs_choices[
            int(rng.integers(0, len(self.mask_kwargs_choices)))])
        kw["seed"] = int(rng.integers(0, 10_000))
        mask = make_uv_mask(**kw)

        sigma = float(rng.uniform(*self.noise_sigma_range))
        I_d, beam = dirty_image(truth, mask, noise_sigma=sigma, rng=rng)

        # to tensors with channel dimension
        return {
            "dirty": torch.from_numpy(I_d.astype(np.float32))[None],
            "truth": torch.from_numpy(truth.astype(np.float32))[None],
            "beam":  torch.from_numpy(beam.astype(np.float32))[None],
            "mask":  torch.from_numpy(mask.astype(np.float32))[None],
            "anomaly": torch.tensor(anomaly_label, dtype=torch.long),
            "anomaly_kind": anomaly_kind,
        }


def build_input(sample: dict, metadata: bool = False) -> torch.Tensor:
    """Assemble the network input tensor from a dataset sample/batch.

    With ``metadata=False`` the input is the 1-channel dirty image.
    With ``metadata=True`` it is a 3-channel stack of (dirty, beam,
    uv-mask) — the "metadata-aware" mode, in which the network is told
    the point-spread function and Fourier sampling that produced the
    dirty image, not just the dirty image itself.

    Works for both a single sample (channel dim = 0) and a batched
    tensor from a DataLoader (channel dim = 1).
    """
    dirty = sample["dirty"]
    if not metadata:
        return dirty
    cat_dim = 1 if dirty.dim() == 4 else 0
    return torch.cat([dirty, sample["beam"], sample["mask"]], dim=cat_dim)


if __name__ == "__main__":
    # quick smoke test: synthesise 4 random Gaussians, pull one sample
    rng = np.random.default_rng(0)
    n = 256
    imgs = np.zeros((4, n, n), dtype=np.float32)
    yy, xx = np.mgrid[:n, :n]
    for i in range(4):
        for _ in range(3):
            cy, cx = rng.integers(64, 192, size=2)
            sig = rng.uniform(3, 8)
            amp = rng.uniform(0.3, 1.0)
            imgs[i] += amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2)
                                     / (2 * sig ** 2))
    ds = SparseAperturePairs(imgs, anomaly_frac=1.0)
    sample = ds[0]
    for k, v in sample.items():
        if torch.is_tensor(v):
            print(f"{k:13s}  shape={tuple(v.shape)}  dtype={v.dtype}")
        else:
            print(f"{k:13s}  value={v!r}")
    inp1 = build_input(sample, metadata=False)
    inp3 = build_input(sample, metadata=True)
    print(f"1-channel input shape: {tuple(inp1.shape)}")
    print(f"3-channel input shape: {tuple(inp3.shape)}  "
          f"(dirty + beam + uv-mask)")
    print(f"anomaly label of sample 0: {int(sample['anomaly'])}")
