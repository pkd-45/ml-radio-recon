"""
download_data.py
----------------
Build a small training dataset of galaxy-like images at 256x256.

Two modes are supported:

1. ``--source synthetic`` (default, no network access required)
   Generates ``--n`` Sersic-profile galaxies with random ellipticity,
   position angle, central concentration, and occasional companion
   sources. This is the *recommended* mode for an initial smoke test
   because it is reproducible, gives perfect ground truth, and is
   cheap to scale.

2. ``--source galaxy10``
   Downloads the Galaxy10 DECaLS dataset
   (https://astronn.readthedocs.io/en/latest/galaxy10.html) of 17k
   real galaxy images and converts to single-channel intensity maps.
   Requires ``astroNN`` and an internet connection.

In both cases the output is an HDF5 file with a dataset ``images``
of shape ``(N, 256, 256)`` and ``dtype=float32`` normalised to [0, 1]
per image.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


# ---------------------------------------------------------------------------
# Sersic helper
# ---------------------------------------------------------------------------

def _b_n(n: float) -> float:
    """Approximate Ciotti & Bertin (1999) constant for a Sersic profile."""
    return 2.0 * n - 1.0 / 3.0 + 4.0 / (405.0 * n) + 46.0 / (25515.0 * n ** 2)


def _sersic_image(n_pix: int, cx: float, cy: float, r_e: float,
                  n_sersic: float, axis_ratio: float, pa_rad: float,
                  amp: float) -> np.ndarray:
    y, x = np.mgrid[:n_pix, :n_pix].astype(float)
    dx = x - cx
    dy = y - cy
    cosp, sinp = np.cos(pa_rad), np.sin(pa_rad)
    xp = cosp * dx + sinp * dy
    yp = (-sinp * dx + cosp * dy) / max(axis_ratio, 1e-3)
    r = np.sqrt(xp ** 2 + yp ** 2)
    return amp * np.exp(-_b_n(n_sersic)
                        * ((r / max(r_e, 1e-3)) ** (1.0 / n_sersic) - 1.0))


def synthesise(n: int, n_pix: int = 256, seed: int = 0) -> np.ndarray:
    """Generate ``n`` galaxy-like images of shape ``(n_pix, n_pix)``."""
    rng = np.random.default_rng(seed)
    out = np.zeros((n, n_pix, n_pix), dtype=np.float32)
    for i in range(n):
        img = np.zeros((n_pix, n_pix), dtype=np.float32)
        # primary galaxy
        cx = rng.uniform(0.35 * n_pix, 0.65 * n_pix)
        cy = rng.uniform(0.35 * n_pix, 0.65 * n_pix)
        r_e = rng.uniform(8.0, 30.0)
        ns = rng.uniform(0.7, 4.0)
        q = rng.uniform(0.3, 1.0)
        pa = rng.uniform(0.0, np.pi)
        img += _sersic_image(n_pix, cx, cy, r_e, ns, q, pa, amp=1.0)
        # occasional companion(s)
        n_comp = int(rng.integers(0, 3))
        for _ in range(n_comp):
            ccx = cx + rng.uniform(-60, 60)
            ccy = cy + rng.uniform(-60, 60)
            ccx = float(np.clip(ccx, 10, n_pix - 10))
            ccy = float(np.clip(ccy, 10, n_pix - 10))
            cre = rng.uniform(3.0, 12.0)
            cns = rng.uniform(0.7, 3.0)
            cq = rng.uniform(0.4, 1.0)
            cpa = rng.uniform(0.0, np.pi)
            camp = rng.uniform(0.05, 0.35)
            img += _sersic_image(n_pix, ccx, ccy, cre, cns, cq, cpa, camp)
        # faint pedestal so background pixels are not exactly zero
        img += rng.uniform(0.0, 0.005)
        # normalise
        lo, hi = float(img.min()), float(img.max())
        if hi - lo > 1e-8:
            img = (img - lo) / (hi - lo)
        out[i] = img.astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# Real galaxies via Galaxy10
# ---------------------------------------------------------------------------

def load_galaxy10(n: int, n_pix: int = 256) -> np.ndarray:
    """Download Galaxy10 DECaLS and return ``n`` grayscale, resized images."""
    try:
        from astroNN.datasets import load_galaxy10
    except ImportError as e:
        raise SystemExit(
            "astroNN is required for --source galaxy10. "
            "Install with: pip install astroNN") from e

    images, _ = load_galaxy10()
    if images.ndim == 4:
        # convert RGB -> luminance
        images = (0.2126 * images[..., 0]
                  + 0.7152 * images[..., 1]
                  + 0.0722 * images[..., 2])
    rng = np.random.default_rng(0)
    idx = rng.choice(len(images), size=min(n, len(images)), replace=False)
    sub = images[idx]
    # resize using PIL
    from PIL import Image
    out = np.empty((len(sub), n_pix, n_pix), dtype=np.float32)
    for i, im in enumerate(sub):
        pil = Image.fromarray(im.astype(np.float32))
        pil = pil.resize((n_pix, n_pix), Image.BILINEAR)
        a = np.asarray(pil).astype(np.float32)
        lo, hi = float(a.min()), float(a.max())
        out[i] = (a - lo) / max(hi - lo, 1e-8)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=("synthetic", "galaxy10"),
                   default="synthetic")
    p.add_argument("--n", type=int, default=2000,
                   help="number of images to generate / sample")
    p.add_argument("--size", type=int, default=128,
                   help="image side length in pixels (128 is ~4× faster than 256)")
    p.add_argument("--out", default="data/galaxies.h5")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.source == "synthetic":
        imgs = synthesise(args.n, n_pix=args.size, seed=args.seed)
    else:
        imgs = load_galaxy10(args.n, n_pix=args.size)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out, "w") as f:
        f.create_dataset("images", data=imgs,
                         compression="gzip", compression_opts=4)
    print(f"wrote {len(imgs)} images of shape {imgs.shape[1:]} to {out}")


if __name__ == "__main__":
    main()
