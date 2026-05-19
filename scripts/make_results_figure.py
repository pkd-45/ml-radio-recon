"""
make_results_figure.py
----------------------
Produce the headline figure of the prototype: a 4 x 4 grid of
(truth, dirty, CLEAN, ML) reconstructions on four held-out galaxies,
with PSNR / SSIM / peak-position-error annotated on each panel.

Usage
-----
    python scripts/make_results_figure.py \
        --data data/galaxies.h5 \
        --checkpoint checkpoints/run01/best.pt \
        --out results/headline.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from clean_baseline import hogbom_clean
from dataset import SparseAperturePairs, build_input
from metrics import psnr, ssim, peak_position_err
from model import UNetAttn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--n-show", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="results/headline.png")
    p.add_argument("--clean-iter", type=int, default=100,
                   help="Hogbom CLEAN iterations (400 for publication quality)")
    p.add_argument("--fast", action="store_true",
                   help="Faster uv-mask synthesis when building dirty images")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with h5py.File(args.data, "r") as f:
        images = f["images"][...]
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(images), size=args.n_show, replace=False)
    test_imgs = images[idx]

    ds = SparseAperturePairs(test_imgs, augment=False,
                             noise_sigma_range=(0.005, 0.005),
                             fast=args.fast,
                             seed=args.seed)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    margs = ckpt.get("args", {})
    metadata = bool(margs.get("metadata", False))
    in_ch = 3 if metadata else 1
    model = UNetAttn(in_ch=in_ch, base=margs.get("base", 16),
                     num_heads=margs.get("heads", 4),
                     residual=True).to(device).eval()
    model.load_state_dict(ckpt["model_state"])
    print(f"input mode: {'metadata-aware' if metadata else 'dirty-only'}")

    rows = []
    for i in range(len(ds)):
        s = ds[i]
        truth = s["truth"].numpy()[0]
        dirty = s["dirty"].numpy()[0]
        beam = s["beam"].numpy()[0]
        clean = hogbom_clean(dirty, beam,
                              gain=0.1, n_iter=args.clean_iter,
                              threshold_frac=0.02)["restored"]
        with torch.no_grad():
            inp = build_input(s, metadata=metadata).unsqueeze(0).to(device)
            ml = model(inp).cpu().numpy()[0, 0]
        rows.append((truth, dirty, np.clip(clean, 0, 1),
                     np.clip(ml, 0, 1)))

    # ----- figure -----
    nrows = len(rows)
    fig, axes = plt.subplots(nrows, 4, figsize=(11, 2.75 * nrows + 0.3),
                              constrained_layout=True)
    col_titles = ["truth", "dirty", "CLEAN", "ML (U-Net + attn)"]
    if nrows == 1:
        axes = axes[None, :]

    for r, (truth, dirty, clean, ml) in enumerate(rows):
        vmax = float(truth.max())
        for c, (name, img) in enumerate(zip(col_titles,
                                             [truth, dirty, clean, ml])):
            ax = axes[r, c]
            im = ax.imshow(img, origin="lower", cmap="magma",
                           vmin=-0.02 * vmax, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(name, fontsize=11)
            if c == 0:
                ax.set_ylabel(f"galaxy {r+1}", fontsize=10)
            if c >= 1:
                ps = psnr(truth, np.clip(img, 0, 1))
                ss = ssim(truth, np.clip(img, 0, 1))
                pp = peak_position_err(truth, img)
                ax.text(0.03, 0.97,
                        f"PSNR  {ps:5.2f} dB\nSSIM  {ss:5.3f}\npeak  {pp:4.1f} px",
                        transform=ax.transAxes,
                        ha="left", va="top",
                        fontsize=8, color="white",
                        bbox=dict(facecolor="black", alpha=0.45,
                                  edgecolor="none", pad=2))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
