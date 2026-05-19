"""
eval.py
-------
Run the trained network and the CLEAN baseline on a held-out test
set and report PSNR / SSIM / peak-position error for each.

Usage
-----
    python eval.py --data data/galaxies.h5 \
                   --checkpoint checkpoints/run01/best.pt \
                   --n-test 64 \
                   --out results/eval_run01.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm.auto import tqdm

from clean_baseline import hogbom_clean
from dataset import SparseAperturePairs, build_input
from metrics import psnr, ssim, peak_position_err
from model import UNetAttn


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with h5py.File(args.data, "r") as f:
        images = f["images"][...]
    rng = np.random.default_rng(args.seed)
    test_idx = rng.choice(len(images), size=min(args.n_test, len(images)),
                          replace=False)
    test_imgs = images[test_idx]
    ds = SparseAperturePairs(test_imgs, augment=False,
                             noise_sigma_range=(0.005, 0.005),
                             fast=args.fast,
                             seed=args.seed)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_args = ckpt.get("args", {})
    metadata = bool(model_args.get("metadata", False))
    in_ch = 3 if metadata else 1
    model = UNetAttn(in_ch=in_ch,
                     base=model_args.get("base", 16),
                     num_heads=model_args.get("heads", 4),
                     residual=True).to(device).eval()
    model.load_state_dict(ckpt["model_state"])
    print(f"input mode: {'metadata-aware (dirty+beam+uv-mask)' if metadata else 'dirty-only'}")

    rows = []
    with torch.no_grad():
        for i in tqdm(range(len(ds)), desc="eval"):
            s = ds[i]
            dirty = s["dirty"].numpy()[0]
            truth = s["truth"].numpy()[0]
            beam = s["beam"].numpy()[0]

            clean = hogbom_clean(dirty, beam,
                                  gain=0.1, n_iter=args.clean_iters,
                                  threshold_frac=0.02)["restored"]
            clean = np.clip(clean, 0.0, 1.0)

            inp = build_input(s, metadata=metadata).unsqueeze(0).to(device)
            ml = model(inp).cpu().numpy()[0, 0]
            ml = np.clip(ml, 0.0, 1.0)

            for name, pred in (("dirty", dirty),
                               ("clean", clean),
                               ("ml",    ml)):
                rows.append({
                    "method": name,
                    "psnr_db": psnr(truth, np.clip(pred, 0, 1)),
                    "ssim":    ssim(truth, np.clip(pred, 0, 1)),
                    "peak_px": peak_position_err(truth, pred),
                })

    summary = {}
    for method in ("dirty", "clean", "ml"):
        sel = [r for r in rows if r["method"] == method]
        summary[method] = {
            "psnr_db_mean": float(np.mean([r["psnr_db"] for r in sel])),
            "psnr_db_std":  float(np.std([r["psnr_db"]  for r in sel])),
            "ssim_mean":    float(np.mean([r["ssim"]    for r in sel])),
            "ssim_std":     float(np.std([r["ssim"]     for r in sel])),
            "peak_px_mean": float(np.mean([r["peak_px"] for r in sel])),
            "peak_px_std":  float(np.std([r["peak_px"]  for r in sel])),
            "n":            len(sel),
        }

    print("\n=== test-set summary ===")
    for method, d in summary.items():
        print(f"{method:6s}  PSNR={d['psnr_db_mean']:.2f} +/- {d['psnr_db_std']:.2f} dB   "
              f"SSIM={d['ssim_mean']:.3f} +/- {d['ssim_std']:.3f}   "
              f"peak_err={d['peak_px_mean']:.2f} +/- {d['peak_px_std']:.2f} px")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"per_sample": rows, "summary": summary}, f, indent=2)
    print(f"\nwrote {out}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--n-test", type=int, default=64)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default="results/eval.json")
    p.add_argument("--clean-iters", type=int, default=100,
                   help="Hogbom CLEAN iterations (400 for publication quality)")
    p.add_argument("--fast", action="store_true",
                   help="Faster uv-mask synthesis when building dirty images")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
