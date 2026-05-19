"""
train.py
--------
Training script for the U-Net + Transformer reconstruction network.

Usage
-----
    python train.py --data data/galaxies.h5 \
                    --out checkpoints/run01 \
                    --epochs 30 --batch-size 8

Loss
----
    L = L1(pred, truth)  +  lambda_ssim * (1 - SSIM(pred, truth))

L1 is robust to the sparse, high-dynamic-range pixel distributions
typical of astronomical images, and the SSIM term keeps morphology
faithful. We do not use plain L2 because it tends to over-smooth.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm

from dataset import SparseAperturePairs, build_input
from model import UNetAttn, count_parameters


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def _gaussian_kernel(window_size: int = 11, sigma: float = 1.5,
                    device=None) -> torch.Tensor:
    coords = torch.arange(window_size, device=device).float() \
        - (window_size - 1) / 2
    g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    return g.unsqueeze(0) * g.unsqueeze(1)         # (k, k)


def ssim_loss(pred: torch.Tensor, truth: torch.Tensor,
              window_size: int = 11, data_range: float = 1.0
              ) -> torch.Tensor:
    """Differentiable SSIM loss = 1 - mean SSIM."""
    k = _gaussian_kernel(window_size, device=pred.device)
    k = k.view(1, 1, window_size, window_size)
    pad = window_size // 2

    mu_p = F.conv2d(pred, k, padding=pad)
    mu_t = F.conv2d(truth, k, padding=pad)
    mu_pt = mu_p * mu_t
    mu_pp = mu_p ** 2
    mu_tt = mu_t ** 2

    sig_pp = F.conv2d(pred * pred, k, padding=pad) - mu_pp
    sig_tt = F.conv2d(truth * truth, k, padding=pad) - mu_tt
    sig_pt = F.conv2d(pred * truth, k, padding=pad) - mu_pt

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    num = (2 * mu_pt + c1) * (2 * sig_pt + c2)
    den = (mu_pp + mu_tt + c1) * (sig_pp + sig_tt + c2)
    return 1.0 - (num / den).mean()


# ---------------------------------------------------------------------------
# AMP helpers (torch.amp.GradScaler needs PyTorch >= 2.3)
# ---------------------------------------------------------------------------

def _make_grad_scaler(enabled: bool):
    if hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _autocast(enabled: bool):
    if hasattr(torch.amp, "GradScaler"):
        return torch.amp.autocast("cuda", enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # ---- data ----
    with h5py.File(args.data, "r") as f:
        n_total = f["images"].shape[0]
        n_load = n_total if args.max_samples <= 0 else min(args.max_samples, n_total)
        images = f["images"][:n_load]
    if args.image_size > 0 and images.shape[1] != args.image_size:
        step = images.shape[1] // args.image_size
        if step < 1 or images.shape[1] // step != args.image_size:
            raise ValueError(
                f"--image-size {args.image_size} must evenly divide "
                f"stored size {images.shape[1]}")
        images = images[:, ::step, ::step]
    print(f"loaded {len(images)} truth images of shape {images.shape[1:]}"
          + ("  [fast uv-masks]" if args.fast else ""))

    workers = args.workers
    if workers < 0:
        workers = min(4, os.cpu_count() or 1)

    full = SparseAperturePairs(images, augment=True,
                               noise_sigma_range=(0.0, 0.02),
                               fast=args.fast)
    n_val = max(1, int(0.1 * len(full)))
    n_train = len(full) - n_val
    train_ds, val_ds = random_split(full, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(0))
    dl_kw = dict(batch_size=args.batch_size, pin_memory=device.type == "cuda")
    if workers > 0:
        dl_kw.update(num_workers=workers, persistent_workers=True)
    train_dl = DataLoader(train_ds, shuffle=True, **dl_kw)
    val_dl = DataLoader(val_ds, shuffle=False, **dl_kw)

    # ---- model ----
    in_ch = 3 if args.metadata else 1
    model = UNetAttn(in_ch=in_ch, base=args.base,
                     num_heads=args.heads, residual=True).to(device)
    print(f"input mode: {'metadata-aware (dirty+beam+uv-mask)' if args.metadata else 'dirty-only'}"
          f"  ->  in_ch={in_ch}")
    print(f"model parameters: {count_parameters(model):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs)
    use_amp = device.type == "cuda"
    scaler = _make_grad_scaler(use_amp)
    l1 = nn.L1Loss()

    # ---- loop ----
    history = []
    best_val = float("inf")
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        for batch in tqdm(train_dl, desc=f"epoch {epoch+1}/{args.epochs}",
                          leave=False):
            inp = build_input(batch, metadata=args.metadata).to(
                device, non_blocking=True)
            truth = batch["truth"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with _autocast(use_amp):
                pred = model(inp)
                loss = l1(pred, truth) \
                       + args.lambda_ssim * ssim_loss(pred, truth)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            train_losses.append(loss.item())

        # ---- validation ----
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_dl:
                inp = build_input(batch, metadata=args.metadata).to(
                    device, non_blocking=True)
                truth = batch["truth"].to(device, non_blocking=True)
                with _autocast(use_amp):
                    pred = model(inp)
                    loss = l1(pred, truth) \
                           + args.lambda_ssim * ssim_loss(pred, truth)
                val_losses.append(loss.item())

        sched.step()
        train_mean = float(np.mean(train_losses))
        val_mean = float(np.mean(val_losses))
        elapsed = time.time() - t0
        print(f"epoch {epoch+1:3d}  "
              f"train={train_mean:.4f}  val={val_mean:.4f}  "
              f"lr={sched.get_last_lr()[0]:.2e}  "
              f"elapsed={elapsed/60:.1f} min")
        history.append({"epoch": epoch + 1,
                        "train_loss": train_mean,
                        "val_loss": val_mean})

        if val_mean < best_val:
            best_val = val_mean
            torch.save({"model_state": model.state_dict(),
                        "args": vars(args),
                        "epoch": epoch + 1},
                       out / "best.pt")
        if epoch + 1 == args.epochs:
            torch.save({"model_state": model.state_dict(),
                        "args": vars(args),
                        "epoch": epoch + 1},
                       out / "last.pt")
        with open(out / "history.json", "w") as f:
            json.dump(history, f, indent=2)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True,
                   help="path to .h5 file with dataset 'images' of shape (N,H,W)")
    p.add_argument("--out", default="checkpoints/run01")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--base", type=int, default=16,
                   help="channel multiplier (model size scales as base^2)")
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--lambda-ssim", type=float, default=0.2)
    p.add_argument("--workers", type=int, default=-1,
                   help="DataLoader workers (-1 = auto: up to 4)")
    p.add_argument("--max-samples", type=int, default=0,
                   help="Use at most this many galaxies (0 = all)")
    p.add_argument("--image-size", type=int, default=0,
                   help="Downsample to this side length via striding (0 = native)")
    p.add_argument("--fast", action="store_true",
                   help="Faster uv-mask synthesis (fewer hour-angle samples)")
    p.add_argument("--metadata", action="store_true",
                   help="Metadata-aware mode: feed (dirty, beam, uv-mask) as "
                        "a 3-channel input instead of the dirty image alone")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
