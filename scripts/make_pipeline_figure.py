"""
make_pipeline_figure.py
-----------------------
A single figure showing the *whole* interferometric ML pipeline, from
the instrument to the reconstruction:

    row 1  (measurement) : array layout | uv-coverage | dirty beam | sky truth
    row 2  (reconstruction): dirty image | CLEAN | ML | ML residual

Unlike the headline figure, this one exposes the radio-interferometry
physics --- the antenna array, the Fourier-plane sampling produced by
Earth-rotation synthesis, and the resulting point-spread function ---
which is the part that signals interferometric literacy.

Usage
-----
    python scripts/make_pipeline_figure.py \
        --data data/galaxies.h5 \
        --checkpoint checkpoints/meta/best.pt \
        --out results/pipeline.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from clean_baseline import hogbom_clean
from metrics import psnr, ssim
from model import UNetAttn
from uv_simulator import (_baselines_with_rotation, _log_spiral_antennas,
                          dirty_image, make_uv_mask)


# --- array / observation parameters (kept consistent across all panels) ---
ARRAY = dict(n_ant=32, r_min=15.0, r_max=400.0)
OBS = dict(declination_deg=-30.0, hour_angle_range_h=(-2.0, 2.0),
           n_times=60, wavelength_m=1.3e-3, fov_arcsec=10.0)


def build_observation(n_pix, seed):
    """Reproduce make_uv_mask's geometry while also returning the
    antenna positions and continuous uv-track, so every panel of the
    figure corresponds to the same physical configuration."""
    ant = _log_spiral_antennas(ARRAY["n_ant"], ARRAY["r_min"],
                               ARRAY["r_max"], seed=seed)
    ha = np.deg2rad(15.0) * np.linspace(*OBS["hour_angle_range_h"],
                                        OBS["n_times"])
    uv_m = _baselines_with_rotation(ant, ha,
                                    np.deg2rad(OBS["declination_deg"]))
    uv_klambda = uv_m / OBS["wavelength_m"] / 1e3        # kilo-lambda
    mask = make_uv_mask(n_pix=n_pix, n_ant=ARRAY["n_ant"],
                        r_min=ARRAY["r_min"], r_max=ARRAY["r_max"],
                        declination_deg=OBS["declination_deg"],
                        hour_angle_range_h=OBS["hour_angle_range_h"],
                        n_times=OBS["n_times"],
                        wavelength_m=OBS["wavelength_m"],
                        fov_arcsec=OBS["fov_arcsec"], seed=seed)
    return ant, uv_klambda, mask


def pick_extended_galaxy(images, rng):
    """Choose a visually extended galaxy (large second moment) so the
    figure shows the regime where the learned prior matters most."""
    idx = rng.choice(len(images), size=min(40, len(images)), replace=False)
    yy, xx = np.mgrid[0:images.shape[1], 0:images.shape[2]]
    best, best_spread = idx[0], -1.0
    for i in idx:
        im = images[i].astype(np.float64)
        s = im.sum()
        if s <= 0:
            continue
        cx, cy = (im * xx).sum() / s, (im * yy).sum() / s
        spread = np.sqrt((im * ((xx - cx) ** 2 + (yy - cy) ** 2)).sum() / s)
        if spread > best_spread:
            best, best_spread = i, spread
    return int(best)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--checkpoint", required=True,
                   help="metadata-aware model checkpoint")
    p.add_argument("--galaxy", type=int, default=395,
                   help="galaxy index to display; -1 picks the most "
                        "extended galaxy in a random subset")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--clean-iter", type=int, default=150)
    p.add_argument("--noise", type=float, default=0.01)
    p.add_argument("--out", default="results/pipeline.png")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)

    with h5py.File(args.data, "r") as f:
        images = f["images"][...]
    if args.galaxy is not None and args.galaxy >= 0:
        gi = args.galaxy % len(images)
    else:
        gi = pick_extended_galaxy(images, rng)
    truth = images[gi].astype(np.float64)
    truth = truth / truth.max()
    n_pix = truth.shape[0]

    # the observation geometry is keyed to the galaxy index so the
    # figure is fully reproducible from --galaxy alone
    obs_seed = gi

    # --- forward model -----------------------------------------------------
    ant, uv_klambda, mask = build_observation(n_pix, seed=obs_seed)
    dirty, beam = dirty_image(truth, mask, noise_sigma=args.noise,
                              rng=np.random.default_rng(obs_seed))

    # --- CLEAN -------------------------------------------------------------
    clean = hogbom_clean(dirty, beam, gain=0.1, n_iter=args.clean_iter,
                         threshold_frac=0.02)["restored"]
    clean = np.clip(clean, 0, 1)

    # --- ML (metadata-aware) ----------------------------------------------
    ckpt = torch.load(args.checkpoint, map_location=device,
                      weights_only=False)
    margs = ckpt.get("args", {})
    metadata = bool(margs.get("metadata", False))
    model = UNetAttn(in_ch=3 if metadata else 1,
                     base=margs.get("base", 16),
                     num_heads=margs.get("heads", 4),
                     residual=True).to(device).eval()
    model.load_state_dict(ckpt["model_state"])

    def to_t(a):
        return torch.from_numpy(a.astype(np.float32))[None, None]

    if metadata:
        inp = torch.cat([to_t(dirty), to_t(beam),
                         to_t(mask.astype(np.float32))], dim=1)
    else:
        inp = to_t(dirty)
    with torch.no_grad():
        ml = np.clip(model(inp.to(device)).cpu().numpy()[0, 0], 0, 1)

    residual = np.abs(truth - ml)

    # --- figure ------------------------------------------------------------
    sky = "magma"
    fig = plt.figure(figsize=(13.5, 7.0))
    gs = GridSpec(2, 4, figure=fig, hspace=0.28, wspace=0.16,
                  left=0.04, right=0.97, top=0.90, bottom=0.07)

    def stamp(ax, img, title, cmap=sky, metric=None):
        vmax = float(np.percentile(img, 99.7)) or 1.0
        ax.imshow(img, origin="lower", cmap=cmap, vmin=0, vmax=vmax)
        ax.set_title(title, fontsize=10.5, pad=4)
        ax.set_xticks([]); ax.set_yticks([])
        if metric:
            ax.text(0.035, 0.965, metric, transform=ax.transAxes,
                    ha="left", va="top", fontsize=8.4, color="white",
                    bbox=dict(facecolor="black", alpha=0.45,
                              edgecolor="none", pad=2))

    # (0,0) antenna array
    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(ant[:, 0], ant[:, 1], s=22, c="#2c6fbb",
               edgecolors="white", linewidths=0.4)
    ax.set_title(f"Array layout ({ARRAY['n_ant']} antennas)",
                 fontsize=10.5, pad=4)
    ax.set_xlabel("east [m]", fontsize=8.5)
    ax.set_ylabel("north [m]", fontsize=8.5)
    ax.tick_params(labelsize=7.5)
    ax.set_aspect("equal")
    ax.grid(alpha=0.25, linewidth=0.5)

    # (0,1) uv-coverage, coloured by baseline length
    ax = fig.add_subplot(gs[0, 1])
    r = np.hypot(uv_klambda[:, 0], uv_klambda[:, 1])
    sc = ax.scatter(uv_klambda[:, 0], uv_klambda[:, 1], s=1.4, c=r,
                    cmap="viridis", alpha=0.7, linewidths=0)
    ax.set_title("uv-coverage", fontsize=10.5, pad=4)
    ax.set_xlabel(r"$u$ [k$\lambda$]", fontsize=8.5)
    ax.set_ylabel(r"$v$ [k$\lambda$]", fontsize=8.5)
    ax.tick_params(labelsize=7.5)
    ax.set_aspect("equal")
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(r"baseline [k$\lambda$]", fontsize=7.8)
    cb.ax.tick_params(labelsize=7)

    # (0,2) dirty beam, zoomed to the core
    ax = fig.add_subplot(gs[0, 2])
    c = n_pix // 2
    half = max(24, n_pix // 6)
    beam_crop = beam[c - half:c + half, c - half:c + half]
    ax.imshow(beam_crop, origin="lower", cmap="cividis")
    ax.set_title("Dirty beam / PSF (core)", fontsize=10.5, pad=4)
    ax.set_xticks([]); ax.set_yticks([])

    # (0,3) sky truth
    stamp(fig.add_subplot(gs[0, 3]), truth, "Sky model (truth)")

    # (1,0) dirty image
    stamp(fig.add_subplot(gs[1, 0]), np.clip(dirty, 0, None),
          "Dirty image")
    # (1,1) CLEAN
    stamp(fig.add_subplot(gs[1, 1]), clean, "Hogbom CLEAN",
          metric=f"PSNR {psnr(truth, clean):.1f}\nSSIM {ssim(truth, clean):.3f}")
    # (1,2) ML
    stamp(fig.add_subplot(gs[1, 2]), ml,
          "ML reconstruction (metadata-aware)",
          metric=f"PSNR {psnr(truth, ml):.1f}\nSSIM {ssim(truth, ml):.3f}")
    # (1,3) residual
    ax = fig.add_subplot(gs[1, 3])
    im = ax.imshow(residual, origin="lower", cmap="inferno",
                   vmin=0, vmax=float(np.percentile(residual, 99.5)) or 1)
    ax.set_title(r"ML residual $|{\rm truth}-{\rm ML}|$",
                 fontsize=10.5, pad=4)
    ax.set_xticks([]); ax.set_yticks([])
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=7)

    fig.suptitle("ML image reconstruction for sparse-aperture radio "
                 "interferometry \u2014 end-to-end pipeline",
                 fontsize=12.5, fontweight="bold", y=0.975)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"wrote {out}  (galaxy index {gi})")


if __name__ == "__main__":
    main()
