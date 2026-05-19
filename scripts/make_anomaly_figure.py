"""
make_anomaly_figure.py
----------------------
Demonstrate reconstruction-based anomaly detection.

A reconstruction model trained on normal galaxies is run on a test
set that is half normal and half anomalous (RFI-like streaks, spurious
point sources, ring artefacts). For each image we compute the
data-consistency residual (see ``src/anomaly.py``): the model's
reconstruction is re-observed through the same uv-coverage and
compared with the actual dirty image. Anomalous flux that the
normal-galaxy model fails to reproduce shows up as a bright, localised
residual.

Outputs:
  - a 4-row figure (2 normal + 2 anomalous galaxies), columns
    truth / dirty / reconstruction / data-consistency residual;
  - a panel with the anomaly-score distributions and ROC-AUC;
  - a JSON summary with per-sample scores and the AUC.

Usage
-----
    python scripts/make_anomaly_figure.py \
        --data data/galaxies.h5 \
        --checkpoint checkpoints/dirty_only/best.pt \
        --out results/anomaly.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from anomaly import data_consistency_residual, roc_auc          # noqa: E402
from dataset import SparseAperturePairs                          # noqa: E402
from model import UNetAttn                                       # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--n-test", type=int, default=40)
    p.add_argument("--anomaly-frac", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--fast", action="store_true")
    p.add_argument("--out", default="results/anomaly.png")
    p.add_argument("--json-out", default="results/anomaly_eval.json")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with h5py.File(args.data, "r") as f:
        images = f["images"][...]
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(images), size=min(args.n_test, len(images)),
                     replace=False)
    test_imgs = images[idx]

    ds = SparseAperturePairs(test_imgs, augment=False,
                             noise_sigma_range=(0.005, 0.005),
                             fast=args.fast,
                             anomaly_frac=args.anomaly_frac,
                             seed=args.seed)

    ckpt = torch.load(args.checkpoint, map_location=device,
                      weights_only=False)
    margs = ckpt.get("args", {})
    metadata = bool(margs.get("metadata", False))
    model = UNetAttn(in_ch=3 if metadata else 1,
                     base=margs.get("base", 16),
                     num_heads=margs.get("heads", 4),
                     residual=True).to(device).eval()
    model.load_state_dict(ckpt["model_state"])

    # ---- score every test image ----
    records = []
    for i in range(len(ds)):
        s = ds[i]
        dirty = s["dirty"].numpy()[0]
        truth = s["truth"].numpy()[0]
        beam = s["beam"].numpy()[0]
        mask = s["mask"].numpy()[0]
        label = int(s["anomaly"])
        res = data_consistency_residual(model, dirty, mask, beam=beam,
                                        metadata=metadata, device=device)
        records.append({"i": i, "label": label,
                        "kind": s.get("anomaly_kind", ""),
                        "score": res["score"],
                        "truth": truth, "dirty": dirty,
                        "recon": res["recon"],
                        "residual": res["residual"]})

    scores_norm = np.array([r["score"] for r in records if r["label"] == 0])
    scores_anom = np.array([r["score"] for r in records if r["label"] == 1])
    auc = roc_auc(scores_norm, scores_anom)
    print(f"anomaly detection: {len(scores_norm)} normal, "
          f"{len(scores_anom)} anomalous")
    print(f"  mean score  normal={scores_norm.mean():.4f}  "
          f"anomalous={scores_anom.mean():.4f}")
    print(f"  ROC-AUC = {auc:.3f}")

    # ---- pick 2 normal + 2 anomalous (distinct anomaly kinds) ----
    normal = [r for r in records if r["label"] == 0][:2]
    anom_all = [r for r in records if r["label"] == 1]
    anomalous, seen_kinds = [], set()
    for r in anom_all:
        if r["kind"] not in seen_kinds:
            anomalous.append(r)
            seen_kinds.add(r["kind"])
        if len(anomalous) == 2:
            break
    while len(anomalous) < 2 and len(anomalous) < len(anom_all):
        anomalous.append(anom_all[len(anomalous)])
    show = normal + anomalous
    row_tags = (["normal", "normal"]
                + [f"anomalous ({r['kind']})" for r in anomalous])

    fig = plt.figure(figsize=(13, 3.0 * len(show) + 0.5),
                     constrained_layout=True)
    gs = fig.add_gridspec(len(show), 5,
                          width_ratios=[1, 1, 1, 1, 1.15])

    # shared residual colour scale so normal vs anomalous contrast is honest
    res_vmax = max(float(np.percentile(rec["residual"], 99.5))
                   for rec in show)

    col_titles = ["truth", "dirty", "reconstruction",
                  "data-consistency residual"]
    for r, (rec, tag) in enumerate(zip(show, row_tags)):
        vmax = float(rec["truth"].max())
        panels = [rec["truth"], rec["dirty"], rec["recon"]]
        for c, (name, img) in enumerate(zip(col_titles[:3], panels)):
            ax = fig.add_subplot(gs[r, c])
            ax.imshow(img, origin="lower", cmap="magma",
                      vmin=-0.02 * vmax, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(name, fontsize=11)
            if c == 0:
                ax.set_ylabel(f"{tag}\n(score {rec['score']:.3f})",
                              fontsize=9)
        # residual map on a colour scale shared across all rows
        ax = fig.add_subplot(gs[r, 3])
        ax.imshow(rec["residual"], origin="lower", cmap="inferno",
                  vmin=0.0, vmax=res_vmax)
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            ax.set_title(col_titles[3], fontsize=11)

    # ---- score-distribution panel spanning the right column ----
    ax = fig.add_subplot(gs[:, 4])
    bins = np.linspace(
        min(scores_norm.min(), scores_anom.min()),
        max(scores_norm.max(), scores_anom.max()), 16)
    ax.hist(scores_norm, bins=bins, orientation="horizontal",
            alpha=0.7, label="normal", color="#3b6fb0")
    ax.hist(scores_anom, bins=bins, orientation="horizontal",
            alpha=0.7, label="anomalous", color="#c4452f")
    ax.set_xlabel("count", fontsize=9)
    ax.set_ylabel("anomaly score (99th-pct residual)", fontsize=9)
    ax.set_title(f"score separation\nROC-AUC = {auc:.3f}", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.tick_params(labelsize=8)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print(f"wrote {out}")

    summary = {
        "n_normal": int(len(scores_norm)),
        "n_anomalous": int(len(scores_anom)),
        "mean_score_normal": float(scores_norm.mean()),
        "mean_score_anomalous": float(scores_anom.mean()),
        "roc_auc": float(auc),
        "per_sample": [{"i": r["i"], "label": r["label"],
                        "score": r["score"]} for r in records],
    }
    jout = Path(args.json_out)
    jout.parent.mkdir(parents=True, exist_ok=True)
    with open(jout, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {jout}")


if __name__ == "__main__":
    main()
