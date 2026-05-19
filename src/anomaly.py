"""
anomaly.py
----------
Reconstruction-based anomaly detection for the interferometric imager.

The reconstruction network is trained on *normal* galaxies only. It
therefore carries a learned prior over normal galaxy morphology. When
an out-of-distribution feature is present in the data — radio-frequency
interference, a calibration artefact, a spurious or transient source —
the network tends to suppress it or reconstruct it as something more
galaxy-like.

We exploit that with a **data-consistency residual**, which requires no
anomaly labels and no extra training:

    1. reconstruct           recon = f(dirty[, beam, mask])
    2. re-observe the recon  dirty_recon = forward_model(recon, mask)
    3. residual map          R = | dirty_recon - dirty |

For a normal source the reconstruction explains the observed dirty
image, so ``dirty_recon`` matches ``dirty`` and R is small everywhere.
For an anomaly the network's normal-galaxy reconstruction fails to
account for the anomalous flux, so R is large and *localised* on the
anomaly. The scalar anomaly score is a high percentile of R.

This is the radio analogue of inspecting residual visibilities / a
residual map after imaging — a quantity radio astronomers already
trust — turned into an unsupervised detector.
"""

from __future__ import annotations

import numpy as np
import torch

from uv_simulator import dirty_image


# ---------------------------------------------------------------------------
# Out-of-distribution feature injection (for evaluation only)
# ---------------------------------------------------------------------------

def inject_anomaly(image: np.ndarray, kind: str = "streak",
                   rng: np.random.Generator | None = None,
                   strength: float = 0.6) -> np.ndarray:
    """Add one out-of-distribution feature to a normalised truth image.

    Parameters
    ----------
    image
        2D truth image, values in [0, 1].
    kind
        ``streak``  : a thin bright linear feature (RFI-like).
        ``point``   : a spurious compact bright source away from centre.
        ``ring``    : a thin ring (calibration-artefact-like).
    rng
        Random generator.
    strength
        Peak amplitude of the injected feature relative to image max.

    Returns
    -------
    A new image (the input is not modified), renormalised to [0, 1].
    """
    if rng is None:
        rng = np.random.default_rng(0)
    out = np.array(image, dtype=np.float32, copy=True)
    n = out.shape[0]
    yy, xx = np.mgrid[:n, :n]

    if kind == "streak":
        angle = rng.uniform(0, np.pi)
        c = n / 2 + rng.uniform(-n / 6, n / 6)
        # signed distance from a line through the centre region
        dist = (np.cos(angle) * (xx - n / 2)
                + np.sin(angle) * (yy - n / 2) - (c - n / 2))
        width = rng.uniform(1.0, 2.0)
        out += strength * np.exp(-(dist ** 2) / (2 * width ** 2))

    elif kind == "point":
        cx = rng.uniform(0.15 * n, 0.85 * n)
        cy = rng.uniform(0.15 * n, 0.85 * n)
        sig = rng.uniform(1.0, 2.0)
        out += strength * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2)
                                 / (2 * sig ** 2))

    elif kind == "ring":
        cx, cy = n / 2, n / 2
        r0 = rng.uniform(0.20 * n, 0.38 * n)
        r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        width = rng.uniform(1.0, 2.0)
        out += strength * np.exp(-((r - r0) ** 2) / (2 * width ** 2))

    else:
        raise ValueError(f"unknown anomaly kind: {kind}")

    lo, hi = float(out.min()), float(out.max())
    if hi - lo > 1e-8:
        out = (out - lo) / (hi - lo)
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Data-consistency residual
# ---------------------------------------------------------------------------

def data_consistency_residual(model: torch.nn.Module,
                              dirty: np.ndarray,
                              mask: np.ndarray,
                              beam: np.ndarray | None = None,
                              metadata: bool = False,
                              device: torch.device | str = "cpu"
                              ) -> dict:
    """Compute the data-consistency residual map for one image.

    Parameters
    ----------
    model
        Trained reconstruction network.
    dirty, mask, beam
        The dirty image, uv-mask and dirty beam (all (H, W) arrays).
        ``beam`` is required only when ``metadata`` is True.
    metadata
        If True, the model expects a 3-channel (dirty, beam, mask)
        input; otherwise a 1-channel (dirty) input.
    device
        Torch device.

    Returns
    -------
    dict with keys ``recon``, ``dirty_recon``, ``residual``, ``score``.
    """
    model.eval()
    d = torch.from_numpy(dirty.astype(np.float32))[None, None]
    if metadata:
        if beam is None:
            raise ValueError("beam is required when metadata=True")
        b = torch.from_numpy(beam.astype(np.float32))[None, None]
        m = torch.from_numpy(mask.astype(np.float32))[None, None]
        inp = torch.cat([d, b, m], dim=1)
    else:
        inp = d
    inp = inp.to(device)

    with torch.no_grad():
        recon = model(inp).cpu().numpy()[0, 0]
    recon = np.clip(recon, 0.0, 1.0)

    # re-observe the reconstruction through the same uv-coverage
    dirty_recon, _ = dirty_image(recon, mask, noise_sigma=0.0)

    residual = np.abs(dirty_recon - dirty)
    score = float(np.percentile(residual, 99.0))
    return {"recon": recon,
            "dirty_recon": dirty_recon,
            "residual": residual,
            "score": score}


def anomaly_score(residual: np.ndarray, percentile: float = 99.0) -> float:
    """Scalar anomaly score: a high percentile of the residual map."""
    return float(np.percentile(residual, percentile))


# ---------------------------------------------------------------------------
# Detection evaluation
# ---------------------------------------------------------------------------

def roc_auc(scores_normal: np.ndarray,
            scores_anomalous: np.ndarray) -> float:
    """ROC-AUC for separating two score distributions, no sklearn dep.

    Equivalent to the Mann-Whitney U statistic normalised by n*m:
    the probability that a random anomalous score exceeds a random
    normal score (ties counted as 0.5).
    """
    a = np.asarray(scores_anomalous, dtype=float)
    b = np.asarray(scores_normal, dtype=float)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    wins = 0.0
    for s in a:
        wins += np.sum(s > b) + 0.5 * np.sum(s == b)
    return float(wins / (len(a) * len(b)))


if __name__ == "__main__":
    # smoke test: injection + a no-op residual
    rng = np.random.default_rng(0)
    n = 128
    yy, xx = np.mgrid[:n, :n]
    base = np.exp(-((xx - n / 2) ** 2 + (yy - n / 2) ** 2) / (2 * 12.0 ** 2))
    base = base / base.max()
    for kind in ("streak", "point", "ring"):
        a = inject_anomaly(base, kind=kind, rng=rng)
        print(f"{kind:7s}  injected; image range [{a.min():.2f}, {a.max():.2f}]")
    # roc sanity: anomalous scores clearly higher -> AUC ~ 1
    sn = rng.normal(0.0, 1.0, size=200)
    sa = rng.normal(3.0, 1.0, size=200)
    print(f"roc_auc (well separated) = {roc_auc(sn, sa):.3f}")
