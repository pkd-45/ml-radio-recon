"""
clean_baseline.py
-----------------
A textbook implementation of Hogbom CLEAN (Hogbom 1974) used as the
reference algorithm to compare the learned reconstruction against.

Algorithm:
  - Start with the dirty image I_d and dirty beam B (both centred).
  - Repeat for at most ``n_iter`` iterations:
       p     <- pixel with largest |residual|
       delta <- gain * residual[p]
       components[p] += delta
       residual -= delta * shift(B, p)
    until |residual|_max < threshold * |I_d|_max  or  n_iter reached.
  - Restore: convolve the component image with a Gaussian "clean beam"
    fitted to the central lobe of B, then add the final residuals.

This is intentionally the simplest possible CLEAN; the goal is a
faithful, transparent baseline, not a competitive imager.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit


# ---------------------------------------------------------------------------
# Clean-beam fitting
# ---------------------------------------------------------------------------

def _gaussian_2d(coords, amp, x0, y0, sig_x, sig_y, theta):
    x, y = coords
    ct, st = np.cos(theta), np.sin(theta)
    xr = ct * (x - x0) + st * (y - y0)
    yr = -st * (x - x0) + ct * (y - y0)
    return amp * np.exp(-(xr ** 2 / (2 * sig_x ** 2)
                          + yr ** 2 / (2 * sig_y ** 2)))


def fit_clean_beam(dirty_beam: np.ndarray, fit_radius: int = 8
                   ) -> tuple[np.ndarray, dict]:
    """Fit a 2D Gaussian to the central lobe of the dirty beam.

    Returns the (n, n) Gaussian "clean beam" with the same shape as
    ``dirty_beam`` and a small dict of fit parameters.
    """
    n = dirty_beam.shape[0]
    c = n // 2
    s = slice(c - fit_radius, c + fit_radius + 1)
    sub = dirty_beam[s, s]
    y, x = np.mgrid[:sub.shape[0], :sub.shape[1]]
    p0 = (sub.max(), fit_radius, fit_radius, 2.0, 2.0, 0.0)
    try:
        popt, _ = curve_fit(_gaussian_2d, (x.ravel(), y.ravel()),
                            sub.ravel(), p0=p0, maxfev=5000)
    except Exception:
        popt = p0
    # build full-size clean beam centred in the grid
    yy, xx = np.mgrid[:n, :n]
    clean_beam = _gaussian_2d((xx, yy),
                              1.0, c, c,
                              abs(popt[3]), abs(popt[4]), popt[5])
    return clean_beam, {"sig_x": float(abs(popt[3])),
                         "sig_y": float(abs(popt[4])),
                         "theta": float(popt[5])}


# ---------------------------------------------------------------------------
# Hogbom CLEAN
# ---------------------------------------------------------------------------

def _shift_2d(arr: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Shift ``arr`` by integer (dy, dx) with zero fill, no wrap."""
    out = np.zeros_like(arr)
    ny, nx = arr.shape
    y0_src = max(0, -dy)
    y1_src = min(ny, ny - dy)
    x0_src = max(0, -dx)
    x1_src = min(nx, nx - dx)
    y0_dst = y0_src + dy
    y1_dst = y1_src + dy
    x0_dst = x0_src + dx
    x1_dst = x1_src + dx
    out[y0_dst:y1_dst, x0_dst:x1_dst] = arr[y0_src:y1_src, x0_src:x1_src]
    return out


def hogbom_clean(dirty_image: np.ndarray,
                 dirty_beam: np.ndarray,
                 gain: float = 0.1,
                 n_iter: int = 500,
                 threshold_frac: float = 0.02,
                 clean_beam: np.ndarray | None = None
                 ) -> dict:
    """Run scalar Hogbom CLEAN on a single image.

    Returns a dict with keys
        ``restored``    : the CLEAN-restored image
        ``components``  : the accumulated point-source model
        ``residual``    : the dirty-image residual at termination
        ``n_used``      : number of iterations actually executed
    """
    residual = dirty_image.copy().astype(np.float64)
    components = np.zeros_like(residual)

    beam_centre = np.array(dirty_beam.shape) // 2
    abs_init = np.abs(residual).max()
    stop = threshold_frac * abs_init

    used = 0
    for k in range(n_iter):
        idx = np.unravel_index(np.argmax(np.abs(residual)), residual.shape)
        peak = residual[idx]
        if abs(peak) < stop:
            break
        delta = gain * peak
        components[idx] += delta
        dy = idx[0] - beam_centre[0]
        dx = idx[1] - beam_centre[1]
        residual -= delta * _shift_2d(dirty_beam, dy, dx)
        used = k + 1

    if clean_beam is None:
        clean_beam, _ = fit_clean_beam(dirty_beam)
    # convolve components with the clean beam (FFT convolution)
    cb_ft = np.fft.fft2(np.fft.ifftshift(clean_beam))
    comp_ft = np.fft.fft2(components)
    restored_components = np.real(np.fft.ifft2(cb_ft * comp_ft))
    restored = restored_components + residual
    return {"restored": restored,
            "components": components,
            "residual": residual,
            "n_used": used}


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from uv_simulator import make_uv_mask, dirty_image

    n = 256
    yy, xx = np.mgrid[:n, :n]
    img = np.zeros((n, n))
    for cy, cx, amp, sig in [(120, 130, 1.0, 4.0),
                             (90, 160, 0.6, 6.0),
                             (170, 110, 0.4, 3.0)]:
        img += amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sig ** 2))

    mask = make_uv_mask(n_pix=n)
    I_d, beam = dirty_image(img, mask)
    out = hogbom_clean(I_d, beam, gain=0.1, n_iter=500)
    print(f"CLEAN used {out['n_used']} iterations")
    print(f"restored: min={out['restored'].min():.3f}  "
          f"max={out['restored'].max():.3f}")
    print(f"residual rms = {out['residual'].std():.4f}")
