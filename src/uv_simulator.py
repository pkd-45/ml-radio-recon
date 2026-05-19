"""
uv_simulator.py
---------------
Forward model for sparse-aperture interferometric imaging.

Given a true sky image I(l,m), this module
   1. computes its 2D FFT V(u,v) (the model visibilities),
   2. samples V on a binary uv-mask M(u,v) constructed to mimic the
      sparse, "core + extended" coverage of an ALMA-like array with
      Earth-rotation aperture synthesis,
   3. returns the corresponding dirty image
           I_d  =  IFFT(M . V)
      and the dirty beam (PSF)
           B    =  IFFT(M).

The mask construction is deliberately schematic: the goal is to
expose the network to the characteristic interferometric inverse
problem (band-limited + non-uniformly sampled Fourier plane), not to
reproduce any specific ALMA configuration. A second, fully physical
mode using literal antenna (x,y) layouts + hour-angle tracks is
included for users who want to plug in real antenna positions.

All transforms use the unitary FFT convention so that Parseval holds
and amplitudes are preserved across image / visibility space.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Mask construction
# ---------------------------------------------------------------------------

def _log_spiral_antennas(n_ant: int, r_min: float, r_max: float,
                        n_arms: int = 3, seed: int = 0) -> np.ndarray:
    """Place ``n_ant`` antennas along ``n_arms`` log-spiral arms.

    Returns an array of shape (n_ant, 2) of (x, y) positions in metres.
    This is a stand-in for ALMA's compact/extended log-spiral pads."""
    rng = np.random.default_rng(seed)
    arms = rng.integers(0, n_arms, size=n_ant)
    # exponentially spaced radii so we sample both short and long baselines
    radii = r_min * (r_max / r_min) ** rng.uniform(0.0, 1.0, size=n_ant)
    base_angle = 2.0 * np.pi * arms / n_arms
    # tighter pitch -> more "armlike", looser -> more uniform
    pitch = 0.7
    theta = base_angle + pitch * np.log(radii / r_min)
    # small jitter so antennas do not lie on perfect curves
    theta = theta + rng.normal(0.0, 0.05, size=n_ant)
    x = radii * np.cos(theta)
    y = radii * np.sin(theta)
    return np.stack([x, y], axis=-1)


def _baselines_with_rotation(ant_xy: np.ndarray,
                             hour_angles_rad: np.ndarray,
                             declination_rad: float) -> np.ndarray:
    """Project array baselines onto the (u,v) plane for a tracked source.

    Implements the standard rotation-synthesis formulae assuming a
    co-planar east-north array. Returns an array of shape
    (n_pairs * n_times, 2) of (u, v) coordinates in metres.
    """
    n = len(ant_xy)
    i, j = np.triu_indices(n, k=1)
    dx = ant_xy[i, 0] - ant_xy[j, 0]      # east-west separation
    dy = ant_xy[i, 1] - ant_xy[j, 1]      # north-south separation

    ha = hour_angles_rad[None, :]
    cos_ha, sin_ha = np.cos(ha), np.sin(ha)
    sin_dec = np.sin(declination_rad)

    u = dx[:, None] * sin_ha + dy[:, None] * cos_ha
    v = (-dx[:, None] * cos_ha + dy[:, None] * sin_ha) * sin_dec
    uv = np.stack([u.ravel(), v.ravel()], axis=-1)
    # interferometric visibilities are Hermitian -> include conjugates
    uv = np.concatenate([uv, -uv], axis=0)
    return uv


def make_uv_mask(n_pix: int = 256,
                 n_ant: int = 32,
                 r_min: float = 15.0,
                 r_max: float = 400.0,
                 declination_deg: float = -30.0,
                 hour_angle_range_h: tuple[float, float] = (-2.0, 2.0),
                 n_times: int = 60,
                 wavelength_m: float = 1.3e-3,
                 fov_arcsec: float = 10.0,
                 seed: int = 0) -> np.ndarray:
    """Build a binary uv-mask in image-FFT pixel coordinates.

    Parameters chosen as defaults give an ALMA Band-6-ish flavour:
    16-32 antennas with baselines from ~15 m to ~400 m, 4-hour track
    on a southern source, 60 sample times. The mask is returned as a
    centred ``(n_pix, n_pix)`` boolean array suitable for direct
    multiplication with ``np.fft.fftshift(np.fft.fft2(image))``.
    """
    ant = _log_spiral_antennas(n_ant, r_min, r_max, seed=seed)
    ha = np.deg2rad(15.0) * np.linspace(*hour_angle_range_h, n_times)
    uv_m = _baselines_with_rotation(ant, ha, np.deg2rad(declination_deg))
    uv_lambda = uv_m / wavelength_m

    # pixel scale in the uv-plane is set by the image field of view:
    #   du = 1 / FOV[rad]
    fov_rad = np.deg2rad(fov_arcsec / 3600.0)
    du = 1.0 / fov_rad
    centre = n_pix // 2
    u_pix = np.round(uv_lambda[:, 0] / du).astype(int) + centre
    v_pix = np.round(uv_lambda[:, 1] / du).astype(int) + centre

    mask = np.zeros((n_pix, n_pix), dtype=bool)
    good = (u_pix >= 0) & (u_pix < n_pix) & (v_pix >= 0) & (v_pix < n_pix)
    mask[v_pix[good], u_pix[good]] = True
    # always retain the DC component so total flux is recoverable
    mask[centre, centre] = True
    return mask


# ---------------------------------------------------------------------------
# Forward model
# ---------------------------------------------------------------------------

def fft2c(x: np.ndarray) -> np.ndarray:
    """Centred, unitary 2D FFT."""
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x), norm="ortho"))


def ifft2c(x: np.ndarray) -> np.ndarray:
    """Centred, unitary 2D inverse FFT."""
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x), norm="ortho"))


def dirty_image(image: np.ndarray, mask: np.ndarray,
                noise_sigma: float = 0.0,
                rng: np.random.Generator | None = None
                ) -> tuple[np.ndarray, np.ndarray]:
    """Apply the interferometric forward model to ``image``.

    Returns ``(I_dirty, beam)`` where ``beam`` is the dirty beam (PSF)
    corresponding to ``mask``. Both outputs are real-valued (the small
    imaginary residuals from the discrete FFT are dropped).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    V = fft2c(image)
    if noise_sigma > 0:
        noise = (rng.normal(size=V.shape) + 1j * rng.normal(size=V.shape)) \
                * noise_sigma / np.sqrt(2)
        V = V + noise
    V_sampled = V * mask
    I_d = ifft2c(V_sampled).real
    beam = ifft2c(mask.astype(np.complex128)).real
    # normalise the beam so that its peak is unity (Hogbom convention)
    beam = beam / beam.max()
    return I_d, beam


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # synthetic test image: a few off-centre Gaussians on a low background
    n = 256
    yy, xx = np.mgrid[:n, :n]
    img = np.zeros((n, n))
    for cy, cx, amp, sig in [(120, 130, 1.0, 4.0),
                             (90, 160, 0.6, 6.0),
                             (170, 110, 0.4, 3.0)]:
        img += amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sig ** 2))

    mask = make_uv_mask(n_pix=n)
    print(f"uv-mask fill fraction: {mask.mean():.4f}")
    I_d, beam = dirty_image(img, mask, noise_sigma=0.0)
    print(f"true image:   min={img.min():.3f}  max={img.max():.3f}")
    print(f"dirty image:  min={I_d.min():.3f}  max={I_d.max():.3f}")
    print(f"dirty beam:   peak={beam.max():.3f}  FWHM ~ "
          f"{(beam > 0.5 * beam.max()).sum() ** 0.5:.1f} pix")
