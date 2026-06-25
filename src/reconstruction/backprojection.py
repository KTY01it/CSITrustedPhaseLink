from __future__ import annotations

import numpy as np


C_LIGHT = 299_792_458.0
DEFAULT_SUBCARRIER_SPACING_HZ = 312_500.0


def subcarrier_frequencies_hz(f0_mhz: float, sub_idx: np.ndarray, spacing_hz: float = DEFAULT_SUBCARRIER_SPACING_HZ) -> np.ndarray:
    """Return absolute subcarrier frequencies in Hz.

    f_k = f0 * 1e6 + sub_idx[k] * spacing_hz.
    """
    return float(f0_mhz) * 1e6 + np.asarray(sub_idx, dtype=np.float64) * float(spacing_hz)


def make_grid(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
    step: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create a Cartesian grid and flattened voxel coordinates."""
    if step <= 0:
        raise ValueError("grid step must be positive")
    xs = np.arange(x_min, x_max + 0.5 * step, step, dtype=np.float64)
    ys = np.arange(y_min, y_max + 0.5 * step, step, dtype=np.float64)
    zs = np.arange(z_min, z_max + 0.5 * step, step, dtype=np.float64)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    return xs, ys, zs, pts


def bistatic_delay_s(tx_xyz: np.ndarray, rx_xyz: np.ndarray, points_xyz: np.ndarray) -> np.ndarray:
    tx = np.asarray(tx_xyz, dtype=np.float64).reshape(1, 3)
    rx = np.asarray(rx_xyz, dtype=np.float64).reshape(1, 3)
    pts = np.asarray(points_xyz, dtype=np.float64)
    d_tx = np.linalg.norm(pts - tx, axis=1)
    d_rx = np.linalg.norm(pts - rx, axis=1)
    return (d_tx + d_rx) / C_LIGHT


def response_for_group(
    hbar: np.ndarray,
    sub_idx: np.ndarray,
    f0_mhz: float,
    tx_xyz: np.ndarray,
    rx_xyz: np.ndarray,
    points_xyz: np.ndarray,
    subcarrier_weight: np.ndarray | None = None,
    spacing_hz: float = DEFAULT_SUBCARRIER_SPACING_HZ,
    chunk_voxels: int = 50_000,
    use_absolute_frequency: bool = False,
) -> np.ndarray:
    """Compute non-normalized bistatic matched-filter response for one group.

    hbar: complex fused CSI vector [K].
    The returned response is |sum_k w_k H_k exp(j 2pi f_k tau(x))|^2.

    By default, the steering uses subcarrier-offset frequencies rather than the
    large carrier term. This is usually more stable when different measurement
    positions do not share an absolute carrier-phase reference. Set
    use_absolute_frequency=True only for controlled diagnostics.
    """
    h = np.asarray(hbar, dtype=np.complex128).reshape(-1)
    sub = np.asarray(sub_idx, dtype=np.float64).reshape(-1)
    if h.shape[0] != sub.shape[0]:
        raise ValueError(f"hbar length {h.shape[0]} != sub_idx length {sub.shape[0]}")

    if subcarrier_weight is None:
        w = np.ones_like(sub, dtype=np.float64)
    else:
        w = np.asarray(subcarrier_weight, dtype=np.float64).reshape(-1)
        if w.shape[0] != sub.shape[0]:
            raise ValueError(f"weight length {w.shape[0]} != sub_idx length {sub.shape[0]}")

    if np.max(np.abs(w)) > 0:
        w = w / (np.max(np.abs(w)) + 1e-12)

    freqs = subcarrier_frequencies_hz(f0_mhz, sub, spacing_hz=spacing_hz)
    if not use_absolute_frequency:
        freqs = sub * float(spacing_hz)

    tau = bistatic_delay_s(tx_xyz, rx_xyz, points_xyz)
    out = np.empty(points_xyz.shape[0], dtype=np.float64)

    hw = w.astype(np.complex128) * h
    two_pi = 2.0 * np.pi

    for start in range(0, points_xyz.shape[0], int(chunk_voxels)):
        end = min(start + int(chunk_voxels), points_xyz.shape[0])
        phase = two_pi * tau[start:end, None] * freqs[None, :]
        steering = np.exp(1j * phase)
        amp = steering @ hw
        out[start:end] = np.abs(amp) ** 2

    return out


def normalize_volume(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    mn = float(np.nanmin(v))
    mx = float(np.nanmax(v))
    return (v - mn) / (mx - mn + eps)
