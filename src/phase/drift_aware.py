import numpy as np
from typing import Dict, Optional

from src.phase.common_offset import (
    estimate_common_offset,
    apply_common_offset,
    group_coherence,
    choose_reference_packet,
    residual_phase_std,
)

EPS = 1e-8


def _valid_time_axis(time_vec: Optional[np.ndarray], n: int) -> np.ndarray:
    """
    Chuẩn hóa time axis về [-0.5, 0.5].
    Nếu time không hợp lệ thì dùng index packet.
    """
    if time_vec is None:
        t = np.arange(n, dtype=np.float64)
    else:
        t = np.asarray(time_vec, dtype=np.float64)
        if t.shape[0] != n:
            t = np.arange(n, dtype=np.float64)

    finite = np.isfinite(t)
    if finite.sum() < max(3, n // 3):
        t = np.arange(n, dtype=np.float64)
    else:
        med = np.nanmedian(t[finite])
        t = np.where(finite, t, med)

    span = float(np.max(t) - np.min(t))
    if span <= EPS:
        t = np.arange(n, dtype=np.float64)
        span = float(np.max(t) - np.min(t))

    if span <= EPS:
        return np.zeros(n, dtype=np.float64)

    return (t - np.min(t)) / span - 0.5


def _robust_linear_fit(t: np.ndarray, y: np.ndarray, n_iter: int = 8):
    """
    Fit y ≈ a + b t bằng IRLS Huber-like.
    t, y: [N]
    Return:
      y_hat, coef [a,b], residual, robust_scale
    """
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    X = np.stack([np.ones_like(t), t], axis=1)
    w = np.ones_like(y, dtype=np.float64)

    coef = np.zeros(2, dtype=np.float64)

    for _ in range(n_iter):
        W = np.sqrt(np.maximum(w, EPS))[:, None]
        Xw = X * W
        yw = y * W[:, 0]

        try:
            coef = np.linalg.lstsq(Xw, yw, rcond=None)[0]
        except np.linalg.LinAlgError:
            coef = np.zeros(2, dtype=np.float64)

        pred = X @ coef
        resid = y - pred

        mad = np.median(np.abs(resid - np.median(resid))) + EPS
        scale = 1.4826 * mad + EPS

        # Huber weight
        r = np.abs(resid) / (2.5 * scale)
        w = np.where(r <= 1.0, 1.0, 1.0 / np.maximum(r, EPS))

    pred = X @ coef
    resid = y - pred
    mad = np.median(np.abs(resid - np.median(resid))) + EPS
    scale = 1.4826 * mad + EPS

    return pred, coef, resid, float(scale)


def _raw_common_offsets_to_reference(csi_group: np.ndarray, ref_idx: int):
    """
    Estimate raw common offset of each packet relative to reference packet.
    """
    href = csi_group[ref_idx]

    amp = np.abs(csi_group)
    w = np.median(amp, axis=0).astype(np.float64)
    w = w / (np.max(w) + EPS)

    n = csi_group.shape[0]
    deltas = np.zeros(n, dtype=np.float64)

    for i in range(n):
        deltas[i] = estimate_common_offset(csi_group[i], href, w=w)

    return deltas.astype(np.float64)


def phase_link_group_drift_aware(
    csi_group: np.ndarray,
    time_vec: Optional[np.ndarray] = None,
    mode: str = "hybrid",
    residual_clip_rad: float = 0.75,
) -> Dict[str, np.ndarray]:
    """
    Drift-aware ToF-preserving phase-link.

    Input:
      csi_group: [N, K]
      time_vec: [N], optional. Prefer SystemNS or Timestamp.
      mode:
        - "trend": chỉ sửa drift trend a+b*t
        - "hybrid": sửa drift trend + clipped residual
        - "raw": tương đương common-offset v1

    Important:
      Chỉ sửa common phase offset per packet.
      Không fit slope theo subcarrier.
      Không xóa ToF-related frequency slope.
    """
    if csi_group.ndim != 2:
        raise ValueError("csi_group must have shape [N, K].")

    n, k = csi_group.shape

    if n < 2:
        coh = group_coherence(csi_group)
        return {
            "csi_corr": csi_group.copy().astype(np.complex64),
            "delta_raw": np.zeros(n, dtype=np.float32),
            "delta_fit": np.zeros(n, dtype=np.float32),
            "delta_used": np.zeros(n, dtype=np.float32),
            "confidence": np.zeros(n, dtype=np.float32),
            "coh_before": np.array([coh], dtype=np.float32),
            "coh_after": np.array([coh], dtype=np.float32),
            "coh_gain": np.array([0.0], dtype=np.float32),
            "residual_std": np.array([np.nan], dtype=np.float32),
            "drift_residual_std": np.array([np.nan], dtype=np.float32),
            "fit_a": np.array([0.0], dtype=np.float32),
            "fit_b": np.array([0.0], dtype=np.float32),
            "ref_idx": np.array([0], dtype=np.int32),
        }

    ref_idx = choose_reference_packet(csi_group)
    href = csi_group[ref_idx]

    coh_before = group_coherence(csi_group)

    raw_delta = _raw_common_offsets_to_reference(csi_group, ref_idx=ref_idx)

    t = _valid_time_axis(time_vec, n=n)
    order = np.argsort(t)

    t_sorted = t[order]
    delta_sorted = raw_delta[order]

    # unwrap theo time order để fit drift liên tục
    delta_unwrapped_sorted = np.unwrap(delta_sorted)

    delta_fit_sorted, coef, drift_resid_sorted, drift_scale = _robust_linear_fit(
        t_sorted,
        delta_unwrapped_sorted,
    )

    if mode == "trend":
        delta_used_sorted = delta_fit_sorted
    elif mode == "raw":
        delta_used_sorted = delta_unwrapped_sorted
    elif mode == "hybrid":
        clipped_resid = np.clip(
            drift_resid_sorted,
            -float(residual_clip_rad),
            float(residual_clip_rad),
        )
        delta_used_sorted = delta_fit_sorted + clipped_resid
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # restore original order
    delta_fit = np.zeros(n, dtype=np.float64)
    delta_used = np.zeros(n, dtype=np.float64)
    drift_resid = np.zeros(n, dtype=np.float64)

    delta_fit[order] = delta_fit_sorted
    delta_used[order] = delta_used_sorted
    drift_resid[order] = drift_resid_sorted

    csi_corr = np.empty_like(csi_group, dtype=np.complex64)
    for i in range(n):
        csi_corr[i] = apply_common_offset(csi_group[i], delta_used[i]).astype(np.complex64)

    coh_after = group_coherence(csi_corr)
    coh_gain = coh_after - coh_before
    res_std = residual_phase_std(csi_corr, href)
    drift_resid_std = float(np.std(drift_resid))

    # confidence v2: coherence + gain + drift fit quality
    fit_quality = float(np.exp(-drift_resid_std))
    residual_quality = float(np.exp(-res_std))

    base_conf = (
        0.40 * coh_after +
        0.25 * max(0.0, coh_gain) +
        0.20 * fit_quality +
        0.15 * residual_quality
    )
    base_conf = float(np.clip(base_conf, 0.0, 1.0))
    confidence = np.full(n, base_conf, dtype=np.float32)

    return {
        "csi_corr": csi_corr,
        "delta_raw": raw_delta.astype(np.float32),
        "delta_fit": delta_fit.astype(np.float32),
        "delta_used": delta_used.astype(np.float32),
        "confidence": confidence,
        "coh_before": np.array([coh_before], dtype=np.float32),
        "coh_after": np.array([coh_after], dtype=np.float32),
        "coh_gain": np.array([coh_gain], dtype=np.float32),
        "residual_std": np.array([res_std], dtype=np.float32),
        "drift_residual_std": np.array([drift_resid_std], dtype=np.float32),
        "fit_a": np.array([coef[0]], dtype=np.float32),
        "fit_b": np.array([coef[1]], dtype=np.float32),
        "ref_idx": np.array([ref_idx], dtype=np.int32),
    }