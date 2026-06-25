import numpy as np
from typing import Dict

from src.phase.common_offset import (
    group_coherence,
    estimate_common_offset,
    apply_common_offset,
)

EPS = 1e-8


def _weighted_norm(x, w):
    return np.sqrt(np.sum(w * np.abs(x) ** 2) + EPS)


def _alignment_score(h_corr, ref, w):
    """
    Cosine similarity phức có trọng số.
    """
    num = np.abs(np.sum(w * h_corr * np.conj(ref)))
    den = _weighted_norm(h_corr, w) * _weighted_norm(ref, w)
    return float(num / (den + EPS))


def _phase_residual_std(h_corr, ref):
    res = np.angle(h_corr * np.conj(ref))
    return float(np.std(res))


def _normalize_ref(ref):
    den = np.linalg.norm(ref)
    if den < EPS:
        return ref
    return ref / den


def _initial_reference(csi_group):
    """
    Chọn init bằng packet có median amplitude lớn nhất.
    Sau đó normalize để reference chỉ đóng vai trò phase/template.
    """
    amp_score = np.median(np.abs(csi_group), axis=1)
    idx = int(np.argmax(amp_score))
    ref = csi_group[idx].astype(np.complex128)
    return _normalize_ref(ref), idx


def _subcarrier_weights(csi_group):
    """
    Weight theo median amplitude trên group.
    Không dùng phase slope.
    """
    amp = np.abs(csi_group)
    w = np.median(amp, axis=0).astype(np.float64)
    w = w / (np.max(w) + EPS)
    return w


def consensus_common_offset_phaselink(
    csi_group: np.ndarray,
    n_iter: int = 6,
    trim_quantile: float = 0.20,
) -> Dict[str, np.ndarray]:
    """
    Consensus-based common-offset phase-link.

    Input:
      csi_group: [N, K]

    Output:
      csi_corr
      delta
      packet_confidence
      group metrics

    Important:
      - Chỉ sửa common offset per packet.
      - Không fit slope theo subcarrier.
      - Không xóa ToF-related phase slope.
    """
    if csi_group.ndim != 2:
        raise ValueError("csi_group must have shape [N, K].")

    n, k = csi_group.shape

    if n < 2:
        coh = group_coherence(csi_group)
        return {
            "csi_corr": csi_group.copy().astype(np.complex64),
            "delta": np.zeros(n, dtype=np.float32),
            "packet_confidence": np.zeros(n, dtype=np.float32),
            "coh_before": np.array([coh], dtype=np.float32),
            "coh_after": np.array([coh], dtype=np.float32),
            "coh_gain": np.array([0.0], dtype=np.float32),
            "residual_std": np.array([np.nan], dtype=np.float32),
            "group_confidence": np.array([0.0], dtype=np.float32),
            "init_ref_idx": np.array([0], dtype=np.int32),
        }

    csi = csi_group.astype(np.complex128)
    w = _subcarrier_weights(csi)

    coh_before = group_coherence(csi)

    ref, init_ref_idx = _initial_reference(csi)

    delta = np.zeros(n, dtype=np.float64)
    csi_corr = np.empty_like(csi, dtype=np.complex128)
    scores = np.ones(n, dtype=np.float64)
    residuals = np.zeros(n, dtype=np.float64)

    for _ in range(n_iter):
        for i in range(n):
            d = estimate_common_offset(csi[i], ref, w=w)
            delta[i] = d
            csi_corr[i] = apply_common_offset(csi[i], d)
            scores[i] = _alignment_score(csi_corr[i], ref, w)
            residuals[i] = _phase_residual_std(csi_corr[i], ref)

        # Robust consensus update: bỏ bớt packet alignment kém.
        thr = np.quantile(scores, trim_quantile)
        keep = scores >= thr

        if keep.sum() < max(2, int(0.2 * n)):
            keep = np.ones(n, dtype=bool)

        # packet weight = alignment score, giảm ảnh hưởng outlier
        pw = scores[keep]
        pw = pw / (np.sum(pw) + EPS)

        # normalize từng packet trước khi lấy mean để tránh amplitude dominate
        x = csi_corr[keep]
        x_norm = x / (np.linalg.norm(x, axis=1, keepdims=True) + EPS)

        ref_new = np.sum(x_norm * pw[:, None], axis=0)
        ref = _normalize_ref(ref_new)

    # Final correction với reference cuối
    for i in range(n):
        d = estimate_common_offset(csi[i], ref, w=w)
        delta[i] = d
        csi_corr[i] = apply_common_offset(csi[i], d)
        scores[i] = _alignment_score(csi_corr[i], ref, w)
        residuals[i] = _phase_residual_std(csi_corr[i], ref)

    coh_after = group_coherence(csi_corr)
    coh_gain = coh_after - coh_before

    residual_std = float(np.median(residuals))

    # confidence packet-level
    packet_conf = (
        0.55 * scores +
        0.25 * np.exp(-residuals) +
        0.20 * max(0.0, coh_gain)
    )
    packet_conf = np.clip(packet_conf, 0.0, 1.0)

    group_conf = float(np.mean(packet_conf))

    return {
        "csi_corr": csi_corr.astype(np.complex64),
        "delta": delta.astype(np.float32),
        "packet_confidence": packet_conf.astype(np.float32),
        "coh_before": np.array([coh_before], dtype=np.float32),
        "coh_after": np.array([coh_after], dtype=np.float32),
        "coh_gain": np.array([coh_gain], dtype=np.float32),
        "residual_std": np.array([residual_std], dtype=np.float32),
        "group_confidence": np.array([group_conf], dtype=np.float32),
        "init_ref_idx": np.array([init_ref_idx], dtype=np.int32),
    }