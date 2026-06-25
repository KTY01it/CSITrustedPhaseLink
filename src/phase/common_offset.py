import numpy as np
from typing import Dict


EPS = 1e-8


def circular_coherence(phases: np.ndarray, axis=0) -> np.ndarray:
    return np.abs(np.mean(np.exp(1j * phases), axis=axis))


def group_coherence(csi_group: np.ndarray) -> float:
    """
    csi_group: [N, K]
    coherence theo frame, rồi lấy mean theo subcarrier.
    """
    phases = np.angle(csi_group)
    gamma_k = circular_coherence(phases, axis=0)
    return float(np.mean(gamma_k))


def choose_reference_packet(csi_group: np.ndarray) -> int:
    """
    Chọn packet reference bằng median amplitude lớn nhất.
    """
    amp_score = np.median(np.abs(csi_group), axis=1)
    return int(np.argmax(amp_score))


def estimate_common_offset(h: np.ndarray, href: np.ndarray, w: np.ndarray = None) -> float:
    """
    Ước lượng common phase offset sao cho h * exp(-j delta) gần href.
    Không fit slope. Không xóa ToF-related frequency slope.
    """
    if w is None:
        w = np.ones_like(h.real, dtype=np.float64)

    z = np.sum(w * h * np.conj(href))
    if np.abs(z) < EPS:
        return 0.0

    return float(np.angle(z))


def apply_common_offset(h: np.ndarray, delta: float) -> np.ndarray:
    return h * np.exp(-1j * delta)


def residual_phase_std(csi_corr: np.ndarray, href: np.ndarray) -> float:
    """
    Residual phase sau khi align common offset tới reference.
    """
    res = np.angle(csi_corr * np.conj(href[None, :]))
    return float(np.std(res))


def phase_link_group_common_offset(csi_group: np.ndarray) -> Dict[str, np.ndarray]:
    """
    csi_group: [N, 53]

    Output:
      csi_corr
      delta
      confidence
      coh_before
      coh_after
      residual_std
      ref_idx
    """
    if csi_group.ndim != 2:
        raise ValueError("csi_group must have shape [N, K].")

    n, k = csi_group.shape

    if n < 2:
        coh = group_coherence(csi_group)
        return {
            "csi_corr": csi_group.copy().astype(np.complex64),
            "delta": np.zeros(n, dtype=np.float32),
            "confidence": np.zeros(n, dtype=np.float32),
            "coh_before": np.array([coh], dtype=np.float32),
            "coh_after": np.array([coh], dtype=np.float32),
            "residual_std": np.array([np.nan], dtype=np.float32),
            "ref_idx": np.array([0], dtype=np.int32),
        }

    ref_idx = choose_reference_packet(csi_group)
    href = csi_group[ref_idx]

    # Subcarrier weight đơn giản theo median amplitude trong group.
    amp = np.abs(csi_group)
    w = np.median(amp, axis=0).astype(np.float64)
    w = w / (np.max(w) + EPS)

    coh_before = group_coherence(csi_group)

    deltas = np.zeros(n, dtype=np.float32)
    csi_corr = np.empty_like(csi_group, dtype=np.complex64)

    for i in range(n):
        delta = estimate_common_offset(csi_group[i], href, w=w)
        deltas[i] = delta
        csi_corr[i] = apply_common_offset(csi_group[i], delta).astype(np.complex64)

    coh_after = group_coherence(csi_corr)
    res_std = residual_phase_std(csi_corr, href)

    gain = max(0.0, coh_after - coh_before)

    # Confidence v1: chưa phải learned confidence, chỉ là physics diagnostic confidence.
    base_conf = 0.5 * coh_after + 0.3 * gain + 0.2 * np.exp(-res_std)
    base_conf = float(np.clip(base_conf, 0.0, 1.0))
    confidence = np.full(n, base_conf, dtype=np.float32)

    return {
        "csi_corr": csi_corr,
        "delta": deltas,
        "confidence": confidence,
        "coh_before": np.array([coh_before], dtype=np.float32),
        "coh_after": np.array([coh_after], dtype=np.float32),
        "residual_std": np.array([res_std], dtype=np.float32),
        "ref_idx": np.array([ref_idx], dtype=np.int32),
    }