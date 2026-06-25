import torch
import torch.nn as nn
import torch.nn.functional as F


EPS = 1e-8


def complex_normalize_rows(x):
    """
    x: complex tensor [N, K]
    """
    den = torch.linalg.norm(x, dim=1, keepdim=True).clamp_min(EPS)
    return x / den


def complex_normalize_vec(x):
    den = torch.linalg.norm(x).clamp_min(EPS)
    return x / den


def phase_unit(x):
    """
    x / |x|, dùng để tính circular coherence.
    """
    return x / torch.abs(x).clamp_min(EPS)


def group_coherence_torch(csi):
    """
    csi: complex tensor [N, K]
    """
    u = phase_unit(csi)
    gamma_k = torch.abs(torch.mean(u, dim=0))
    return torch.mean(gamma_k)


def representative_vector(csi):
    """
    Đại diện phase vector của group.
    """
    x = complex_normalize_rows(csi)
    rep = torch.mean(x, dim=0)
    return complex_normalize_vec(rep)


def best_aligned_similarity(rep_a, rep_b):
    """
    Similarity giữa hai representative vectors sau khi bù best common phase.
    """
    z = torch.sum(rep_b * torch.conj(rep_a))
    phasor = torch.conj(z / torch.abs(z).clamp_min(EPS))
    rep_b_aligned = rep_b * phasor

    num = torch.abs(torch.sum(rep_b_aligned * torch.conj(rep_a)))
    den = torch.linalg.norm(rep_a) * torch.linalg.norm(rep_b_aligned)
    return num / den.clamp_min(EPS)


class TPLToFNet(nn.Module):
    """
    Learned reliability layer for ToF-preserving phase-link.

    AI học:
      - subcarrier reliability w_k
      - packet reliability alpha_i

    Correction vẫn là common offset:
      H_corr_i(k) = H_i(k) * exp(-j delta_i)

    Không output delta_i(k), nên không xóa ToF slope.
    """

    def __init__(self, num_bands=5, num_subcarriers=53, hidden=32, n_iter=6):
        super().__init__()
        self.num_bands = num_bands
        self.num_subcarriers = num_subcarriers
        self.n_iter = n_iter

        # Learnable subcarrier logits theo band
        self.subcarrier_logits = nn.Parameter(torch.zeros(num_bands, num_subcarriers))

        # Packet reliability MLP
        # features:
        # amp_mean, amp_std, amp_max, phase_coherence_over_k
        self.packet_mlp = nn.Sequential(
            nn.Linear(4, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def packet_features(self, csi):
        amp = torch.abs(csi)
        amp_mean = torch.mean(amp, dim=1)
        amp_std = torch.std(amp, dim=1)
        amp_max = torch.max(amp, dim=1).values

        # phase coherence across subcarriers trong từng packet
        u = phase_unit(csi)
        phase_coh = torch.abs(torch.mean(u, dim=1))

        feat = torch.stack([amp_mean, amp_std, amp_max, phase_coh], dim=1)

        # normalize trong group để tránh scale quá lớn
        mu = torch.mean(feat, dim=0, keepdim=True)
        sig = torch.std(feat, dim=0, keepdim=True).clamp_min(1e-6)
        feat = (feat - mu) / sig

        return feat

    def get_weights(self, csi, band_id):
        """
        Return:
          w_k: [K]
          alpha_i: [N]
        """
        w = torch.sigmoid(self.subcarrier_logits[band_id])
        w = w / torch.max(w).clamp_min(EPS)

        feat = self.packet_features(csi)
        logits = self.packet_mlp(feat).squeeze(-1)

        alpha = torch.sigmoid(logits)
        alpha = alpha / torch.sum(alpha).clamp_min(EPS)

        return w, alpha

    def estimate_delta_phasor(self, h, ref, w):
        """
        h: [N, K]
        ref: [K]
        w: [K]

        z_i = sum_k w_k H_i(k) conj(ref(k))
        correction phasor = exp(-j angle(z_i)) = conj(z_i / |z_i|)
        """
        z = torch.sum(w[None, :] * h * torch.conj(ref)[None, :], dim=1)
        unit = z / torch.abs(z).clamp_min(EPS)
        return torch.conj(unit)

    def forward(self, csi, band_id):
        """
        csi: complex tensor [N, 53]
        band_id: int
        """
        if not torch.is_complex(csi):
            raise ValueError("csi must be complex tensor.")

        w, alpha = self.get_weights(csi, band_id)

        # Init reference bằng weighted normalized mean của raw CSI
        x0 = complex_normalize_rows(csi)
        ref = torch.sum(alpha[:, None] * x0, dim=0)
        ref = complex_normalize_vec(ref)

        csi_corr = csi

        for _ in range(self.n_iter):
            phasor = self.estimate_delta_phasor(csi, ref, w)
            csi_corr = csi * phasor[:, None]

            x = complex_normalize_rows(csi_corr)
            ref = torch.sum(alpha[:, None] * x, dim=0)
            ref = complex_normalize_vec(ref)

        # Final correction
        phasor = self.estimate_delta_phasor(csi, ref, w)
        csi_corr = csi * phasor[:, None]

        # confidence diagnostic
        coh_before = group_coherence_torch(csi)
        coh_after = group_coherence_torch(csi_corr)
        coh_gain = coh_after - coh_before

        return {
            "csi_corr": csi_corr,
            "w": w,
            "alpha": alpha,
            "coh_before": coh_before,
            "coh_after": coh_after,
            "coh_gain": coh_gain,
        }


def split_loss(model, csi, band_id):
    """
    Self-supervised split consistency loss.
    """
    n = csi.shape[0]
    perm = torch.randperm(n, device=csi.device)
    mid = n // 2

    a = csi[perm[:mid]]
    b = csi[perm[mid:]]

    if a.shape[0] < 2 or b.shape[0] < 2:
        return torch.tensor(0.0, device=csi.device)

    out_a = model(a, band_id)
    out_b = model(b, band_id)

    rep_a = representative_vector(out_a["csi_corr"])
    rep_b = representative_vector(out_b["csi_corr"])

    sim = best_aligned_similarity(rep_a, rep_b)

    return 1.0 - sim

def consensus_residual_loss(csi_corr):
    """
    Differentiable residual-to-consensus loss.

    Mục tiêu:
      - giảm residual phase spread sau correction
      - không cho model chỉ tối đa coherence nhưng residual còn rộng

    Correction vẫn chỉ là common offset, nên không xóa ToF slope trực tiếp.
    """
    x = complex_normalize_rows(csi_corr)
    ref = torch.mean(x, dim=0)
    ref = complex_normalize_vec(ref)

    residual = torch.angle(x * torch.conj(ref)[None, :])

    # std theo subcarrier cho từng packet, rồi lấy mean
    per_packet_std = torch.std(residual, dim=1)

    return torch.mean(per_packet_std)