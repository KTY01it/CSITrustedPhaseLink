import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

from src.training.group_dataset import load_canonical_npz, BAND_TO_ID
from src.phase.tpl_tof_net import TPLToFNet, group_coherence_torch
from src.phase.common_offset import residual_phase_std

def learned_consensus_residual_std(csi_corr_np):
    """
    Residual std công bằng hơn cho learned model:
    so từng packet với representative consensus vector của chính corrected group.
    """
    eps = 1e-8

    x = csi_corr_np.astype(np.complex128)

    # Normalize từng packet để amplitude không dominate.
    den = np.linalg.norm(x, axis=1, keepdims=True)
    den = np.maximum(den, eps)
    x_norm = x / den

    # Consensus representative vector.
    ref = np.mean(x_norm, axis=0)
    ref_den = np.linalg.norm(ref)
    if ref_den < eps:
        return np.nan
    ref = ref / ref_den

    # Per-packet phase residual to consensus reference.
    residual = np.angle(x_norm * np.conj(ref[None, :]))
    per_packet_std = np.std(residual, axis=1)

    return float(np.median(per_packet_std))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/canonical/canonical_k53_packets.npz")
    parser.add_argument("--ckpt", default="outputs/phaselink/tpl_tof_net_v3.pt")
    parser.add_argument("--output", default="outputs/phaselink/tpl_tof_net_v3.npz")
    parser.add_argument("--metrics", default="outputs/metrics/tpl_tof_net_v3_metrics.csv")
    parser.add_argument("--min_frames", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    csi, sub_idx, meta = load_canonical_npz(args.input)

    ckpt = torch.load(args.ckpt, map_location=device)
    model_args = ckpt.get("args", {})

    model = TPLToFNet(
        num_bands=5,
        num_subcarriers=53,
        hidden=int(model_args.get("hidden", 32)),
        n_iter=int(model_args.get("n_iter", 6)),
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    out_csi = np.empty_like(csi, dtype=np.complex64)
    confidence = np.zeros(csi.shape[0], dtype=np.float32)

    metric_rows = []

    grouped = meta.groupby(["view", "array", "point", "f0"], sort=False)

    with torch.no_grad():
        for gkey, gdf in grouped:
            idx = gdf.index.to_numpy()
            csi_g_np = csi[idx]

            if len(idx) < args.min_frames:
                out_csi[idx] = csi_g_np
                metric_rows.append({
                    "view": gkey[0],
                    "array": gkey[1],
                    "point": gkey[2],
                    "f0": gkey[3],
                    "n": len(idx),
                    "status": "skip_too_few_frames",
                    "coh_before": np.nan,
                    "coh_after": np.nan,
                    "coh_gain": np.nan,
                    "residual_std": np.nan,
                    "confidence": 0.0,
                })
                continue

            f0 = float(gkey[3])
            if f0 not in BAND_TO_ID:
                out_csi[idx] = csi_g_np
                continue

            band_id = BAND_TO_ID[f0]

            csi_t = torch.from_numpy(csi_g_np).to(device).to(torch.complex64)
            out = model(csi_t, band_id)

            csi_corr = out["csi_corr"].detach().cpu().numpy().astype(np.complex64)
            
            res_std = learned_consensus_residual_std(csi_corr)
            
            alpha = out["alpha"].detach().cpu().numpy().astype(np.float32)

            out_csi[idx] = csi_corr

            # packet confidence proxy từ alpha scale về [0,1]
            alpha_norm = alpha / (alpha.max() + 1e-8)
            confidence[idx] = alpha_norm

            coh_before = float(out["coh_before"].detach().cpu())
            coh_after = float(out["coh_after"].detach().cpu())

            metric_rows.append({
                "view": gkey[0],
                "array": gkey[1],
                "point": gkey[2],
                "f0": gkey[3],
                "n": len(idx),
                "status": "ok",
                "coh_before": coh_before,
                "coh_after": coh_after,
                "coh_gain": coh_after - coh_before,
                "residual_std": float(res_std),
                "confidence": float(np.mean(alpha_norm)),
            })

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics).parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        args.output,
        csi_corr=out_csi,
        csi_raw=csi,
        sub_idx=sub_idx,
        confidence=confidence,
        view=meta["view"].to_numpy(dtype=object),
        array=meta["array"].to_numpy(dtype=object),
        point=meta["point"].to_numpy(dtype=object),
        f0=meta["f0"].to_numpy(np.float32),
        frame=meta["frame"].to_numpy(np.int32),
    )

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(args.metrics, index=False)

    ok = metrics[metrics["status"] == "ok"]

    print("=" * 80)
    print("APPLY LEARNED TPL-ToF-Net v3")
    print("=" * 80)
    print("Saved:", args.output)
    print("Saved metrics:", args.metrics)
    print(metrics["status"].value_counts())

    print("\n[GLOBAL]")
    print("num ok groups:", len(ok))
    print("median coherence before:", ok["coh_before"].median(skipna=True))
    print("median coherence after:", ok["coh_after"].median(skipna=True))
    print("median coherence gain:", ok["coh_gain"].median(skipna=True))
    print("mean confidence:", ok["confidence"].mean(skipna=True))
    print("median residual std:", ok["residual_std"].median(skipna=True))


    print("\n[BY VIEW / ARRAY]")
    print(
        ok.groupby(["view", "array"])[
            ["coh_before", "coh_after", "coh_gain", "residual_std", "confidence"]
        ].median()
    )


if __name__ == "__main__":
    main()