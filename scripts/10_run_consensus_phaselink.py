import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.phase.consensus_common_offset import consensus_common_offset_phaselink


def load_canonical(path):
    z = np.load(path, allow_pickle=True)

    meta = pd.DataFrame({
        "view": z["view"].astype(str),
        "array": z["array"].astype(str),
        "point": z["point"].astype(str),
        "f0": z["f0"].astype(float),
        "frame": z["frame"].astype(int),
        "timestamp": z["timestamp"].astype(float),
        "system_ns": z["system_ns"].astype(float),
        "rssi_dbm": z["rssi_dbm"].astype(float),
        "noise_floor_dbm": z["noise_floor_dbm"].astype(float),
    })

    return z["csi"].astype(np.complex64), z["sub_idx"].astype(np.int32), meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/canonical/canonical_k53_packets.npz")
    parser.add_argument("--output", default="outputs/phaselink/consensus_common_offset_v2.npz")
    parser.add_argument("--metrics", default="outputs/metrics/consensus_common_offset_v2_metrics.csv")
    parser.add_argument("--min-frames", type=int, default=2)
    parser.add_argument("--n-iter", type=int, default=6)
    parser.add_argument("--trim-quantile", type=float, default=0.20)
    args = parser.parse_args()

    csi, sub_idx, meta = load_canonical(args.input)

    out_csi = np.empty_like(csi, dtype=np.complex64)
    delta = np.zeros(csi.shape[0], dtype=np.float32)
    confidence = np.zeros(csi.shape[0], dtype=np.float32)

    metric_rows = []

    grouped = meta.groupby(["view", "array", "point", "f0"], sort=False)

    for gkey, gdf in grouped:
        idx = gdf.index.to_numpy()
        csi_g = csi[idx]

        if len(idx) < args.min_frames:
            out_csi[idx] = csi_g
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
                "init_ref_idx": -1,
            })
            continue

        res = consensus_common_offset_phaselink(
            csi_g,
            n_iter=args.n_iter,
            trim_quantile=args.trim_quantile,
        )

        out_csi[idx] = res["csi_corr"]
        delta[idx] = res["delta"]
        confidence[idx] = res["packet_confidence"]

        metric_rows.append({
            "view": gkey[0],
            "array": gkey[1],
            "point": gkey[2],
            "f0": gkey[3],
            "n": len(idx),
            "status": "ok",
            "coh_before": float(res["coh_before"][0]),
            "coh_after": float(res["coh_after"][0]),
            "coh_gain": float(res["coh_gain"][0]),
            "residual_std": float(res["residual_std"][0]),
            "confidence": float(res["group_confidence"][0]),
            "init_ref_idx": int(res["init_ref_idx"][0]),
        })

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics).parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        args.output,
        csi_corr=out_csi,
        csi_raw=csi,
        sub_idx=sub_idx,
        delta=delta,
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
    print("CONSENSUS COMMON-OFFSET TPL-ToF V2")
    print("=" * 80)
    print("Saved:", args.output)
    print("Saved metrics:", args.metrics)
    print(metrics["status"].value_counts())

    print("\n[GLOBAL]")
    print("num ok groups:", len(ok))
    print("median coherence before:", ok["coh_before"].median(skipna=True))
    print("median coherence after:", ok["coh_after"].median(skipna=True))
    print("median coherence gain:", ok["coh_gain"].median(skipna=True))
    print("median residual std:", ok["residual_std"].median(skipna=True))
    print("mean confidence:", ok["confidence"].mean(skipna=True))

    print("\n[BY VIEW / ARRAY]")
    print(
        ok.groupby(["view", "array"])[
            ["coh_before", "coh_after", "coh_gain", "residual_std", "confidence"]
        ].median()
    )


if __name__ == "__main__":
    main()