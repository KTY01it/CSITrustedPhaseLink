import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd


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

    return meta


def safe_span_ms(x):
    x = pd.to_numeric(x, errors="coerce")
    x = x[np.isfinite(x)]
    if len(x) <= 1:
        return np.nan
    return float((x.max() - x.min()) / 1e6)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/canonical/canonical_k53_packets.npz")
    parser.add_argument("--outdir", default="outputs/metrics")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    meta = load_canonical(args.input)

    print("=" * 80)
    print("APERTURE SYNCHRONIZATION / AOA READINESS AUDIT")
    print("=" * 80)

    print("\n[BASIC COUNTS BY VIEW / ARRAY / F0]")
    basic = meta.groupby(["view", "array", "f0"]).agg(
        n_packets=("point", "size"),
        n_points=("point", "nunique"),
        n_frames=("frame", "nunique"),
        system_ns_valid=("system_ns", lambda x: np.isfinite(pd.to_numeric(x, errors="coerce")).sum()),
    ).reset_index()
    print(basic)
    basic.to_csv(outdir / "aperture_basic_counts.csv", index=False)

    # Frame coverage: within each view/array/f0/frame, how many points are observed?
    frame_cov = meta.groupby(["view", "array", "f0", "frame"]).agg(
        n_packets=("point", "size"),
        n_points=("point", "nunique"),
        system_ns_span_ms=("system_ns", safe_span_ms),
        timestamp_span=("timestamp", lambda x: float(pd.to_numeric(x, errors="coerce").max() - pd.to_numeric(x, errors="coerce").min()) if len(x) > 1 else np.nan),
    ).reset_index()

    frame_cov.to_csv(outdir / "aperture_frame_coverage.csv", index=False)

    print("\n[FRAME COVERAGE SUMMARY]")
    cov_summary = frame_cov.groupby(["view", "array", "f0"]).agg(
        n_frame_groups=("frame", "size"),
        median_points_per_frame=("n_points", "median"),
        p95_points_per_frame=("n_points", lambda x: float(np.percentile(x, 95))),
        max_points_per_frame=("n_points", "max"),
        median_system_ns_span_ms=("system_ns_span_ms", "median"),
        p95_system_ns_span_ms=("system_ns_span_ms", lambda x: float(np.nanpercentile(x, 95)) if np.isfinite(x).any() else np.nan),
    ).reset_index()

    print(cov_summary)
    cov_summary.to_csv(outdir / "aperture_frame_coverage_summary.csv", index=False)

    # Good synchronous frame candidates:
    # At least 4 points in one frame and system_ns span small.
    # Thresholds are diagnostic only.
    candidate = frame_cov[
        (frame_cov["n_points"] >= 4) &
        (
            frame_cov["system_ns_span_ms"].isna() |
            (frame_cov["system_ns_span_ms"] <= 10.0)
        )
    ].copy()

    print("\n[SYNCHRONOUS FRAME CANDIDATES]")
    print("candidate frame groups:", len(candidate))
    print(candidate.head(30))

    candidate.to_csv(outdir / "aperture_sync_frame_candidates.csv", index=False)

    # Per array decision
    decisions = []
    for key, g in cov_summary.groupby(["view", "array", "f0"]):
        view, array, f0 = key
        max_pts = float(g["max_points_per_frame"].iloc[0])
        p95_pts = float(g["p95_points_per_frame"].iloc[0])
        med_span = float(g["median_system_ns_span_ms"].iloc[0]) if np.isfinite(g["median_system_ns_span_ms"].iloc[0]) else np.nan
        p95_span = float(g["p95_system_ns_span_ms"].iloc[0]) if np.isfinite(g["p95_system_ns_span_ms"].iloc[0]) else np.nan

        if max_pts >= 8 and (not np.isfinite(p95_span) or p95_span <= 10.0):
            decision = "POSSIBLE_FRAME_SYNCHRONOUS_APERTURE"
        elif max_pts >= 8:
            decision = "FRAME_INDEX_SHARED_BUT_TIME_SPAN_LARGE_CHECK_SESSION"
        elif max_pts >= 4:
            decision = "WEAK_PARTIAL_APERTURE_ONLY"
        else:
            decision = "NO_FRAME_SYNCHRONOUS_APERTURE"

        decisions.append({
            "view": view,
            "array": array,
            "f0": f0,
            "max_points_per_frame": max_pts,
            "p95_points_per_frame": p95_pts,
            "median_system_ns_span_ms": med_span,
            "p95_system_ns_span_ms": p95_span,
            "decision": decision,
        })

    decision_df = pd.DataFrame(decisions)
    print("\n[AOA READINESS DECISION BY VIEW / ARRAY / F0]")
    print(decision_df)
    decision_df.to_csv(outdir / "aperture_aoa_readiness_decision.csv", index=False)

    print("\nSaved:")
    print(outdir / "aperture_basic_counts.csv")
    print(outdir / "aperture_frame_coverage.csv")
    print(outdir / "aperture_frame_coverage_summary.csv")
    print(outdir / "aperture_sync_frame_candidates.csv")
    print(outdir / "aperture_aoa_readiness_decision.csv")


if __name__ == "__main__":
    main()