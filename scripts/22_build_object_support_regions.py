from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


OBJECT_CENTERS = {
    "obj1": np.array([0.6425, 0.2050, 0.1500], dtype=np.float64),
    "obj2": np.array([0.8400, 0.4925, 0.1600], dtype=np.float64),
    "obj3": np.array([0.4425, 0.3425, 0.0900], dtype=np.float64),
}


def _weighted_centroid(xyz: np.ndarray, w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64)
    w = np.maximum(w, 1e-12)
    return np.sum(xyz * w[:, None], axis=0) / np.sum(w)


def _weighted_spread(xyz: np.ndarray, w: np.ndarray, center: np.ndarray) -> float:
    w = np.asarray(w, dtype=np.float64)
    w = np.maximum(w, 1e-12)
    d2 = np.sum((xyz - center[None, :]) ** 2, axis=1)
    return float(np.sqrt(np.sum(w * d2) / np.sum(w)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build object-wise support regions from RF candidate points.")
    parser.add_argument("--input", required=True, help="Candidate CSV from scripts/20_extract_candidate_points.py")
    parser.add_argument("--output-prefix", default="outputs/reconstruction/object_support_diff_mean")
    parser.add_argument("--radius-m", type=float, default=0.15)
    parser.add_argument("--anchor-radius-m", type=float, default=0.08)
    parser.add_argument("--min-candidates", type=int, default=1)
    parser.add_argument("--max-candidates-per-object", type=int, default=12)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    support_rows = []
    point_rows = []

    for obj, gt_center in OBJECT_CENTERS.items():
        xyz_all = df[["x", "y", "z"]].to_numpy(dtype=np.float64)
        dist = np.linalg.norm(xyz_all - gt_center[None, :], axis=1)
        sub = df.copy()
        sub["dist_to_gt_center_m"] = dist
        sub = sub[sub["dist_to_gt_center_m"] <= float(args.radius_m)].copy()
        sub = sub.sort_values(["dist_to_gt_center_m", "rank"]).head(int(args.max_candidates_per_object))

        if len(sub) < int(args.min_candidates):
            support_rows.append({
                "object": obj,
                "status": "insufficient_candidates",
                "n_candidates": int(len(sub)),
                "gt_x": gt_center[0],
                "gt_y": gt_center[1],
                "gt_z": gt_center[2],
                "centroid_x": np.nan,
                "centroid_y": np.nan,
                "centroid_z": np.nan,
                "centroid_error_m": np.nan,
                "weighted_spread_m": np.nan,
                "best_rank": np.nan,
                "best_dist_m": np.nan,
                "mean_value": np.nan,
            })
            continue

        xyz = sub[["x", "y", "z"]].to_numpy(dtype=np.float64)
        val = sub["value"].to_numpy(dtype=np.float64)
        sharp = sub["local_sharpness"].to_numpy(dtype=np.float64)
        # Weight combines normalized evidence and mild local-sharpness. The current evidence is smooth,
        # so do not overtrust sharpness.
        w = np.maximum(val, 1e-6) * np.maximum(sharp, 1e-6)
        centroid = _weighted_centroid(xyz, w)
        spread = _weighted_spread(xyz, w, centroid)
        best = sub.sort_values(["dist_to_gt_center_m", "rank"]).iloc[0]
        centroid_error = float(np.linalg.norm(centroid - gt_center))

        support_rows.append({
            "object": obj,
            "status": "ok",
            "n_candidates": int(len(sub)),
            "gt_x": gt_center[0],
            "gt_y": gt_center[1],
            "gt_z": gt_center[2],
            "centroid_x": float(centroid[0]),
            "centroid_y": float(centroid[1]),
            "centroid_z": float(centroid[2]),
            "centroid_error_m": centroid_error,
            "weighted_spread_m": spread,
            "best_rank": int(best["rank"]),
            "best_dist_m": float(best["dist_to_gt_center_m"]),
            "mean_value": float(np.mean(val)),
        })

        for _, r in sub.iterrows():
            role = "anchor" if float(r["dist_to_gt_center_m"]) <= float(args.anchor_radius_m) else "support"
            point_rows.append({
                "object": obj,
                "role": role,
                "rank": int(r["rank"]),
                "x": float(r["x"]),
                "y": float(r["y"]),
                "z": float(r["z"]),
                "value": float(r["value"]),
                "local_sharpness": float(r["local_sharpness"]),
                "dist_to_gt_center_m": float(r["dist_to_gt_center_m"]),
            })

    support_df = pd.DataFrame(support_rows)
    points_df = pd.DataFrame(point_rows)

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_csv = prefix.with_suffix(".summary.csv")
    points_csv = prefix.with_suffix(".points.csv")
    support_df.to_csv(summary_csv, index=False)
    points_df.to_csv(points_csv, index=False)

    print("=" * 80)
    print("OBJECT SUPPORT REGIONS FROM RF CANDIDATES")
    print("=" * 80)
    print("input:", args.input)
    print("radius_m:", args.radius_m)
    print("anchor_radius_m:", args.anchor_radius_m)
    cols = [
        "object", "status", "n_candidates", "centroid_x", "centroid_y", "centroid_z",
        "centroid_error_m", "weighted_spread_m", "best_rank", "best_dist_m", "mean_value",
    ]
    print(support_df[cols].to_string(index=False))
    if len(points_df):
        print("\n[POINT ROLES]")
        print(points_df.groupby(["object", "role"]).size())
    print("summary:", summary_csv)
    print("points:", points_csv)


if __name__ == "__main__":
    main()
