from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_OBJECT_CENTERS = {
    "obj1": np.array([0.6425, 0.2050, 0.1500], dtype=np.float64),
    "obj2": np.array([0.8400, 0.4925, 0.1600], dtype=np.float64),
    "obj3": np.array([0.4425, 0.3425, 0.0900], dtype=np.float64),
}


def _local_patch(vol: np.ndarray, i: int, j: int, k: int, r: int) -> np.ndarray:
    return vol[
        max(0, i - r): min(vol.shape[0], i + r + 1),
        max(0, j - r): min(vol.shape[1], j + r + 1),
        max(0, k - r): min(vol.shape[2], k + r + 1),
    ]


def extract_candidates(
    vol: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    top_k: int,
    nms_radius_vox: int,
    min_value: float,
    boundary_margin_vox: int,
) -> pd.DataFrame:
    score = vol.copy().astype(np.float64)

    if boundary_margin_vox > 0:
        m = boundary_margin_vox
        score[:m, :, :] = -np.inf
        score[-m:, :, :] = -np.inf
        score[:, :m, :] = -np.inf
        score[:, -m:, :] = -np.inf
        score[:, :, :m] = -np.inf
        score[:, :, -m:] = -np.inf

    rows = []
    for rank in range(1, top_k + 1):
        flat = int(np.nanargmax(score))
        val = float(score.ravel()[flat])
        if not np.isfinite(val) or val < min_value:
            break
        i, j, k = np.unravel_index(flat, score.shape)
        patch = _local_patch(vol, i, j, k, max(1, nms_radius_vox))
        local_mean = float(np.mean(patch))
        local_std = float(np.std(patch))
        local_sharpness = float(vol[i, j, k] / (local_mean + 1e-12))

        xyz = np.array([float(xs[i]), float(ys[j]), float(zs[k])], dtype=np.float64)
        nearest_name = None
        nearest_dist = np.nan
        for name, center in DEFAULT_OBJECT_CENTERS.items():
            dist = float(np.linalg.norm(xyz - center))
            if nearest_name is None or dist < nearest_dist:
                nearest_name = name
                nearest_dist = dist

        rows.append({
            "rank": rank,
            "i": int(i),
            "j": int(j),
            "k": int(k),
            "x": xyz[0],
            "y": xyz[1],
            "z": xyz[2],
            "value": float(vol[i, j, k]),
            "local_mean": local_mean,
            "local_std": local_std,
            "local_sharpness": local_sharpness,
            "nearest_object": nearest_name,
            "nearest_object_dist_m": nearest_dist,
        })

        # Suppress neighborhood around selected candidate.
        r = int(nms_radius_vox)
        score[
            max(0, i - r): min(score.shape[0], i + r + 1),
            max(0, j - r): min(score.shape[1], j + r + 1),
            max(0, k - r): min(score.shape[2], k + r + 1),
        ] = -np.inf

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract top-K non-maximum-suppressed candidate points from an evidence volume.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--nms-radius-vox", type=int, default=2)
    parser.add_argument("--min-value", type=float, default=0.0)
    parser.add_argument("--boundary-margin-vox", type=int, default=0, help="Ignore candidates within this many voxels of any volume boundary.")
    args = parser.parse_args()

    z = np.load(args.input, allow_pickle=True)
    vol = z["evidence_norm"].astype(np.float64) if "evidence_norm" in z.files else z["evidence"].astype(np.float64)
    xs = z["xs"].astype(np.float64)
    ys = z["ys"].astype(np.float64)
    zs = z["zs"].astype(np.float64)
    z.close()

    df = extract_candidates(
        vol=vol,
        xs=xs,
        ys=ys,
        zs=zs,
        top_k=args.top_k,
        nms_radius_vox=args.nms_radius_vox,
        min_value=args.min_value,
        boundary_margin_vox=args.boundary_margin_vox,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print("=" * 80)
    print("TOP-K EVIDENCE CANDIDATES")
    print("=" * 80)
    print("input:", args.input)
    print("output:", out)
    print("top_k:", args.top_k)
    print("nms_radius_vox:", args.nms_radius_vox)
    print("boundary_margin_vox:", args.boundary_margin_vox)
    if len(df) == 0:
        print("No candidates extracted.")
    else:
        cols = ["rank", "x", "y", "z", "value", "local_sharpness", "nearest_object", "nearest_object_dist_m"]
        print(df[cols].head(min(20, len(df))).to_string(index=False))


if __name__ == "__main__":
    main()
