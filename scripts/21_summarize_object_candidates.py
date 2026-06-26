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


def load_candidate_file(label: str, path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["method"] = label
    return df


def summarize_one(df: pd.DataFrame, radius_m: float) -> pd.DataFrame:
    rows = []
    for method, sub in df.groupby("method"):
        for obj, center in OBJECT_CENTERS.items():
            xyz = sub[["x", "y", "z"]].to_numpy(dtype=np.float64)
            dist = np.linalg.norm(xyz - center[None, :], axis=1)
            tmp = sub.copy()
            tmp["dist_to_object_m"] = dist
            tmp = tmp.sort_values(["dist_to_object_m", "rank"])
            best = tmp.iloc[0]
            within = tmp[tmp["dist_to_object_m"] <= radius_m]
            rows.append({
                "method": method,
                "object": obj,
                "center_x": center[0],
                "center_y": center[1],
                "center_z": center[2],
                "best_rank": int(best["rank"]),
                "best_x": float(best["x"]),
                "best_y": float(best["y"]),
                "best_z": float(best["z"]),
                "best_value": float(best["value"]),
                "best_local_sharpness": float(best["local_sharpness"]),
                "best_dist_m": float(best["dist_to_object_m"]),
                "n_candidates_within_radius": int(len(within)),
                "best_within_radius": bool(len(within) > 0),
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize top-K RF candidates at object level.")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--output", default="outputs/reconstruction/object_candidate_summary.csv")
    parser.add_argument("--radius-m", type=float, default=0.15)
    args = parser.parse_args()

    if len(args.inputs) != len(args.labels):
        raise ValueError("--inputs and --labels must have the same length")

    frames = [load_candidate_file(label, path) for label, path in zip(args.labels, args.inputs)]
    df = pd.concat(frames, ignore_index=True)
    out_df = summarize_one(df, radius_m=args.radius_m)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False)

    print("=" * 80)
    print("OBJECT-LEVEL CANDIDATE SUMMARY")
    print("=" * 80)
    print("radius_m:", args.radius_m)
    cols = [
        "method", "object", "best_rank", "best_dist_m", "best_x", "best_y", "best_z",
        "best_value", "best_local_sharpness", "n_candidates_within_radius", "best_within_radius",
    ]
    print(out_df[cols].to_string(index=False))
    print("saved:", out)


if __name__ == "__main__":
    main()
