from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd


EPS = 1e-12


def entropy_of_volume(v: np.ndarray) -> float:
    x = np.asarray(v, dtype=np.float64).ravel()
    x = np.clip(x, 0.0, None)
    p = x / (np.sum(x) + EPS)
    return float(-np.sum(p * np.log(p + EPS)))


def compactness(v: np.ndarray, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> float:
    x = np.asarray(v, dtype=np.float64)
    p = np.clip(x, 0.0, None)
    p = p / (np.sum(p) + EPS)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    mx = float(np.sum(p * X))
    my = float(np.sum(p * Y))
    mz = float(np.sum(p * Z))
    dist2 = (X - mx) ** 2 + (Y - my) ** 2 + (Z - mz) ** 2
    return float(np.sum(p * dist2))


def peak_to_sidelobe(v: np.ndarray, exclude_radius_vox: int = 2) -> tuple[float, float, float]:
    vol = np.asarray(v, dtype=np.float64)
    peak_idx = np.unravel_index(np.argmax(vol), vol.shape)
    peak = float(vol[peak_idx])

    mask = np.ones(vol.shape, dtype=bool)
    i, j, k = peak_idx
    r = int(exclude_radius_vox)
    mask[max(0, i-r):i+r+1, max(0, j-r):j+r+1, max(0, k-r):k+r+1] = False
    side = vol[mask]
    mu = float(np.mean(side))
    sig = float(np.std(side))
    psr = (peak - mu) / (sig + EPS)
    return float(psr), peak, sig


def local_sharpness(v: np.ndarray, radius_vox: int = 2) -> float:
    vol = np.asarray(v, dtype=np.float64)
    i, j, k = np.unravel_index(np.argmax(vol), vol.shape)
    r = int(radius_vox)
    nb = vol[max(0, i-r):i+r+1, max(0, j-r):j+r+1, max(0, k-r):k+r+1]
    peak = float(vol[i, j, k])
    return float(peak / (np.mean(nb) + EPS))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate TPL-BP-Occ evidence volume.")
    parser.add_argument("--input", default="outputs/reconstruction/evidence_tpl_v3.npz")
    parser.add_argument("--out_csv", default="outputs/reconstruction/evidence_tpl_v3_metrics.csv")
    parser.add_argument("--exclude-radius-vox", type=int, default=2)
    args = parser.parse_args()

    z = np.load(args.input, allow_pickle=True)
    if "evidence_norm" in z.files:
        vol = z["evidence_norm"].astype(np.float64)
    else:
        vol = z["evidence"].astype(np.float64)
    xs = z["xs"].astype(float)
    ys = z["ys"].astype(float)
    zs = z["zs"].astype(float)
    z.close()

    peak_idx = np.unravel_index(np.argmax(vol), vol.shape)
    peak_xyz = [float(xs[peak_idx[0]]), float(ys[peak_idx[1]]), float(zs[peak_idx[2]])]
    psr, peak_value, side_std = peak_to_sidelobe(vol, exclude_radius_vox=args.exclude_radius_vox)
    sharp = local_sharpness(vol, radius_vox=args.exclude_radius_vox)
    ent = entropy_of_volume(vol)
    comp = compactness(vol, xs, ys, zs)

    row = {
        "input": args.input,
        "shape_x": vol.shape[0],
        "shape_y": vol.shape[1],
        "shape_z": vol.shape[2],
        "peak_x": peak_xyz[0],
        "peak_y": peak_xyz[1],
        "peak_z": peak_xyz[2],
        "peak_value": peak_value,
        "psr": psr,
        "sidelobe_std": side_std,
        "local_sharpness": sharp,
        "entropy": ent,
        "compactness": comp,
    }

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(out, index=False)

    print("=" * 80)
    print("EVIDENCE VOLUME METRICS")
    print("=" * 80)
    for k, v in row.items():
        print(f"{k}: {v}")
    print("saved:", out)


if __name__ == "__main__":
    main()
