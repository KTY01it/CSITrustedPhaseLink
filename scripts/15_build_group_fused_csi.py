from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.reconstruction.io_utils import (
    build_meta_from_phase_npz,
    load_phase_npz,
    make_layout_lookup,
    safe_mkdir,
)


EPS = 1e-8


def _weighted_average_csi(csi_g: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    alpha = np.asarray(alpha, dtype=np.float64)
    alpha = np.where(np.isfinite(alpha), alpha, 0.0)
    alpha = np.clip(alpha, 0.0, None)
    if np.sum(alpha) <= EPS:
        alpha = np.ones(csi_g.shape[0], dtype=np.float64)
    alpha = alpha / (np.sum(alpha) + EPS)
    return np.sum(csi_g * alpha[:, None], axis=0).astype(np.complex64)


def _subcarrier_reliability(csi_g: np.ndarray) -> np.ndarray:
    """Simple per-group subcarrier reliability from amplitude stability.

    This is intentionally conservative for Baseline 1. It does not select
    spatial peaks and does not alter phase slopes.
    """
    amp = np.abs(csi_g).astype(np.float64)
    med = np.nanmedian(amp, axis=0)
    mad = np.nanmedian(np.abs(amp - med[None, :]), axis=0)
    score = med / (mad + EPS)
    if np.nanmax(score) > 0:
        score = score / (np.nanmax(score) + EPS)
    score = np.clip(score, 0.0, 1.0)
    return score.astype(np.float32)


def _print_no_group_diagnostics(meta: pd.DataFrame, layout_lookup: dict, skipped: list) -> None:
    print("\n[NO GROUP DIAGNOSTICS]")
    print("metadata groups:", meta.groupby(["view", "array", "point", "point_norm", "f0"]).ngroups)
    print("layout keys:", len(layout_lookup))
    print("sample meta points:")
    print(meta[["view", "array", "point", "point_norm", "f0"]].drop_duplicates().head(20).to_string(index=False))
    print("sample layout keys:")
    for k in list(layout_lookup.keys())[:20]:
        print(" ", k, "->", layout_lookup[k])
    if skipped:
        skipped_df = pd.DataFrame(skipped, columns=["view", "array", "point", "point_norm", "f0", "n_packets", "reason"])
        print("skip reasons:")
        print(skipped_df["reason"].value_counts())
        print("sample skipped:")
        print(skipped_df.head(20).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build group-level fused CSI for TPL-BP-Occ Baseline 1.")
    parser.add_argument("--input", default="outputs/phaselink/tpl_tof_net_v3.npz")
    parser.add_argument("--output", default="outputs/reconstruction/group_fused_tpl_v3.npz")
    parser.add_argument("--layout-main", default="")
    parser.add_argument("--layout-opposite", default="")
    parser.add_argument("--layout-unit-scale", type=float, default=1.0)
    parser.add_argument("--use", choices=["corr", "raw"], default="corr", help="Use csi_corr or csi_raw from the phase-link npz.")
    parser.add_argument("--min-frames", type=int, default=2)
    parser.add_argument("--missing-geometry", choices=["skip", "error"], default="skip")
    args = parser.parse_args()

    z = load_phase_npz(args.input)
    meta = build_meta_from_phase_npz(z)

    csi_key = "csi_corr" if args.use == "corr" else "csi_raw"
    if csi_key not in z:
        raise KeyError(f"{csi_key} not found in {args.input}; available keys={list(z.keys())}")
    csi = z[csi_key].astype(np.complex64)
    sub_idx = z["sub_idx"].astype(np.int32)

    layout_lookup = make_layout_lookup(
        args.layout_main or None,
        args.layout_opposite or None,
        unit_scale=args.layout_unit_scale,
    )

    hbars = []
    weights = []
    rx_xyz = []
    rows = []
    skipped = []

    grouped = meta.groupby(["view", "array", "point", "point_norm", "f0"], sort=False)
    for gkey, gdf in grouped:
        view, array, point, point_norm, f0 = gkey
        idx = gdf.index.to_numpy()
        if len(idx) < args.min_frames:
            skipped.append((view, array, point, point_norm, f0, len(idx), "too_few_frames"))
            continue

        geom_key = (str(view), str(point_norm))
        if geom_key not in layout_lookup:
            msg = f"missing geometry for {geom_key} from raw point={point}"
            if args.missing_geometry == "error":
                raise KeyError(msg)
            skipped.append((view, array, point, point_norm, f0, len(idx), "missing_geometry"))
            continue

        csi_g = csi[idx]
        alpha = gdf["packet_confidence"].to_numpy(dtype=np.float64)
        hbar = _weighted_average_csi(csi_g, alpha)
        wk = _subcarrier_reliability(csi_g)
        group_conf = float(np.nanmean(np.clip(alpha, 0.0, 1.0)))

        hbars.append(hbar)
        weights.append(wk)
        rx_xyz.append(layout_lookup[geom_key])
        rows.append({
            "view": view,
            "array": array,
            "point": point,
            "point_norm": point_norm,
            "f0": float(f0),
            "n_packets": int(len(idx)),
            "group_confidence": group_conf,
            "rx_x": float(layout_lookup[geom_key][0]),
            "rx_y": float(layout_lookup[geom_key][1]),
            "rx_z": float(layout_lookup[geom_key][2]),
        })

    if len(hbars) == 0:
        _print_no_group_diagnostics(meta, layout_lookup, skipped)
        raise RuntimeError("No fused groups produced. Check layout paths and point naming.")

    hbar_arr = np.stack(hbars, axis=0).astype(np.complex64)
    w_arr = np.stack(weights, axis=0).astype(np.float32)
    rx_arr = np.stack(rx_xyz, axis=0).astype(np.float64)
    group_meta = pd.DataFrame(rows)
    skipped_df = pd.DataFrame(skipped, columns=["view", "array", "point", "point_norm", "f0", "n_packets", "reason"])

    out = Path(args.output)
    safe_mkdir(out.parent)
    np.savez_compressed(
        out,
        hbar=hbar_arr,
        subcarrier_weight=w_arr,
        sub_idx=sub_idx,
        rx_xyz=rx_arr,
        view=group_meta["view"].to_numpy(dtype=object),
        array=group_meta["array"].to_numpy(dtype=object),
        point=group_meta["point"].to_numpy(dtype=object),
        point_norm=group_meta["point_norm"].to_numpy(dtype=object),
        f0=group_meta["f0"].to_numpy(np.float32),
        n_packets=group_meta["n_packets"].to_numpy(np.int32),
        group_confidence=group_meta["group_confidence"].to_numpy(np.float32),
    )

    meta_csv = out.with_suffix(".groups.csv")
    skip_csv = out.with_suffix(".skipped.csv")
    group_meta.to_csv(meta_csv, index=False)
    skipped_df.to_csv(skip_csv, index=False)

    print("=" * 80)
    print("BUILD GROUP-FUSED CSI FOR TPL-BP-OCC BASELINE 1")
    print("=" * 80)
    print("input:", args.input)
    print("use:", args.use)
    print("output:", out)
    print("groups:", len(group_meta))
    print("skipped:", len(skipped_df))
    print("hbar shape:", hbar_arr.shape)
    print("sub_idx shape:", sub_idx.shape)
    print("meta csv:", meta_csv)
    print("skipped csv:", skip_csv)
    print("\n[BY VIEW / ARRAY]")
    print(group_meta.groupby(["view", "array"])[["n_packets", "group_confidence"]].median())


if __name__ == "__main__":
    main()
