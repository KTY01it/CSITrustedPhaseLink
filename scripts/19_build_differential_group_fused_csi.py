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
    amp = np.abs(csi_g).astype(np.float64)
    med = np.nanmedian(amp, axis=0)
    mad = np.nanmedian(np.abs(amp - med[None, :]), axis=0)
    score = med / (mad + EPS)
    if np.nanmax(score) > 0:
        score = score / (np.nanmax(score) + EPS)
    return np.clip(score, 0.0, 1.0).astype(np.float32)


def _is_ref_group(view: str, array: str, point: str, point_norm: str) -> bool:
    a = str(array).lower()
    p = str(point).lower()
    pn = str(point_norm).lower()
    return a == "ref" or p.startswith("z-") or pn.startswith("z")


def _complex_ls_scale(h_ref: np.ndarray, h_scan: np.ndarray) -> complex:
    denom = np.vdot(h_ref, h_ref)
    if np.abs(denom) < EPS:
        return 0.0 + 0.0j
    return np.vdot(h_ref, h_scan) / denom


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build reference-subtracted group CSI for differential TPL backprojection."
    )
    parser.add_argument("--input", default="outputs/phaselink/tpl_tof_net_v3.npz")
    parser.add_argument("--output", default="outputs/reconstruction/group_fused_diff_tpl_v3_allviews.npz")
    parser.add_argument("--layout-main", default="")
    parser.add_argument("--layout-opposite", default="")
    parser.add_argument("--layout-unit-scale", type=float, default=1.0)
    parser.add_argument("--use", choices=["corr", "raw"], default="corr")
    parser.add_argument("--min-frames", type=int, default=2)
    parser.add_argument(
        "--background-mode",
        choices=["ref_by_f0", "mean_by_view_array_f0"],
        default="ref_by_f0",
        help="ref_by_f0 uses reference groups; mean_by_view_array_f0 subtracts the scan-array mean as a fallback diagnostic.",
    )
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

    # First pass: fuse every valid group, including ref groups without geometry.
    fused_rows = []
    grouped = meta.groupby(["view", "array", "point", "point_norm", "f0"], sort=False)
    for gkey, gdf in grouped:
        view, array, point, point_norm, f0 = gkey
        idx = gdf.index.to_numpy()
        if len(idx) < args.min_frames:
            fused_rows.append({
                "view": view,
                "array": array,
                "point": point,
                "point_norm": point_norm,
                "f0": float(f0),
                "n_packets": int(len(idx)),
                "status": "too_few_frames",
                "hbar": None,
                "wk": None,
                "group_confidence": np.nan,
                "rx_xyz": None,
                "is_ref": False,
            })
            continue

        csi_g = csi[idx]
        alpha = gdf["packet_confidence"].to_numpy(dtype=np.float64)
        hbar = _weighted_average_csi(csi_g, alpha)
        wk = _subcarrier_reliability(csi_g)
        group_conf = float(np.nanmean(np.clip(alpha, 0.0, 1.0)))
        is_ref = _is_ref_group(view, array, point, point_norm)

        geom_key = (str(view), str(point_norm))
        rx = layout_lookup.get(geom_key, None)
        status = "ok" if (rx is not None or is_ref) else "missing_geometry"
        fused_rows.append({
            "view": view,
            "array": array,
            "point": point,
            "point_norm": point_norm,
            "f0": float(f0),
            "n_packets": int(len(idx)),
            "status": status,
            "hbar": hbar,
            "wk": wk,
            "group_confidence": group_conf,
            "rx_xyz": rx,
            "is_ref": is_ref,
        })

    fused = pd.DataFrame(fused_rows)

    # Build background dictionary.
    bg: dict[tuple, np.ndarray] = {}
    if args.background_mode == "ref_by_f0":
        ref_ok = fused[(fused["is_ref"] == True) & fused["hbar"].notna()]
        for f0, sub in ref_ok.groupby("f0"):
            hs = np.stack(sub["hbar"].to_list(), axis=0)
            bg[(float(f0),)] = np.mean(hs, axis=0).astype(np.complex64)
    else:
        scan_ok = fused[(fused["is_ref"] == False) & (fused["status"] == "ok") & fused["hbar"].notna()]
        for key, sub in scan_ok.groupby(["view", "array", "f0"]):
            hs = np.stack(sub["hbar"].to_list(), axis=0)
            bg[(str(key[0]), str(key[1]), float(key[2]))] = np.mean(hs, axis=0).astype(np.complex64)

    hbars = []
    weights = []
    rx_xyz = []
    rows = []
    skipped = []

    for _, row in fused.iterrows():
        if row["status"] != "ok" or bool(row["is_ref"]):
            skipped.append((row["view"], row["array"], row["point"], row["point_norm"], row["f0"], row["n_packets"], row["status"]))
            continue

        h_scan = row["hbar"]
        if h_scan is None:
            skipped.append((row["view"], row["array"], row["point"], row["point_norm"], row["f0"], row["n_packets"], "missing_hbar"))
            continue

        if args.background_mode == "ref_by_f0":
            bg_key = (float(row["f0"]),)
        else:
            bg_key = (str(row["view"]), str(row["array"]), float(row["f0"]))

        if bg_key not in bg:
            skipped.append((row["view"], row["array"], row["point"], row["point_norm"], row["f0"], row["n_packets"], "missing_background"))
            continue

        h_bg = bg[bg_key]
        eta = _complex_ls_scale(h_bg, h_scan)
        h_diff = (h_scan - eta * h_bg).astype(np.complex64)

        hbars.append(h_diff)
        weights.append(row["wk"])
        rx_xyz.append(row["rx_xyz"])
        rows.append({
            "view": row["view"],
            "array": row["array"],
            "point": row["point"],
            "point_norm": row["point_norm"],
            "f0": float(row["f0"]),
            "n_packets": int(row["n_packets"]),
            "group_confidence": float(row["group_confidence"]),
            "eta_real": float(np.real(eta)),
            "eta_imag": float(np.imag(eta)),
            "diff_norm": float(np.linalg.norm(h_diff)),
            "scan_norm": float(np.linalg.norm(h_scan)),
            "bg_norm": float(np.linalg.norm(h_bg)),
            "rx_x": float(row["rx_xyz"][0]),
            "rx_y": float(row["rx_xyz"][1]),
            "rx_z": float(row["rx_xyz"][2]),
        })

    if len(hbars) == 0:
        skipped_df = pd.DataFrame(skipped, columns=["view", "array", "point", "point_norm", "f0", "n_packets", "reason"])
        print("[NO DIFFERENTIAL GROUPS]")
        print("background mode:", args.background_mode)
        print("background keys:", list(bg.keys())[:20], "... total", len(bg))
        print(skipped_df["reason"].value_counts())
        print(skipped_df.head(30).to_string(index=False))
        raise RuntimeError("No differential fused groups produced.")

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
        background_mode=np.array([args.background_mode], dtype=object),
    )

    meta_csv = out.with_suffix(".groups.csv")
    skip_csv = out.with_suffix(".skipped.csv")
    group_meta.to_csv(meta_csv, index=False)
    skipped_df.to_csv(skip_csv, index=False)

    print("=" * 80)
    print("BUILD DIFFERENTIAL GROUP-FUSED CSI")
    print("=" * 80)
    print("input:", args.input)
    print("use:", args.use)
    print("background mode:", args.background_mode)
    print("background keys:", len(bg))
    print("output:", out)
    print("groups:", len(group_meta))
    print("skipped:", len(skipped_df))
    print("hbar shape:", hbar_arr.shape)
    print("meta csv:", meta_csv)
    print("skipped csv:", skip_csv)
    print("\n[BY VIEW / ARRAY]")
    print(group_meta.groupby(["view", "array"])[["n_packets", "group_confidence", "diff_norm", "scan_norm"]].median())


if __name__ == "__main__":
    main()
