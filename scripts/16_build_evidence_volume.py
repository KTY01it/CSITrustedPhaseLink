from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.reconstruction.backprojection import (
    DEFAULT_SUBCARRIER_SPACING_HZ,
    make_grid,
    normalize_volume,
    response_for_group,
)
from src.reconstruction.io_utils import safe_mkdir


def main() -> None:
    parser = argparse.ArgumentParser(description="Build confidence-weighted TPL backprojection evidence volume.")
    parser.add_argument("--input", default="outputs/reconstruction/group_fused_tpl_v3.npz")
    parser.add_argument("--output", default="outputs/reconstruction/evidence_tpl_v3.npz")

    parser.add_argument("--tx-x", type=float, required=True)
    parser.add_argument("--tx-y", type=float, required=True)
    parser.add_argument("--tx-z", type=float, required=True)
    parser.add_argument("--tx-op-x", type=float, default=None, help="Optional Tx x for opposite view.")
    parser.add_argument("--tx-op-y", type=float, default=None, help="Optional Tx y for opposite view.")
    parser.add_argument("--tx-op-z", type=float, default=None, help="Optional Tx z for opposite view.")

    parser.add_argument("--x-min", type=float, required=True)
    parser.add_argument("--x-max", type=float, required=True)
    parser.add_argument("--y-min", type=float, required=True)
    parser.add_argument("--y-max", type=float, required=True)
    parser.add_argument("--z-min", type=float, required=True)
    parser.add_argument("--z-max", type=float, required=True)
    parser.add_argument("--grid-step", type=float, required=True)

    parser.add_argument("--subcarrier-spacing-hz", type=float, default=DEFAULT_SUBCARRIER_SPACING_HZ)
    parser.add_argument("--chunk-voxels", type=int, default=50_000)
    parser.add_argument("--max-groups", type=int, default=0, help="Debug: limit number of groups; 0 means all.")
    parser.add_argument("--use-absolute-frequency", action="store_true", help="Diagnostic only; default uses subcarrier-offset frequencies.")
    parser.add_argument("--normalize-per-group", action="store_true", help="Normalize each group response before fusion.")
    parser.add_argument("--min-confidence", type=float, default=0.0)
    args = parser.parse_args()

    z = np.load(args.input, allow_pickle=True)
    hbar = z["hbar"].astype(np.complex64)
    sub_w = z["subcarrier_weight"].astype(np.float32)
    sub_idx = z["sub_idx"].astype(np.int32)
    rx_xyz = z["rx_xyz"].astype(np.float64)
    f0 = z["f0"].astype(float)
    group_conf = z["group_confidence"].astype(float)
    view = z["view"].astype(str)
    array = z["array"].astype(str)
    point = z["point"].astype(str)
    z.close()

    keep = np.where(group_conf >= float(args.min_confidence))[0]
    if args.max_groups and args.max_groups > 0:
        keep = keep[: int(args.max_groups)]
    if keep.size == 0:
        raise RuntimeError("No groups selected for evidence volume.")

    xs, ys, zs, pts = make_grid(
        args.x_min, args.x_max,
        args.y_min, args.y_max,
        args.z_min, args.z_max,
        args.grid_step,
    )
    tx_xyz = np.array([args.tx_x, args.tx_y, args.tx_z], dtype=np.float64)
    if args.tx_op_x is not None and args.tx_op_y is not None and args.tx_op_z is not None:
        tx_op_xyz = np.array([args.tx_op_x, args.tx_op_y, args.tx_op_z], dtype=np.float64)
    else:
        tx_op_xyz = tx_xyz.copy()

    evidence = np.zeros(pts.shape[0], dtype=np.float64)
    per_group_energy = []

    print("=" * 80)
    print("BUILD TPL-BP-OCC BASELINE 1 EVIDENCE VOLUME")
    print("=" * 80)
    print("input:", args.input)
    print("output:", args.output)
    print("selected groups:", keep.size, "/", hbar.shape[0])
    print("grid shape:", (len(xs), len(ys), len(zs)), "voxels:", pts.shape[0])
    print("tx_xyz main:", tx_xyz)
    print("tx_xyz opposite:", tx_op_xyz)
    print("use_absolute_frequency:", bool(args.use_absolute_frequency))

    for c, gi in enumerate(keep, start=1):
        tx_g = tx_op_xyz if str(view[gi]).lower() == "opposite" else tx_xyz
        resp = response_for_group(
            hbar=hbar[gi],
            sub_idx=sub_idx,
            f0_mhz=float(f0[gi]),
            tx_xyz=tx_g,
            rx_xyz=rx_xyz[gi],
            points_xyz=pts,
            subcarrier_weight=sub_w[gi],
            spacing_hz=args.subcarrier_spacing_hz,
            chunk_voxels=args.chunk_voxels,
            use_absolute_frequency=bool(args.use_absolute_frequency),
        )
        if args.normalize_per_group:
            resp = normalize_volume(resp)
        beta = float(np.clip(group_conf[gi], 0.0, 1.0))
        evidence += beta * resp
        per_group_energy.append(float(np.sum(resp)))

        if c == 1 or c % 50 == 0 or c == keep.size:
            print(f"processed {c}/{keep.size}: view={view[gi]} array={array[gi]} point={point[gi]} f0={f0[gi]:.1f} beta={beta:.3f}")

    evidence_norm = normalize_volume(evidence)
    volume = evidence.reshape(len(xs), len(ys), len(zs))
    volume_norm = evidence_norm.reshape(len(xs), len(ys), len(zs))

    out = Path(args.output)
    safe_mkdir(out.parent)
    np.savez_compressed(
        out,
        evidence=volume.astype(np.float32),
        evidence_norm=volume_norm.astype(np.float32),
        xs=xs.astype(np.float32),
        ys=ys.astype(np.float32),
        zs=zs.astype(np.float32),
        tx_xyz=tx_xyz.astype(np.float32),
        tx_op_xyz=tx_op_xyz.astype(np.float32),
        selected_group_idx=keep.astype(np.int32),
        per_group_energy=np.asarray(per_group_energy, dtype=np.float64),
        source_group_file=str(args.input),
        use_absolute_frequency=np.array([bool(args.use_absolute_frequency)]),
    )

    peak_idx = np.unravel_index(np.argmax(volume_norm), volume_norm.shape)
    peak_xyz = np.array([xs[peak_idx[0]], ys[peak_idx[1]], zs[peak_idx[2]]])
    print("saved:", out)
    print("peak index:", peak_idx)
    print("peak xyz:", peak_xyz)
    print("peak value norm:", float(volume_norm[peak_idx]))


if __name__ == "__main__":
    main()
