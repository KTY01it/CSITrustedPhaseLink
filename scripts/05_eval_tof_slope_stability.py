import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd


EPS = 1e-8
SUBCARRIER_SPACING_HZ = 312_500.0


def load_phaselink_npz(path):
    z = np.load(path, allow_pickle=True)

    meta = pd.DataFrame({
        "view": z["view"].astype(str),
        "array": z["array"].astype(str),
        "point": z["point"].astype(str),
        "f0": z["f0"].astype(float),
        "frame": z["frame"].astype(int),
    })

    return (
        z["csi_raw"].astype(np.complex64),
        z["csi_corr"].astype(np.complex64),
        z["sub_idx"].astype(np.int32),
        meta,
    )


def normalize_complex_rows(x):
    den = np.linalg.norm(x, axis=1, keepdims=True)
    den = np.maximum(den, EPS)
    return x / den


def representative_vector(csi_group):
    """
    Complex representative vector after per-packet normalization.
    """
    x = normalize_complex_rows(csi_group)
    rep = np.mean(x, axis=0)
    den = np.linalg.norm(rep)
    if den < EPS:
        return rep
    return rep / den


def estimate_phase_slope(rep, sub_idx):
    """
    Estimate linear phase slope over subcarrier index.
    Slope unit: rad / subcarrier.
    ToF proxy:
        tau = - slope / (2*pi*delta_f)
    This is only a relative proxy, not absolute ToF.
    """
    phase = np.unwrap(np.angle(rep))
    x = sub_idx.astype(np.float64)

    # Remove mean for stable least squares
    xc = x - x.mean()
    yc = phase - phase.mean()

    denom = np.sum(xc ** 2)
    if denom < EPS:
        return np.nan, np.nan

    slope = float(np.sum(xc * yc) / denom)

    # Goodness of linear fit
    pred = slope * xc
    resid = yc - pred
    residual_std = float(np.std(resid))

    tau_proxy = -slope / (2.0 * np.pi * SUBCARRIER_SPACING_HZ)

    return slope, tau_proxy, residual_std


def split_indices(n, rng):
    perm = rng.permutation(n)
    mid = n // 2
    return perm[:mid], perm[mid:]


def eval_group(csi_raw_g, csi_corr_g, sub_idx, n_boot=20, seed=0):
    n = csi_raw_g.shape[0]
    rng = np.random.default_rng(seed)

    rows = []

    for b in range(n_boot):
        ia, ib = split_indices(n, rng)

        if len(ia) < 2 or len(ib) < 2:
            continue

        raw_a = representative_vector(csi_raw_g[ia])
        raw_b = representative_vector(csi_raw_g[ib])

        corr_a = representative_vector(csi_corr_g[ia])
        corr_b = representative_vector(csi_corr_g[ib])

        raw_slope_a, raw_tau_a, raw_res_a = estimate_phase_slope(raw_a, sub_idx)
        raw_slope_b, raw_tau_b, raw_res_b = estimate_phase_slope(raw_b, sub_idx)

        corr_slope_a, corr_tau_a, corr_res_a = estimate_phase_slope(corr_a, sub_idx)
        corr_slope_b, corr_tau_b, corr_res_b = estimate_phase_slope(corr_b, sub_idx)

        raw_slope_diff = abs(raw_slope_a - raw_slope_b)
        corr_slope_diff = abs(corr_slope_a - corr_slope_b)

        raw_tau_diff_ns = abs(raw_tau_a - raw_tau_b) * 1e9
        corr_tau_diff_ns = abs(corr_tau_a - corr_tau_b) * 1e9

        rows.append({
            "boot": b,

            "raw_slope_a": raw_slope_a,
            "raw_slope_b": raw_slope_b,
            "corr_slope_a": corr_slope_a,
            "corr_slope_b": corr_slope_b,

            "raw_slope_diff": raw_slope_diff,
            "corr_slope_diff": corr_slope_diff,
            "slope_diff_reduction": raw_slope_diff - corr_slope_diff,

            "raw_tau_diff_ns": raw_tau_diff_ns,
            "corr_tau_diff_ns": corr_tau_diff_ns,
            "tau_diff_reduction_ns": raw_tau_diff_ns - corr_tau_diff_ns,

            "raw_linear_residual_std": 0.5 * (raw_res_a + raw_res_b),
            "corr_linear_residual_std": 0.5 * (corr_res_a + corr_res_b),
            "linear_residual_reduction": 0.5 * (raw_res_a + raw_res_b) - 0.5 * (corr_res_a + corr_res_b),
        })

    if not rows:
        return None

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/phaselink/common_offset_v1.npz")
    parser.add_argument("--out_csv", default="outputs/metrics/tof_slope_stability_v1.csv")
    parser.add_argument("--summary_csv", default="outputs/metrics/tof_slope_stability_v1_summary.csv")
    parser.add_argument("--min_frames", type=int, default=8)
    parser.add_argument("--n_boot", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    csi_raw, csi_corr, sub_idx, meta = load_phaselink_npz(args.input)

    all_rows = []
    summary_rows = []

    grouped = meta.groupby(["view", "array", "point", "f0"], sort=False)

    total_groups = 0
    used_groups = 0

    for gkey, gdf in grouped:
        total_groups += 1
        idx = gdf.index.to_numpy()

        if len(idx) < args.min_frames:
            summary_rows.append({
                "view": gkey[0],
                "array": gkey[1],
                "point": gkey[2],
                "f0": gkey[3],
                "n": len(idx),
                "status": "skip_too_few_frames",
            })
            continue

        raw_g = csi_raw[idx]
        corr_g = csi_corr[idx]

        boot_df = eval_group(
            raw_g,
            corr_g,
            sub_idx,
            n_boot=args.n_boot,
            seed=args.seed + total_groups * 19,
        )

        if boot_df is None or boot_df.empty:
            summary_rows.append({
                "view": gkey[0],
                "array": gkey[1],
                "point": gkey[2],
                "f0": gkey[3],
                "n": len(idx),
                "status": "skip_no_valid_bootstrap",
            })
            continue

        used_groups += 1

        boot_df.insert(0, "view", gkey[0])
        boot_df.insert(1, "array", gkey[1])
        boot_df.insert(2, "point", gkey[2])
        boot_df.insert(3, "f0", gkey[3])
        boot_df.insert(4, "n", len(idx))

        all_rows.append(boot_df)

        summary_rows.append({
            "view": gkey[0],
            "array": gkey[1],
            "point": gkey[2],
            "f0": gkey[3],
            "n": len(idx),
            "status": "ok",

            "raw_slope_diff_median": boot_df["raw_slope_diff"].median(),
            "corr_slope_diff_median": boot_df["corr_slope_diff"].median(),
            "slope_diff_reduction_median": boot_df["slope_diff_reduction"].median(),

            "raw_tau_diff_ns_median": boot_df["raw_tau_diff_ns"].median(),
            "corr_tau_diff_ns_median": boot_df["corr_tau_diff_ns"].median(),
            "tau_diff_reduction_ns_median": boot_df["tau_diff_reduction_ns"].median(),

            "raw_linear_residual_std_median": boot_df["raw_linear_residual_std"].median(),
            "corr_linear_residual_std_median": boot_df["corr_linear_residual_std"].median(),
            "linear_residual_reduction_median": boot_df["linear_residual_reduction"].median(),
        })

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_csv).parent.mkdir(parents=True, exist_ok=True)

    all_df = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)

    all_df.to_csv(args.out_csv, index=False)
    summary.to_csv(args.summary_csv, index=False)

    ok = summary[summary["status"] == "ok"].copy()

    print("=" * 80)
    print("TOF-SLOPE STABILITY AUDIT V1")
    print("=" * 80)
    print("total groups:", total_groups)
    print("used groups:", used_groups)
    print(summary["status"].value_counts())

    print("\n[GLOBAL SUMMARY]")
    cols = [
        "n",
        "raw_slope_diff_median",
        "corr_slope_diff_median",
        "slope_diff_reduction_median",
        "raw_tau_diff_ns_median",
        "corr_tau_diff_ns_median",
        "tau_diff_reduction_ns_median",
        "raw_linear_residual_std_median",
        "corr_linear_residual_std_median",
        "linear_residual_reduction_median",
    ]
    print(ok[cols].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]))

    print("\n[BY VIEW / ARRAY]")
    by_array = ok.groupby(["view", "array"])[[
        "slope_diff_reduction_median",
        "tau_diff_reduction_ns_median",
        "linear_residual_reduction_median",
        "corr_tau_diff_ns_median",
        "corr_linear_residual_std_median",
    ]].median().reset_index()
    print(by_array)

    print("\n[BY VIEW / ARRAY / F0]")
    by_band = ok.groupby(["view", "array", "f0"])[[
        "slope_diff_reduction_median",
        "tau_diff_reduction_ns_median",
        "linear_residual_reduction_median",
        "corr_tau_diff_ns_median",
        "corr_linear_residual_std_median",
    ]].median().reset_index()
    print(by_band)

    # Gate
    med_slope_red = float(ok["slope_diff_reduction_median"].median())
    med_tau_red = float(ok["tau_diff_reduction_ns_median"].median())
    med_lin_red = float(ok["linear_residual_reduction_median"].median())

    # Since common-offset does not directly modify slope, slope gain may be modest.
    # The important fail signal is strongly negative median.
    pass_gate = (
        med_slope_red > -0.01 and
        med_lin_red > -0.05
    )

    print("\n[GATE]")
    print("median_slope_diff_reduction:", med_slope_red)
    print("median_tau_diff_reduction_ns:", med_tau_red)
    print("median_linear_residual_reduction:", med_lin_red)
    print("decision:", "PASS_TO_PHASE_2B_AOA_AUDIT" if pass_gate else "NEED_DEBUG_TOF_SLOPE_STABILITY")

    print("\nSaved:")
    print(args.out_csv)
    print(args.summary_csv)


if __name__ == "__main__":
    main()