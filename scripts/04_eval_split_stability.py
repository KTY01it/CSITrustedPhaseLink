import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.phase.common_offset import (
    phase_link_group_common_offset,
    group_coherence,
)


EPS = 1e-8


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


def normalize_complex_rows(x):
    """
    Normalize từng packet theo norm để giảm ảnh hưởng biên độ.
    x: [N, K]
    """
    den = np.linalg.norm(x, axis=1, keepdims=True)
    den = np.maximum(den, EPS)
    return x / den


def representative_vector(csi_group):
    """
    Lấy representative phase vector của group.
    Dùng complex mean sau khi normalize từng packet.
    """
    x = normalize_complex_rows(csi_group)
    rep = np.mean(x, axis=0)
    den = np.linalg.norm(rep)
    if den < EPS:
        return rep
    return rep / den


def best_common_align_residual(rep_a, rep_b):
    """
    So sánh hai representative vectors sau khi bù common offset tốt nhất.
    Không fit slope, không xóa ToF-related slope.
    """
    z = np.sum(rep_b * np.conj(rep_a))
    delta = np.angle(z) if np.abs(z) > EPS else 0.0

    rep_b_aligned = rep_b * np.exp(-1j * delta)

    # similarity càng cao càng tốt
    sim_num = np.abs(np.sum(rep_b_aligned * np.conj(rep_a)))
    sim_den = np.linalg.norm(rep_a) * np.linalg.norm(rep_b_aligned) + EPS
    similarity = float(sim_num / sim_den)

    # residual phase std càng thấp càng tốt
    residual = np.angle(rep_b_aligned * np.conj(rep_a))
    residual_std = float(np.std(residual))

    return similarity, residual_std, float(delta)


def split_once_indices(n, rng):
    perm = rng.permutation(n)
    mid = n // 2
    return perm[:mid], perm[mid:]


def evaluate_group_split(csi_group, n_boot=20, seed=0):
    n = csi_group.shape[0]
    rng = np.random.default_rng(seed)

    rows = []

    for b in range(n_boot):
        ia, ib = split_once_indices(n, rng)

        if len(ia) < 2 or len(ib) < 2:
            continue

        raw_a = csi_group[ia]
        raw_b = csi_group[ib]

        # Raw split representatives
        raw_rep_a = representative_vector(raw_a)
        raw_rep_b = representative_vector(raw_b)
        raw_sim, raw_resstd, raw_delta = best_common_align_residual(raw_rep_a, raw_rep_b)

        raw_coh_a = group_coherence(raw_a)
        raw_coh_b = group_coherence(raw_b)

        # Phase-link each split independently
        link_a = phase_link_group_common_offset(raw_a)["csi_corr"]
        link_b = phase_link_group_common_offset(raw_b)["csi_corr"]

        link_rep_a = representative_vector(link_a)
        link_rep_b = representative_vector(link_b)
        link_sim, link_resstd, link_delta = best_common_align_residual(link_rep_a, link_rep_b)

        link_coh_a = group_coherence(link_a)
        link_coh_b = group_coherence(link_b)

        rows.append({
            "boot": b,
            "raw_similarity": raw_sim,
            "link_similarity": link_sim,
            "similarity_gain": link_sim - raw_sim,

            "raw_residual_std": raw_resstd,
            "link_residual_std": link_resstd,
            "residual_std_reduction": raw_resstd - link_resstd,

            "raw_coh_mean": 0.5 * (raw_coh_a + raw_coh_b),
            "link_coh_mean": 0.5 * (link_coh_a + link_coh_b),
            "split_coh_gain": 0.5 * (link_coh_a + link_coh_b) - 0.5 * (raw_coh_a + raw_coh_b),

            "raw_delta": raw_delta,
            "link_delta": link_delta,
        })

    if not rows:
        return None

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/canonical/canonical_k53_packets.npz")
    parser.add_argument("--out_csv", default="outputs/metrics/split_stability_v1.csv")
    parser.add_argument("--summary_csv", default="outputs/metrics/split_stability_v1_summary.csv")
    parser.add_argument("--min_frames", type=int, default=8)
    parser.add_argument("--n_boot", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    csi, sub_idx, meta = load_canonical(args.input)

    all_rows = []
    summary_rows = []

    grouped = meta.groupby(["view", "array", "point", "f0"], sort=False)

    group_counter = 0
    used_counter = 0

    for gkey, gdf in grouped:
        group_counter += 1

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

        csi_g = csi[idx]
        seed_g = args.seed + group_counter * 17

        boot_df = evaluate_group_split(
            csi_g,
            n_boot=args.n_boot,
            seed=seed_g,
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

        used_counter += 1

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

            "raw_similarity_median": boot_df["raw_similarity"].median(),
            "link_similarity_median": boot_df["link_similarity"].median(),
            "similarity_gain_median": boot_df["similarity_gain"].median(),

            "raw_residual_std_median": boot_df["raw_residual_std"].median(),
            "link_residual_std_median": boot_df["link_residual_std"].median(),
            "residual_std_reduction_median": boot_df["residual_std_reduction"].median(),

            "raw_coh_mean_median": boot_df["raw_coh_mean"].median(),
            "link_coh_mean_median": boot_df["link_coh_mean"].median(),
            "split_coh_gain_median": boot_df["split_coh_gain"].median(),
        })

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_csv).parent.mkdir(parents=True, exist_ok=True)

    if all_rows:
        all_df = pd.concat(all_rows, ignore_index=True)
    else:
        all_df = pd.DataFrame()

    summary = pd.DataFrame(summary_rows)

    all_df.to_csv(args.out_csv, index=False)
    summary.to_csv(args.summary_csv, index=False)

    ok = summary[summary["status"] == "ok"].copy()

    print("=" * 80)
    print("SPLIT-STABILITY PHASE-LINK V1")
    print("=" * 80)
    print("total groups:", group_counter)
    print("used groups:", used_counter)
    print(summary["status"].value_counts())

    print("\n[GLOBAL SPLIT SUMMARY]")
    cols = [
        "n",
        "raw_similarity_median",
        "link_similarity_median",
        "similarity_gain_median",
        "raw_residual_std_median",
        "link_residual_std_median",
        "residual_std_reduction_median",
        "raw_coh_mean_median",
        "link_coh_mean_median",
        "split_coh_gain_median",
    ]
    print(ok[cols].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]))

    print("\n[BY VIEW / ARRAY]")
    by_array = ok.groupby(["view", "array"])[[
        "similarity_gain_median",
        "residual_std_reduction_median",
        "split_coh_gain_median",
        "link_similarity_median",
        "link_residual_std_median",
    ]].median().reset_index()
    print(by_array)

    print("\n[BY VIEW / ARRAY / F0]")
    by_band = ok.groupby(["view", "array", "f0"])[[
        "similarity_gain_median",
        "residual_std_reduction_median",
        "split_coh_gain_median",
        "link_similarity_median",
        "link_residual_std_median",
    ]].median().reset_index()
    print(by_band)

    worst = ok.sort_values(
        ["similarity_gain_median", "residual_std_reduction_median"],
        ascending=[True, True],
    ).head(30)

    worst_path = Path("outputs/metrics/split_stability_v1_worst_groups.csv")
    worst.to_csv(worst_path, index=False)

    print("\n[WORST SPLIT-STABILITY GROUPS]")
    print(worst[[
        "view", "array", "point", "f0", "n",
        "similarity_gain_median",
        "residual_std_reduction_median",
        "split_coh_gain_median",
        "link_similarity_median",
        "link_residual_std_median",
    ]])

    # Gate
    med_sim_gain = float(ok["similarity_gain_median"].median())
    med_res_reduction = float(ok["residual_std_reduction_median"].median())
    med_coh_gain = float(ok["split_coh_gain_median"].median())

    pass_gate = (
        med_coh_gain > 0.10 and
        med_res_reduction > 0.0
    )

    print("\n[GATE]")
    print("median_similarity_gain:", med_sim_gain)
    print("median_residual_std_reduction:", med_res_reduction)
    print("median_split_coh_gain:", med_coh_gain)
    print("decision:", "PASS_TO_PHASE_2_PATH_AUDIT" if pass_gate else "NEED_DEBUG_SPLIT_STABILITY")

    print("\nSaved:")
    print(args.out_csv)
    print(args.summary_csv)
    print(worst_path)


if __name__ == "__main__":
    main()