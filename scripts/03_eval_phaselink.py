import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default="outputs/metrics/common_offset_v1_metrics.csv")
    parser.add_argument("--outdir", default="outputs/metrics")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.metrics)
    ok = df[df["status"] == "ok"].copy()
    skip = df[df["status"] != "ok"].copy()

    print("=" * 80)
    print("COMMON-OFFSET PHASE-LINK V1 EVALUATION")
    print("=" * 80)

    print("\n[GROUP COUNTS]")
    print(df["status"].value_counts())

    print("\n[GLOBAL SUMMARY]")
    cols = ["n", "coh_before", "coh_after", "coh_gain", "residual_std", "confidence"]
    print(ok[cols].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]))

    improved_ratio = float((ok["coh_gain"] > 0).mean())
    strong_ratio = float((ok["coh_gain"] > 0.05).mean())
    very_strong_ratio = float((ok["coh_gain"] > 0.20).mean())

    print("\n[IMPROVEMENT RATIOS]")
    print("improved_ratio gain>0:", improved_ratio)
    print("strong_ratio gain>0.05:", strong_ratio)
    print("very_strong_ratio gain>0.20:", very_strong_ratio)

    print("\n[BY VIEW / ARRAY]")
    by_array = ok.groupby(["view", "array"])[
        ["n", "coh_before", "coh_after", "coh_gain", "residual_std", "confidence"]
    ].median().reset_index()
    print(by_array)
    by_array.to_csv(outdir / "common_offset_v1_by_view_array.csv", index=False)

    print("\n[BY VIEW / ARRAY / F0]")
    by_band = ok.groupby(["view", "array", "f0"])[
        ["n", "coh_before", "coh_after", "coh_gain", "residual_std", "confidence"]
    ].median().reset_index()
    print(by_band)
    by_band.to_csv(outdir / "common_offset_v1_by_view_array_band.csv", index=False)

    worst = ok.sort_values(["coh_gain", "coh_after"], ascending=[True, True]).head(30)
    best = ok.sort_values(["coh_gain", "coh_after"], ascending=[False, False]).head(30)

    worst.to_csv(outdir / "common_offset_v1_worst_groups.csv", index=False)
    best.to_csv(outdir / "common_offset_v1_best_groups.csv", index=False)
    skip.to_csv(outdir / "common_offset_v1_skipped_groups.csv", index=False)

    print("\n[WORST GROUPS]")
    print(worst[["view", "array", "point", "f0", "n", "coh_before", "coh_after", "coh_gain", "residual_std", "confidence"]])

    print("\n[SKIPPED GROUPS]")
    print(skip[["view", "array", "point", "f0", "n", "status"]].head(50))

    # Simple gate decision
    med_gain = float(ok["coh_gain"].median())
    med_after = float(ok["coh_after"].median())
    med_conf = float(ok["confidence"].median())

    pass_gate = (
        med_gain > 0.10 and
        med_after > 0.40 and
        improved_ratio > 0.75
    )

    print("\n[GATE]")
    print("median_gain:", med_gain)
    print("median_coh_after:", med_after)
    print("median_confidence:", med_conf)
    print("decision:", "PASS_TO_PHASE_1_5_SPLIT_STABILITY" if pass_gate else "NEED_DEBUG_PHASELINK_GROUPING")

    print("\nSaved:")
    print(outdir / "common_offset_v1_by_view_array.csv")
    print(outdir / "common_offset_v1_by_view_array_band.csv")
    print(outdir / "common_offset_v1_worst_groups.csv")
    print(outdir / "common_offset_v1_best_groups.csv")
    print(outdir / "common_offset_v1_skipped_groups.csv")


if __name__ == "__main__":
    main()