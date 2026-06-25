import argparse
from pathlib import Path
import pandas as pd


def summarize(path, name):
    df = pd.read_csv(path)
    ok = df[df["status"] == "ok"].copy()

    row = {
        "name": name,
        "n_groups": len(ok),
        "median_coh_before": ok["coh_before"].median(),
        "median_coh_after": ok["coh_after"].median(),
        "median_coh_gain": ok["coh_gain"].median(),
        "median_residual_std": ok["residual_std"].median(),
        "mean_confidence": ok["confidence"].mean(),
    }

    if "drift_residual_std" in ok.columns:
        row["median_drift_residual_std"] = ok["drift_residual_std"].median()
    else:
        row["median_drift_residual_std"] = float("nan")

    return row


def add_if_exists(rows, path, name):
    if Path(path).exists():
        rows.append(summarize(path, name))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1", default="outputs/metrics/common_offset_v1_metrics.csv")
    parser.add_argument("--v2_hybrid", default="outputs/metrics/drift_aware_v2_hybrid_metrics.csv")
    parser.add_argument("--v2_trend", default="outputs/metrics/drift_aware_v2_trend_metrics.csv")
    parser.add_argument("--v2_consensus", default="outputs/metrics/consensus_common_offset_v2_metrics.csv")
    args = parser.parse_args()

    rows = []
    add_if_exists(rows, args.v1, "common_offset_v1")
    add_if_exists(rows, args.v2_hybrid, "drift_aware_v2_hybrid")
    add_if_exists(rows, args.v2_trend, "drift_aware_v2_trend")
    add_if_exists(rows, args.v2_consensus, "consensus_common_offset_v2")

    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    out.to_csv("outputs/metrics/compare_v1_v2_metrics.csv", index=False)
    print("\nSaved: outputs/metrics/compare_v1_v2_metrics.csv")


if __name__ == "__main__":
    main()