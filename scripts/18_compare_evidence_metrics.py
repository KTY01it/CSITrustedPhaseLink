from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare multiple evidence metric CSV files.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Metric CSV files from scripts/17_eval_evidence_volume.py")
    parser.add_argument("--labels", nargs="+", required=True, help="Labels matching --inputs, e.g. raw tpl_v3")
    parser.add_argument("--output", default="outputs/reconstruction/evidence_compare_metrics.csv")
    args = parser.parse_args()

    if len(args.inputs) != len(args.labels):
        raise ValueError("--inputs and --labels must have the same length")

    rows = []
    for label, path in zip(args.labels, args.inputs):
        df = pd.read_csv(path)
        if len(df) != 1:
            raise ValueError(f"expected one row in {path}, got {len(df)}")
        row = df.iloc[0].to_dict()
        row["label"] = label
        rows.append(row)

    out_df = pd.DataFrame(rows)
    front_cols = ["label", "peak_x", "peak_y", "peak_z", "psr", "local_sharpness", "entropy", "compactness"]
    cols = [c for c in front_cols if c in out_df.columns] + [c for c in out_df.columns if c not in front_cols]
    out_df = out_df[cols]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False)

    print("=" * 80)
    print("EVIDENCE METRIC COMPARISON")
    print("=" * 80)
    print(out_df[[c for c in front_cols if c in out_df.columns]].to_string(index=False))
    print("saved:", out)


if __name__ == "__main__":
    main()
