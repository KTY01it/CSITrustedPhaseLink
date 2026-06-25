import argparse
import yaml
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_dataset
from src.data.canonicalize import build_canonical_arrays, save_canonical_npz


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))
    out_path = Path(cfg["output"]["canonical_npz"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(
        root_scan0=cfg["data"]["root_scan0"],
        root_scan1=cfg["data"]["root_scan1"],
        root_zref=cfg["data"]["root_zref"],
        layout_csv=cfg["data"]["layout_main"],
        root_opposite=cfg["data"].get("root_opposite", None),
        layout_csv_opposite=cfg["data"].get("layout_opposite", None),
        load_opposite=True,
    )

    arrays = build_canonical_arrays(
        ds,
        drop_noncanonical=cfg["canonical"].get("drop_noncanonical_k", True),
    )
    save_canonical_npz(str(out_path), arrays)

    print("Saved:", out_path)
    print("canonical packets:", arrays["csi"].shape[0])
    print("dropped packets:", int(arrays["dropped"][0]))
    print("csi shape:", arrays["csi"].shape)


if __name__ == "__main__":
    main()