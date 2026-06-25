from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


REQUIRED_PHASE_NPZ_KEYS = [
    "sub_idx",
    "view",
    "array",
    "point",
    "f0",
]


def load_phase_npz(path: str | Path) -> dict:
    """Load a phase-link npz produced by the TPL-ToF scripts.

    Expected arrays include either csi_corr or csi_raw, plus metadata arrays.
    The function returns a normal dict so callers do not keep an open NpzFile.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"phase npz not found: {path}")

    z = np.load(path, allow_pickle=True)
    missing = [k for k in REQUIRED_PHASE_NPZ_KEYS if k not in z.files]
    if missing:
        raise KeyError(f"missing keys in {path}: {missing}; found={z.files}")

    out = {k: z[k] for k in z.files}
    z.close()
    return out


def build_meta_from_phase_npz(z: dict) -> pd.DataFrame:
    n = len(z["view"])
    meta = pd.DataFrame({
        "view": z["view"].astype(str),
        "array": z["array"].astype(str),
        "point": z["point"].astype(str),
        "f0": z["f0"].astype(float),
    })
    if "frame" in z:
        meta["frame"] = z["frame"].astype(int)
    else:
        meta["frame"] = np.arange(n, dtype=np.int32)

    if "confidence" in z:
        meta["packet_confidence"] = z["confidence"].astype(float)
    else:
        meta["packet_confidence"] = 1.0

    return meta


def _norm_col_name(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_")


def read_layout_csv(path: Optional[str | Path]) -> Optional[pd.DataFrame]:
    if path is None or str(path).strip() == "":
        return None
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"layout csv not found: {path}")
    df = pd.read_csv(path)
    rename = {c: _norm_col_name(c) for c in df.columns}
    df = df.rename(columns=rename)
    return df


def infer_point_column(df: pd.DataFrame) -> str:
    candidates = ["point", "point_id", "pid", "name", "rx", "rx_id", "location", "loc"]
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"cannot infer point column from layout columns: {list(df.columns)}")


def infer_xyz_columns(df: pd.DataFrame, prefix: str = "") -> tuple[str, str, str]:
    candidates = [
        (f"{prefix}x", f"{prefix}y", f"{prefix}z"),
        ("x", "y", "z"),
        ("rx_x", "rx_y", "rx_z"),
        ("pos_x", "pos_y", "pos_z"),
        ("x_m", "y_m", "z_m"),
        ("x_cm", "y_cm", "z_cm"),
    ]
    for cols in candidates:
        if all(c in df.columns for c in cols):
            return cols
    raise KeyError(f"cannot infer xyz columns from layout columns: {list(df.columns)}")


def make_layout_lookup(
    layout_main: Optional[str | Path],
    layout_opposite: Optional[str | Path] = None,
    unit_scale: float = 1.0,
) -> dict[tuple[str, str], np.ndarray]:
    """Build lookup keyed by (view, point) -> rx_xyz.

    The loader normalizes column names and infers point/x/y/z columns. Use
    unit_scale when layout coordinates are stored in centimeters or millimeters.
    Example: centimeters to meters uses --layout-unit-scale 0.01.
    """
    lookup: dict[tuple[str, str], np.ndarray] = {}

    for view_name, path in [("main", layout_main), ("opposite", layout_opposite)]:
        df = read_layout_csv(path)
        if df is None:
            continue
        point_col = infer_point_column(df)
        x_col, y_col, z_col = infer_xyz_columns(df)
        for _, row in df.iterrows():
            point = str(row[point_col])
            xyz = np.array([row[x_col], row[y_col], row[z_col]], dtype=np.float64) * float(unit_scale)
            lookup[(view_name, point)] = xyz

    return lookup


def safe_mkdir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)
