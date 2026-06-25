from __future__ import annotations

from pathlib import Path
from typing import Optional
import re

import numpy as np
import pandas as pd


REQUIRED_PHASE_NPZ_KEYS = [
    "sub_idx",
    "view",
    "array",
    "point",
    "f0",
]


def normalize_point_id(value: object) -> str:
    """Normalize point/node identifiers across npz metadata and layout CSVs.

    Handles common mismatches:
      1 vs 1.0
      "P01" vs "p1"
      "rx_01" vs "rx1"
    """
    if value is None:
        return ""
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)) and np.isfinite(value):
        if abs(float(value) - round(float(value))) < 1e-9:
            return str(int(round(float(value))))
        return str(float(value)).strip().lower()

    s = str(value).strip().lower()
    if s.endswith(".0"):
        base = s[:-2]
        if base.lstrip("+-").isdigit():
            s = base

    # collapse separators only; keep alphabetic prefixes such as p/rx when present
    s = re.sub(r"[\s_\-]+", "", s)

    # normalize p01 -> p1, rx01 -> rx1, node01 -> node1
    m = re.match(r"^([a-z]+)0*([0-9]+)$", s)
    if m:
        return f"{m.group(1)}{int(m.group(2))}"

    # normalize 001 -> 1
    if s.isdigit():
        return str(int(s))

    return s


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
    raw_point = z["point"].astype(str)
    meta = pd.DataFrame({
        "view": z["view"].astype(str),
        "array": z["array"].astype(str),
        "point": raw_point,
        "point_norm": [normalize_point_id(p) for p in raw_point],
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
    """Infer the layout node identifier column.

    The Data26_11 layouts use columns:
      node_type,node_id,x_m,y_m,z_m,notes
    so node_id is intentionally included near the front.
    """
    candidates = [
        "point",
        "point_id",
        "node_id",
        "node",
        "pid",
        "name",
        "rx",
        "rx_id",
        "location",
        "loc",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"cannot infer point column from layout columns: {list(df.columns)}")


def infer_xyz_columns(df: pd.DataFrame, prefix: str = "") -> tuple[str, str, str]:
    candidates = [
        (f"{prefix}x", f"{prefix}y", f"{prefix}z"),
        ("x_m", "y_m", "z_m"),
        ("x", "y", "z"),
        ("rx_x", "rx_y", "rx_z"),
        ("pos_x", "pos_y", "pos_z"),
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
    """Build lookup keyed by (view, normalized_point) -> rx_xyz."""
    lookup: dict[tuple[str, str], np.ndarray] = {}

    for view_name, path in [("main", layout_main), ("opposite", layout_opposite)]:
        df = read_layout_csv(path)
        if df is None:
            continue
        point_col = infer_point_column(df)
        x_col, y_col, z_col = infer_xyz_columns(df)
        for _, row in df.iterrows():
            point_norm = normalize_point_id(row[point_col])
            xyz = np.array([row[x_col], row[y_col], row[z_col]], dtype=np.float64) * float(unit_scale)
            lookup[(view_name, point_norm)] = xyz

    return lookup


def safe_mkdir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)
