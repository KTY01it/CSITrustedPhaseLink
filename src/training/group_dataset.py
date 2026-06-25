import numpy as np
import pandas as pd
import torch


BANDS = [5180.0, 5200.0, 5220.0, 5240.0, 5280.0]
BAND_TO_ID = {b: i for i, b in enumerate(BANDS)}


def load_canonical_npz(path):
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


class CSIGroupDataset:
    """
    Dataset trả về từng group:
      group = (view, array, point, f0)

    Mỗi sample:
      csi: complex tensor [N, 53]
      band_id
      meta info
    """

    def __init__(
        self,
        canonical_npz,
        min_frames=8,
        max_frames=128,
        seed=0,
        include_arrays=None,
    ):
        self.csi, self.sub_idx, self.meta = load_canonical_npz(canonical_npz)
        self.min_frames = int(min_frames)
        self.max_frames = int(max_frames)
        self.rng = np.random.default_rng(seed)

        df = self.meta.copy()

        if include_arrays is not None:
            include_arrays = set(str(x) for x in include_arrays)
            df = df[df["array"].astype(str).isin(include_arrays)]

        self.groups = []
        grouped = df.groupby(["view", "array", "point", "f0"], sort=False)

        for gkey, gdf in grouped:
            idx = gdf.index.to_numpy()
            if len(idx) < self.min_frames:
                continue

            f0 = float(gkey[3])
            if f0 not in BAND_TO_ID:
                continue

            self.groups.append({
                "key": gkey,
                "idx": idx,
                "band_id": BAND_TO_ID[f0],
                "n": len(idx),
            })

        if len(self.groups) == 0:
            raise RuntimeError("No valid groups for training.")

    def __len__(self):
        return len(self.groups)

    def sample_group(self):
        g = self.groups[self.rng.integers(0, len(self.groups))]
        idx = g["idx"]

        if len(idx) > self.max_frames:
            idx = self.rng.choice(idx, size=self.max_frames, replace=False)

        # Sort theo frame để ổn định hơn
        frames = self.meta.loc[idx, "frame"].to_numpy()
        idx = idx[np.argsort(frames)]

        csi_np = self.csi[idx]
        csi = torch.from_numpy(csi_np)

        return {
            "csi": csi,
            "band_id": int(g["band_id"]),
            "key": g["key"],
            "n": len(idx),
        }