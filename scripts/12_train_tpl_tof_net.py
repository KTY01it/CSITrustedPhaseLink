import argparse
from pathlib import Path
import sys
import random

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import pandas as pd

from src.training.group_dataset import CSIGroupDataset
from src.phase.tpl_tof_net import (
    TPLToFNet,
    group_coherence_torch,
    split_loss,
    consensus_residual_loss,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/canonical/canonical_k53_packets.npz")
    parser.add_argument("--out_ckpt", default="outputs/phaselink/tpl_tof_net_v3.pt")
    parser.add_argument("--metrics", default="outputs/metrics/tpl_tof_net_v3_train.csv")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--max_frames", type=int, default=128)
    parser.add_argument("--min_frames", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--n_iter", type=int, default=6)
    parser.add_argument("--lambda_split", type=float, default=0.5)
    parser.add_argument("--lambda_residual", type=float, default=0.15)
    parser.add_argument("--lambda_weight_reg", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device(args.device)

    dataset = CSIGroupDataset(
        canonical_npz=args.input,
        min_frames=args.min_frames,
        max_frames=args.max_frames,
        seed=args.seed,
    )

    model = TPLToFNet(
        num_bands=5,
        num_subcarriers=53,
        hidden=args.hidden,
        n_iter=args.n_iter,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    rows = []
    best_score = -1e9
    best_state = None
    best_step = -1

    print("=" * 80)
    print("TRAIN TPL-ToF-Net v3")
    print("=" * 80)
    print("num groups:", len(dataset))
    print("device:", device)
    print("steps:", args.steps)

    for step in range(1, args.steps + 1):
        sample = dataset.sample_group()

        csi = sample["csi"].to(device)
        band_id = sample["band_id"]

        # PyTorch from_numpy complex64 gives torch.complex64
        csi = csi.to(torch.complex64)

        out = model(csi, band_id)

        coh_before = out["coh_before"]
        coh_after = out["coh_after"]
        coh_gain = out["coh_gain"]

        # loss chính: maximize coherence sau correction
        loss_coh = -coh_after

        # split consistency
        loss_split = split_loss(model, csi, band_id)

        # regularize weights không collapse quá cực đoan
        w = out["w"]
        alpha = out["alpha"]

        loss_residual = consensus_residual_loss(out["csi_corr"])

        # encourage non-degenerate weight distribution
        w_reg = torch.mean((w - torch.mean(w)) ** 2)
        alpha_reg = torch.mean((alpha - torch.mean(alpha)) ** 2)

        loss_reg = w_reg + alpha_reg

        loss = (
            loss_coh
            + args.lambda_split * loss_split
            + args.lambda_residual * loss_residual
            + args.lambda_weight_reg * loss_reg
        )
        
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        # Best model score: ưu tiên coherence cao, split loss thấp.
        with torch.no_grad():
            score = float(
                (
                    coh_after
                    - args.lambda_split * loss_split
                    - args.lambda_residual * loss_residual
                ).detach().cpu()
            )
        if score > best_score:
            best_score = score
            best_step = step
            best_state = {
                "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "args": vars(args),
                "best_score": best_score,
                "best_step": best_step,
            }
    
        if step % 50 == 0 or step == 1:
            row = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "loss_coh": float(loss_coh.detach().cpu()),
                "loss_split": float(loss_split.detach().cpu()),
                "loss_residual": float(loss_residual.detach().cpu()),
                "loss_reg": float(loss_reg.detach().cpu()),
                "coh_before": float(coh_before.detach().cpu()),
                "coh_after": float(coh_after.detach().cpu()),
                "coh_gain": float(coh_gain.detach().cpu()),
                "band_id": band_id,
                "n": sample["n"],
                "key": str(sample["key"]),
            }
            rows.append(row)

            print(
                f"step={step:05d} "
                f"loss={row['loss']:.4f} "
                f"coh_before={row['coh_before']:.4f} "
                f"coh_after={row['coh_after']:.4f} "
                f"gain={row['coh_gain']:.4f} "
                f"split={row['loss_split']:.4f} "
                f"residual={row['loss_residual']:.4f}"
            )

    Path(args.out_ckpt).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics).parent.mkdir(parents=True, exist_ok=True)

    if best_state is None:
        best_state = {
            "model_state": model.state_dict(),
            "args": vars(args),
            "best_score": None,
            "best_step": None,
        }

    torch.save(best_state, args.out_ckpt)

    pd.DataFrame(rows).to_csv(args.metrics, index=False)
    
    print("Best step:", best_step)
    print("Best score:", best_score)
    print("\nSaved checkpoint:", args.out_ckpt)
    print("Saved metrics:", args.metrics)


if __name__ == "__main__":
    main()