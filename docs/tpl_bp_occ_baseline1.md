# TPL-BP-Occ Baseline 1: Phase-Linked Backprojection

This baseline starts after TPL-ToF phase-link. It is designed for the current CSI dataset where repeated packets at the same measurement point can be phase-linked, but scan points do not form a frame-synchronous coherent virtual aperture.

## Physical rule

- Inside one group `(view, array, point, f0)`: coherent packet fusion is allowed after phase-link.
- Across different measurement points: only non-coherent evidence fusion is used.
- Direct AoA/AoD from scan-point phase differences is not used in this baseline.

## Group fusion

For packet `i` and subcarrier `k` in group `g`:

```text
Hbar_g(k) = sum_i alpha_gi Hcorr_gi(k) / sum_i alpha_gi
```

where `alpha_gi` is packet confidence from TPL-ToF-Net when available.

## Bistatic response

For voxel `x`, Tx position `t`, and Rx position `r_g`:

```text
d_g(x)   = ||x - t||_2 + ||x - r_g||_2
tau_g(x) = d_g(x) / c
R_g(x)   = | sum_k w_gk Hbar_g(k) exp(j 2pi f_k tau_g(x)) |^2
```

By default, the script uses subcarrier-offset frequencies `f_k = sub_idx[k] * 312.5 kHz` rather than absolute carrier frequencies. This is safer for the current non-synchronous scanned data. Absolute-frequency mode is available only as a diagnostic flag.

## Cross-point fusion

```text
E(x) = sum_g beta_g R_g(x)
```

where `beta_g` is the group confidence.

## Commands

Build group-fused CSI:

```powershell
python scripts/15_build_group_fused_csi.py `
  --input outputs/phaselink/tpl_tof_net_v3.npz `
  --output outputs/reconstruction/group_fused_tpl_v3.npz `
  --layout-main D:/Project/WifiSenssing/Data/Data26_11/layout_positions.csv `
  --layout-opposite D:/Project/WifiSenssing/Data/Data26_11/layout_positions_txop_rxop.csv `
  --layout-unit-scale 1.0 `
  --use corr
```

Build evidence volume. Replace Tx and grid values with your calibrated coordinate system:

```powershell
python scripts/16_build_evidence_volume.py `
  --input outputs/reconstruction/group_fused_tpl_v3.npz `
  --output outputs/reconstruction/evidence_tpl_v3.npz `
  --tx-x 0 --tx-y 0 --tx-z 0 `
  --x-min -1 --x-max 1 `
  --y-min -1 --y-max 1 `
  --z-min 0 --z-max 2 `
  --grid-step 0.05 `
  --normalize-per-group
```

Evaluate evidence metrics:

```powershell
python scripts/17_eval_evidence_volume.py `
  --input outputs/reconstruction/evidence_tpl_v3.npz `
  --out_csv outputs/reconstruction/evidence_tpl_v3_metrics.csv
```

## Required checks

1. Run the same pipeline for raw CSI by passing `--input outputs/canonical/canonical_k53_packets.npz --use raw` if the raw canonical file includes compatible metadata, or use a phase-link npz containing `csi_raw`.
2. Compare raw, consensus v2, and learned v3 evidence volumes using peak sharpness, PSR, entropy, and compactness.
3. Do not interpret direct cross-point phase as AoA until an inter-point aperture synchronization module exists.
