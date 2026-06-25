\# CSITrustedPhaseLink



Physics-constrained phase-link module for raw WiFi CSI sensing.



\## Current Scope



This project implements a ToF-preserving trusted phase-link front-end for raw CSI data. The current module focuses on intra-point packet-wise phase stabilization, not full coherent AoA/AoD aperture reconstruction.



\## Main Components



\- Canonical CSI construction with fixed K=53 subcarriers.

\- Common-offset phase-link baseline.

\- Robust consensus common-offset phase-link.

\- Learned TPL-ToF-Net with packet/subcarrier reliability learning.

\- ToF-slope stability audit.

\- Aperture synchronization audit.



\## Key Design Rule



The phase correction is constrained as:



```text

H\_corr\_i(k) = H\_i(k) \* exp(-j \* delta\_i)

