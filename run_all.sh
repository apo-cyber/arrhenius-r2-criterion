#!/usr/bin/env bash
# Reproduce every table and figure of the paper. ~15 min on a laptop (20,000-replicate real-time tables dominate).
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
mkdir -p output figures
$PY r2_width.py            > output/r2_width.txt            # Table 1, Figure 1, residual-vs-SE (seconds)
$PY r2_arrhenius_sweep.py  > output/r2_arrhenius_sweep.txt  # design-space sweep; Section 4.1-4.3 (~2 min)
$PY robustness_checks.py   > output/robustness_checks.txt   # Table S1; anchors, Ea, n, thresholds, LOF-F, kinetic order, seeds (~5 min)
$PY r2_break.py            > output/r2_break.txt            # Table 3 (~2 min)
$PY realtime_power.py      > output/realtime_power.txt      # Table 2, Table S2, Section 4.5 (~5 min)
PAPERB_FIG_DIR=figures $PY make_figures.py                  # Figures 1, 2, S1 as PNG/PDF/TIFF (~3 min)
echo "done: output/*.txt, figures/*"
