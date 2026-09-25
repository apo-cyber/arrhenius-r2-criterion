# Simulation code for "The Arrhenius-plot R-squared criterion bounds the precision of the activation energy and cannot detect non-Arrhenius behaviour: an exact identity and a simulation study of accelerated stability designs"

[![DOI](https://zenodo.org/badge/1367340138.svg)](https://doi.org/10.5281/zenodo.22726459)

Yasushi Arai, 2026. Companion code archive for the manuscript (under submission). Version 1.0.0 of this archive was released under the manuscript's earlier title, "Does the Arrhenius-plot R-squared criterion detect non-Arrhenius behaviour in accelerated stability data?"; the code and outputs are unchanged.

Six short, self-contained Python scripts (NumPy + SciPy; Matplotlib for the figures). No measured stability data are used;
every table and figure in the paper is produced from the generating model described in the paper's Methods section,
which is the same first-order / modified-Arrhenius model as the companion study's public repository
(apo-cyber/bayesian-shelf-life-arrhenius, https://doi.org/10.5281/zenodo.20576829), re-implemented here independently.

## What produces what

| Script | Paper | What it does |
|---|---|---|
| `r2_width.py` | Table 1, Figure 1 (incl. the one-sided $t$ lower bound); Sections 4.1/4.3 (within-range residual vs SE of $\ln\hat k$) | The identity $t^2 = R^2(n-2)/(1-R^2)$ → relative SE of $E_a$ → prediction SE of $\ln t_{90}$ at 25 °C (unweighted least squares); plus the maximum within-range residual of the $T^{\pm4}$ truths and the per-temperature SE of $\ln\hat k$ |
| `r2_arrhenius_sweep.py` | Sections 4.1–4.3 (design-space statements) | Pass rate of $R^2>0.95$ and lack-of-fit ($\chi^2$) rejection over $n_T\in\{3,4,5\}$, $n_{pts}\in\{3,4,6\}$, $\sigma\in\{0.01,0.02,0.05\}$ for Arrhenius / concave / convex truths; pass rate vs $E_a$; apparent $E_a$ |
| `robustness_checks.py` | Table S1; Sections 4.1–4.3 | Calibration anchors with matched pure-Arrhenius controls (A), $E_a$ levels (B), deviation size $n=\pm1,\pm2,\pm4$ (C), thresholds 0.90/0.95/0.99 (D), lack-of-fit $F$ vs $\chi^2$ (E), zero-/second-order truths (F), seed stability (G) |
| `r2_break.py` | Table 3 | Two-segment Arrhenius truths (break inside / below the accelerated range) with matched controls |
| `realtime_power.py` | Table 2, Table S2, Section 4.5 (batch variation, one-sided variant); Figure S1 data | Probability that real-time data at 25 °C reject a claimed shelf life $X$ times the true value: duration, batches × replicates, batch-to-batch variation with three treatments of the batches (replicates / ICH Q1E poolability gate / batch as unit) |
| `make_figures.py` | Figures 1, 2 and S1 | Imports `robustness_checks` and `realtime_power`; writes PNG (draft), PDF (vector, submission) and TIFF 600 dpi |

Each script prints its tables to stdout; the `.txt` files in `output/` are the runs used for the paper (seeds are fixed in the scripts).
Script docstrings and printed table headers are in Japanese (the working language of the project); column meanings are given in this README and in the paper.

## Run

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
./run_all.sh            # ~15 min; writes output/*.txt and figures/*
```

Tested with Python 3.12, NumPy 2.4, SciPy 1.17, Matplotlib 3.10. Monte Carlo sizes: 4,000 replicates per design point (pass rates), 20,000 (real-time power); the binomial standard error of a pass rate is therefore < 0.8 percentage points.

## Model in one paragraph

$\ln(C/C_0) = -k(T)\,t + \varepsilon$, $\varepsilon\sim N(0,\sigma^2)$; $\ln k(T) = \ln A - E_a/(RT) + n\ln(T/313.15\,\mathrm{K})$ with $n=0$ (Arrhenius), $n>0$ concave, $n<0$ convex; $E_a=80$ kJ/mol unless varied; the three forms share $k$ at 25 °C (true $t_{90}=61.6$ months) unless the calibration anchor is varied. Per temperature: ordinary least squares on content–time → $\hat k$; Arrhenius plot: unweighted least squares of $\ln\hat k$ on $1/T$ → $R^2$. Real-time: ICH long-term schedule {0,3,6,9,12,18,24,36} months at 25 °C, $B$ batches, $r$ replicate assays, two-sided $t$-test of $\hat k_{25}$ against the claimed rate $k_{25}/X$.

## Licence and citation

Code: MIT (see `LICENSE`). Please cite the paper (reference to be added on publication) and this archive: concept DOI https://doi.org/10.5281/zenodo.22726459 (v1.0.0: https://doi.org/10.5281/zenodo.22726460). See `CITATION.cff`.
