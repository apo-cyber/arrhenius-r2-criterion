#!/usr/bin/env python
"""paper-B の図 3 枚。出力は paper-B/figures/。数値の定義は r2_arrhenius_sweep / robustness_checks / r2_width / realtime_power と同一。

  fig2_passrate_apparent_ea.png  R²>0.95 の通過率を「見かけの Ea」に対して描く。Arrhenius / concave / convex が 1 本の曲線に乗る
                                 (= R² は形を見ていない)。LOF-F の棄却率は size に張り付く。§4.1–4.2(旧 表 3 は SI 表 S1 へ)
  fig1_r2_nomogram.png           R² → SE(Êa)/Êa → 25 °C の t90 の 1 SE 幅(表 1 の連続版。実務者が自分の R² を読む図)
  fig2_realtime_power.png        実時間 m ヶ月で X 倍の申請値を棄却できる確率(表 2 の m 依存版)

    ~/dev/papers/bayesian-shelf-life-arrhenius/.venv/bin/python docs/audit/make_figures.py
"""
from __future__ import annotations
import sys, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import t as tdist, f as fdist

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import robustness_checks as rc          # lnk_form, fit, FirstOrder, TSETS, TIMES
import realtime_power as rp             # K25, ICH, ols_slope

# 出力先: 環境変数 PAPERB_FIG_DIR > リポジトリ内 paper-B/figures > スクリプト隣の figures/(公開パッケージで単独実行するとき)
import os
_default = HERE.parent.parent / "paper-B" / "figures"
OUT = pathlib.Path(os.environ.get("PAPERB_FIG_DIR", _default if _default.parent.exists() else HERE / "figures"))
OUT.mkdir(exist_ok=True)
R = 8.314
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 100, "savefig.dpi": 300,
                     "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                     "mathtext.fontset": "dejavusans", "pdf.fonttype": 42, "ps.fonttype": 42})   # J Pharm Sci: Helvetica/Arial, fonts embedded


def save(fig, stem):
    """PNG(原稿の LaTeX 用)+ PDF(ベクタ、投稿用)+ TIFF 600 dpi(投稿用の代替。線画+ハーフトーン混合は ≥500 dpi)。"""
    fig.savefig(OUT / f"{stem}.png")
    fig.savefig(OUT / f"{stem}.pdf")
    fig.savefig(OUT / f"{stem}.tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    # matplotlib の TIFF は RGBA(全画素不透明でもアルファが残る)。投稿用に白背景へ平坦化して RGB にする
    from PIL import Image
    with Image.open(OUT / f"{stem}.tiff") as im:
        flat = Image.new("RGB", im.size, "white"); flat.paste(im, mask=im.getchannel("A"))
    flat.save(OUT / f"{stem}.tiff", compression="tiff_lzw", dpi=(600, 600))
GREY = "#555555"


def apparent_ea(temps, ea, n):
    """真の ln k(T) に直線を当てたときの傾きから見かけの Ea。"""
    x = 1 / (np.asarray(temps) + 273.15); y = rc.lnk_form(temps, ea, n)
    return -np.polyfit(x, y, 1)[0] * R / 1000


def fig_passrate():
    rng = np.random.default_rng(1)
    sigma = 0.02
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4), sharey=True)
    for ax, nt in zip(axes, (3, 4)):
        temps = rc.TSETS[nt]
        for n, lab, mk, col in ((0.0, "Arrhenius truth", "o", "black"),
                                (4.0, "concave truth ($n=+4$)", "^", "#7f7f7f"),
                                (-4.0, "convex truth ($n=-4$)", "v", "#bfbfbf")):
            eas = np.arange(45, 126, 5.0)
            pr, lof, app = [], [], []
            for ea in eas:
                p, _, pf, _ = rc.pass_rate(temps, 4, sigma, ea, n, rng)
                pr.append(p); lof.append(pf); app.append(apparent_ea(temps, ea, n))
            ax.plot(app, np.array(pr) * 100, mk + "-", color=col, ms=4.5, lw=1, label=lab, mec="black", mew=0.4)
            ax.plot(app, np.array(lof) * 100, mk + ":", color=col, ms=3.5, lw=0.8, mfc="none", mec="black", mew=0.4)
        ax.axhline(5, color=GREY, lw=0.6, ls="--")
        ax.text(124, 7, "lack-of-fit $F$, $\\alpha$=0.05", ha="right", va="bottom", fontsize=7, color=GREY)
        ax.set_title(f"$n_T={nt}$ ({int(temps.min())}–{int(temps.max())} °C), 4 time points, $\\sigma$=0.02", fontsize=9)
        ax.set_xlim(45, 126); ax.set_ylim(0, 100)
    axes[0].set_ylabel("Rate (%)")
    for ax in axes: ax.set_xlabel("Apparent $E_a$ over the accelerated range (kJ/mol)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, 0.0),
               title="solid: $R^2>0.95$ pass rate;  dotted: lack-of-fit ($F$) rejection rate", title_fontsize=7.5)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    save(fig, "fig2_passrate_apparent_ea"); plt.close(fig)


def fig_nomogram():
    ea = 80.0; x0 = 1 / 298.15
    r2 = 1 - np.logspace(np.log10(0.10), np.log10(0.0005), 300)      # 0.90 … 0.9995
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9))
    for nt, ls in ((3, "-"), (4, "--"), (5, "-."), (6, ":")):
        temps = 40 + 10 * np.arange(nt)
        x = 1 / (temps + 273.15); xb = x.mean(); sxx = ((x - xb) ** 2).sum()
        rel = np.sqrt((1 - r2) / (r2 * (nt - 2)))
        se_ln = rel * ea * 1000 / R * np.sqrt(sxx / nt + (x0 - xb) ** 2)
        axes[0].plot(1 - r2, rel * 100, ls, color="black", lw=1, label=f"$n_T={nt}$")
        axes[1].plot(1 - r2, np.exp(se_ln), ls, color="black", lw=1, label=f"$n_T={nt}$")
        axes[2].plot(1 - r2, np.exp(-tdist.ppf(0.95, nt - 2) * se_ln), ls, color="black", lw=1, label=f"$n_T={nt}$")
    for ax in axes:
        ax.set_xscale("log"); ax.set_xlim(0.10, 0.0005); ax.set_xlabel("$R^2$ of the Arrhenius plot")
        ticks = [0.10, 0.05, 0.01, 0.002]
        ax.set_xticks(ticks); ax.set_xticklabels([f"{1-t:.3g}" for t in ticks])
        for thr in (0.95, 0.99):
            ax.axvline(1 - thr, color=GREY, lw=0.6, ls="--")
    axes[0].set_ylabel("SE$(\\hat E_a)/\\hat E_a$ (%)"); axes[0].set_ylim(0, 40)
    axes[0].text(0.05, 38, "0.95", ha="center", va="top", fontsize=7, color=GREY)
    axes[0].text(0.01, 38, "0.99", ha="center", va="top", fontsize=7, color=GREY)
    axes[1].set_ylabel("Factor on $t_{90}$(25 °C) per 1 SE"); axes[1].set_ylim(1, 2.6)
    axes[1].text(0.05, 2.55, "0.95", ha="center", va="top", fontsize=7, color=GREY)
    axes[1].text(0.01, 2.55, "0.99", ha="center", va="top", fontsize=7, color=GREY)
    axes[1].legend(fontsize=7, frameon=False, loc="upper right")
    axes[2].set_ylabel("One-sided 95% lower bound / estimate"); axes[2].set_ylim(0, 1)
    axes[2].text(0.05, 0.98, "0.95", ha="center", va="top", fontsize=7, color=GREY)
    axes[2].text(0.01, 0.98, "0.99", ha="center", va="top", fontsize=7, color=GREY)
    axes[0].set_title("(a) What $R^2$ certifies about $E_a$", fontsize=8.5, loc="left")
    axes[1].set_title("(b) …as width of $t_{90}$(25 °C), $E_a$=80", fontsize=8.5, loc="left")
    axes[2].set_title("(c) …as 95% lower bound", fontsize=8.5, loc="left")
    fig.tight_layout()
    save(fig, "fig1_r2_nomogram"); plt.close(fig)


def realtime_power(sigma, m, B, r, X, mode, rng, nrep=20000):
    """実時間データ単独で申請値(真値の X 倍)を棄却する確率。mode: 'rep' 反復扱い / 'q1e' プール判定 / 'unit' バッチ単位。"""
    t = rp.ICH[rp.ICH <= m]; npts = len(t); tc = t - t.mean(); sxx = (tc ** 2).sum()
    y = -rp.K25 * t[None, None, :] + rng.normal(0, sigma / np.sqrt(r), (nrep, B, npts))
    b, se = rp.ols_slope(t, y); khat = -b; kbar = khat.mean(1)
    kcl = rp.K25 / X
    if mode == "rep":
        s = np.sqrt((se ** 2).sum(1)) / B; crit = tdist.ppf(0.975, B * npts - 2)
        return np.mean(np.abs((kbar - kcl) / s) > crit)
    s_b = khat.std(1, ddof=1) / np.sqrt(B); crit_b = tdist.ppf(0.975, B - 1)
    rej_b = np.abs((kbar - kcl) / s_b) > crit_b
    if mode == "unit":
        return rej_b.mean()
    rss_sep = (se ** 2 * sxx * (npts - 2)).sum(1)
    yc = y - y.mean(-1, keepdims=True)
    rss_com = ((yc + kbar[:, None, None] * tc) ** 2).sum((1, 2))
    F = ((rss_com - rss_sep) / (B - 1)) / (rss_sep / (B * (npts - 2)))
    pool = fdist.sf(F, B - 1, B * (npts - 2)) > 0.25
    s_c = np.sqrt(rss_com / (B * npts - B - 1) / (B * sxx)); crit_c = tdist.ppf(0.975, B * npts - B - 1)
    return np.mean(np.where(pool, np.abs((kbar - kcl) / s_c) > crit_c, rej_b))


def fig_power():
    rng = np.random.default_rng(3)
    ms = (12, 18, 24, 36)
    designs = (("1 batch, single assay", 1, 1, "rep", GREY, "-"),
               ("3 batches, single assay, batches as replicates", 3, 1, "rep", "black", "-"),
               ("3 batches, single assay, ICH Q1E poolability gate", 3, 1, "q1e", "black", "--"),
               ("3 batches, duplicate assay, batches as replicates", 3, 2, "rep", "black", ":"))
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.6), sharex=True, sharey=True)
    for j, sigma in enumerate((0.01, 0.02, 0.05)):
        for i, X in enumerate((1.4, 2.0)):
            ax = axes[i, j]
            for lab, B, r, mode, col, ls in designs:
                pw = [realtime_power(sigma, m, B, r, X, mode, rng) * 100 for m in ms]
                ax.plot(ms, pw, ls, color=col, lw=1, marker="o", ms=2.5, label=lab)
            ax.axhline(5, color=GREY, lw=0.5, ls="--")
            ax.set_ylim(0, 100); ax.set_xticks(ms)
            if i == 0: ax.set_title(f"$\\sigma$ = {sigma:.2f}", fontsize=9)
            if j == 0: ax.set_ylabel(f"Rejection probability (%)\n$X$ = {X}")
    axes[1, 1].set_xlabel("Months of real-time data at 25 °C (ICH long-term schedule)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    save(fig, "figS1_realtime_power"); plt.close(fig)


if __name__ == "__main__":
    fig_nomogram(); print("fig1"); fig_passrate(); print("fig2"); fig_power(); print("figS1")
