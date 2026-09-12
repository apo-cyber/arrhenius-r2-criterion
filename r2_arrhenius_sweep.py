#!/usr/bin/env python
"""ideas/arrhenius-validity-check.md go/no-go 3: 「R²>0.95 の逆転」は固定設計の偶然か。

paper-A の datagen(paper_a/datagen/{temperature,config}.py)と同じ生成モデルを
自前で再実装し、成果物リポジトリに依存せず設計空間全体を掃く。

生成: ln(C/C0) = -k(T)·t + N(0, σ)。k(T) は Arrhenius / T^{±4}·Arrhenius(Tref=313.15K)、
      ln A は 25 °C の真値 t90 = 61.62 月に較正(規格 95%、目標 SL 30 月)。
判定: 各温度で ln(C/C0) を切片つき OLS → k̂。Arrhenius プロット (1/T, ln k̂) の OLS R²。
      加えて lack-of-fit χ²(残差² / Var(ln k̂), df = n_T − 2)を代替統計量として同時に測る。

    ~/dev/papers/bayesian-shelf-life-arrhenius/.venv/bin/python \
        ~/dev/generate_papers/docs/audit/r2_arrhenius_sweep.py
"""
from __future__ import annotations
import sys
import numpy as np
from scipy import stats

R = 8.314
TREF = 313.15
T90_TRUE = -np.log(0.90) / (-np.log(0.95) / 30.0)   # = 61.62 月(25 °C)
K25 = -np.log(0.95) / 30.0                          # datagen: 規格 95%, 目標 SL 30 月
TIMES = {3: [0., 3., 6.], 4: [0., 2., 4., 6.], 6: [0., 1., 2., 3., 4., 6.]}
TSETS = {3: [40., 50., 60.], 4: [40., 50., 60., 70.], 5: [40., 50., 60., 70., 80.]}
NEXP = {"arrhenius": 0.0, "concave": 4.0, "convex": -4.0}
EAS = (67.0, 80.0, 104.6)
SIGMAS = (0.01, 0.02, 0.05)
NREP = 4000
THRESH = 0.95


def k_true(temp_c, ea, n):
    """datagen と同じ較正: k(25°C) = K25 となる ln A。"""
    T = np.asarray(temp_c, float) + 273.15
    lna = np.log(K25) + ea * 1000 / (R * 298.15) - n * np.log(298.15 / TREF)
    return np.exp(lna - ea * 1000 / (R * T)) * (T / TREF) ** n


def simulate(nt, npts, sigma, ea, n, rng):
    """返り値: R²(OLS), lack-of-fit p 値, k̂≤0 で落ちた反復数。"""
    temps = np.array(TSETS[nt]); t = np.array(TIMES[npts])
    k = k_true(temps, ea, n)
    # (NREP, nT, npts)
    y = -k[None, :, None] * t[None, None, :] + rng.normal(0, sigma, (NREP, len(temps), len(t)))
    # 切片つき OLS: slope = Sxy/Sxx
    tc = t - t.mean()
    sxx = (tc ** 2).sum()
    slope = (y * tc).sum(-1) / sxx
    resid = y - y.mean(-1, keepdims=True) - slope[..., None] * tc
    s2 = (resid ** 2).sum(-1) / (len(t) - 2)
    khat = -slope
    se_k = np.sqrt(s2 / sxx)
    ok = (khat > 0).all(-1)
    khat, se_k = khat[ok], se_k[ok]
    lnk = np.log(khat)
    x = 1.0 / (temps + 273.15); xc = x - x.mean()
    b = (lnk * xc).sum(-1) / (xc ** 2).sum()
    fit = lnk.mean(-1, keepdims=True) + b[:, None] * xc
    ss_res = ((lnk - fit) ** 2).sum(-1)
    ss_tot = ((lnk - lnk.mean(-1, keepdims=True)) ** 2).sum(-1)
    r2 = 1 - ss_res / ss_tot
    # lack-of-fit: 重みつき残差 χ²(delta 法 Var(ln k̂) = (SE_k/k)²)
    v = (se_k / khat) ** 2
    W = 1 / v
    bw = ((lnk - (W * lnk).sum(-1, keepdims=True) / W.sum(-1, keepdims=True))
          * (x - (W * x).sum(-1, keepdims=True) / W.sum(-1, keepdims=True)) * W).sum(-1)
    xw = (W * x).sum(-1) / W.sum(-1); yw = (W * lnk).sum(-1) / W.sum(-1)
    bw = bw / ((W * (x - xw[:, None]) ** 2).sum(-1))
    fitw = yw[:, None] + bw[:, None] * (x - xw[:, None])
    chi2 = (((lnk - fitw) ** 2) / v).sum(-1)
    p_lof = stats.chi2.sf(chi2, len(temps) - 2)
    return r2, p_lof, NREP - ok.sum()


def main():
    rng = np.random.default_rng(20260912)
    print("=" * 96)
    print("§A 設計点ごとの R²>0.95 通過率 — Ea=80 で真の形を揃えた比較(逆転が再現するか)")
    print("    括弧内は lack-of-fit χ² の α=0.05 棄却率(= 代替統計量の検出力)")
    print("=" * 96)
    hdr = f"{'n_T':>4}{'n_pts':>6}{'σ':>6} | {'Arrhenius':>18}{'concave n=+4':>18}{'convex n=-4':>18} | {'逆転':>5}{'k≤0落':>7}"
    print(hdr)
    rev = tot = 0
    for nt in (3, 4, 5):
        for npts in (3, 4, 6):
            for s in SIGMAS:
                row = {}; drops = 0
                for name, n in NEXP.items():
                    r2, p, d = simulate(nt, npts, s, 80.0, n, rng)
                    row[name] = (np.mean(r2 > THRESH), np.median(r2), np.mean(p < 0.05)); drops = max(drops, d)
                flag = row["concave"][0] > row["arrhenius"][0]
                rev += flag; tot += 1
                cells = "".join(f"{row[k][0]:>8.1%} ({row[k][2]:>5.1%})" for k in NEXP)
                print(f"{nt:>4}{npts:>6}{s:>6} | {cells} | {'★' if flag else '':>5}{drops:>7}")
    print(f"  → concave > Arrhenius の逆転: {rev}/{tot} 設計点")

    print("\n" + "=" * 96)
    print("§B 真の Arrhenius のみ: R² は Ea と温度幅で決まる(通過率 / R² 中央値)")
    print("=" * 96)
    print(f"{'n_T':>4}{'n_pts':>6}{'σ':>6} | " + "".join(f"{'Ea='+str(e):>18}" for e in EAS))
    for nt in (3, 4, 5):
        for npts in (4,):
            for s in SIGMAS:
                cells = ""
                for e in EAS:
                    r2, _, _ = simulate(nt, npts, s, e, 0.0, rng)
                    cells += f"{np.mean(r2 > THRESH):>9.1%} {np.median(r2):>8.3f}"
                print(f"{nt:>4}{npts:>6}{s:>6} | {cells}")

    print("\n" + "=" * 96)
    print("§C 同じ R² 通過率になる (Ea, 真の形) の組 — 「R² は Ea の代理」の直接確認")
    print("    n_T=3, n_pts=4, σ=0.02。Ea を連続に振り、各形の通過率 50% となる Ea を内挿")
    print("=" * 96)
    for name, n in NEXP.items():
        eas = np.arange(55, 130, 5.0); pr = []
        for e in eas:
            r2, _, _ = simulate(3, 4, 0.02, e, n, rng); pr.append(np.mean(r2 > THRESH))
        pr = np.array(pr)
        i = np.searchsorted(pr, 0.5)
        e50 = np.interp(0.5, pr[i - 1:i + 1], eas[i - 1:i + 1]) if 0 < i < len(pr) else float("nan")
        print(f"  {name:<10} 通過率 50% となる Ea ≈ {e50:6.1f} kJ/mol   "
              f"(通過率: " + " ".join(f"{e:.0f}:{p:.0%}" for e, p in zip(eas[::3], pr[::3])) + ")")

    print("\n" + "=" * 96)
    print("§D 加速域内での見かけの Ea(真の形を Arrhenius で当てはめたときの傾き)")
    print("=" * 96)
    for nt in (3, 4, 5):
        temps = np.array(TSETS[nt]); x = 1 / (temps + 273.15)
        out = []
        for name, n in NEXP.items():
            b = np.polyfit(x, np.log(k_true(temps, 80.0, n)), 1)[0]
            out.append(f"{name}: {-b * R / 1000:6.1f}")
        print(f"  n_T={nt} ({temps.min():.0f}–{temps.max():.0f} °C)  " + "   ".join(out))
    print("  → concave(n=+4)は見かけの Ea が高い = 温度間の ln k の広がり(SS_tot)が大きい = R² が上がる")
    print("     convex(n=-4)はその逆。R² の差は曲率ではなく見かけの Ea の差で説明される")

    print("\n" + "=" * 96)
    print("§E 恒等式: 単回帰では t² = R²(n−2)/(1−R²)。ゆえに R²>0.95 は SE(Êa)/Êa の上限と同値")
    print("=" * 96)
    for nt in (3, 4, 5, 6):
        rel = np.sqrt((1 - THRESH) / (THRESH * (nt - 2)))
        print(f"  n_T={nt}: R²>{THRESH} ⇔ SE(Êa)/Êa < {rel:.1%}  (両側 95% 幅 ±{1.96*rel:.0%})")
    print("  → 判定基準は見かけの Ea の相対精度に対する閾値であり、モデル形には一切触れていない")


if __name__ == "__main__":
    main()
