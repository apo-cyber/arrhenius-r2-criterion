#!/usr/bin/env python
"""paper-B 表 1 の材料: R²>0.95 が保証する SE(Êa)/Êa(Taira Eq. 17 の恒等式)を
25 °C の外挿 ln t90 の幅に伝播する。

恒等式(単回帰、n 点): t² = R²(n−2)/(1−R²) → SE(Êa)/Êa < sqrt((1−R²₀)/(R²₀(n−2)))。
外挿幅(重みなし OLS の予測分散、σ̂ を SE(Êa) で表す):
  Var(ln k̂25) = σ̂² [1/n + (x0−x̄)²/Sxx] = (SE(Êa)/R)² [Sxx/n + (x0−x̄)²],  x = 1/T
  → SE(ln t̂90(25)) = (SE(Êa)/R) · sqrt(Sxx/n + (x0−x̄)²)          ... (a) OLS 予測(正)
  参考 (b): 最低温度 40 °C を既知として傾きだけ伝播 = (SE(Êa)/R)·(1/298.15 − 1/313.15)。
  handoff の「1 SE で 1.4 倍、2 SE で 2 倍」は (b)。(a) の方が広い(切片の不確かさが加わる)。
真値 Ea = 80 kJ/mol。温度は 40, 50, …(n_T 点)。

    ~/dev/papers/bayesian-shelf-life-arrhenius/.venv/bin/python docs/audit/r2_width.py
"""
import numpy as np
from scipy.stats import t as tdist

R = 8.314
EA = 80.0
X0 = 1 / 298.15


def main():
    print("R²>R²₀ が保証する SE(Êa)/Êa と、それを 25 °C の t90 に伝播した幅(Ea=80 kJ/mol、40 °C から 10 °C 刻み)")
    print("幅 = exp(z·SE(ln t90))。(a) OLS 予測分散(切片込み)、(b) 40 °C 既知・傾きのみ(参考)\n")
    for r2 in (0.90, 0.95, 0.99):
        print(f"=== R²₀ = {r2} ===")
        print(f"{'n_T':>4} {'温度 (°C)':>14} {'SE/Ea':>7} {'±1.96SE':>8} | {'(a) SE ln t90':>13} {'1SE':>5} {'2SE':>5} {'z95%':>5} | {'t(.95)':>6} {'片側下限':>8} | {'(b) SE':>7} {'1SE':>5} {'2SE':>5}")
        for nt in (3, 4, 5, 6):
            temps = 40 + 10 * np.arange(nt)
            x = 1 / (temps + 273.15); xb = x.mean(); sxx = ((x - xb) ** 2).sum()
            rel = np.sqrt((1 - r2) / (r2 * (nt - 2)))
            se_ea = rel * EA * 1000
            d_a = np.sqrt(sxx / nt + (X0 - xb) ** 2)
            se_a = se_ea / R * d_a
            se_b = se_ea / R * (X0 - 1 / 313.15)
            t1 = tdist.ppf(0.95, nt - 2)   # ICH Q1E 流: 下限規格のみの属性は片側 95% 下限。自由度は n_T−2(cmc-platform も t に変更済み)
            print(f"{nt:>4} {'–'.join(map(str, temps)):>14} {rel:>7.1%} {'±'+format(1.96*rel,'.0%'):>8} | "
                  f"{se_a:>13.2f} {np.exp(se_a):>5.2f} {np.exp(2*se_a):>5.2f} {np.exp(1.96*se_a):>5.2f} | "
                  f"{t1:>6.2f} {np.exp(-t1*se_a):>8.3f} | "
                  f"{se_b:>7.2f} {np.exp(se_b):>5.2f} {np.exp(2*se_b):>5.2f}")
        print()
    curvature_vs_noise()
    print("読み方: n_T=3 で R²>0.95 を通った Arrhenius プロットが名目上保証するのは、25 °C の t90 が 1 SE で約 1.8 倍、")
    print("        z 近似 95% 幅で約 3 倍動く範囲まで。しかもこの SE 自体が残差 1 自由度の推定(Taira: n=3 では 25% の確率で 1/10 未満に出る)。")
    print("        '片側下限' = t(0.95, n_T−2) による片側 95% 下限 / 点推定(ICH Q1E の下限規格の流儀)。n_T=3 では点推定の 2%、R²=0.99 でも 19%。")


def curvature_vs_noise():
    """本文 §4.1 / §4.3 の材料(2026-09-12 追加): 真の ln k(T^{±4})に直線を当てたときの域内残差の最大値と、
    各温度の ln k̂ の SE(= SE(k̂)/k̂、delta 法)の中央値。前者 ≈ 10^-3、後者 0.04–1.2 → 曲率はノイズに埋もれる。"""
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import robustness_checks as rc
    print("\n=== 真の ln k(n=±4, Ea=80)に直線を当てた域内残差の最大値(log スケール)===")
    for nt in (3, 4, 5):
        temps = np.array([40, 50, 60, 70, 80][:nt], float); x = 1 / (temps + 273.15)
        r = [np.abs(rc.lnk_form(temps, 80., n) - np.polyval(np.polyfit(x, rc.lnk_form(temps, 80., n), 1), x)).max() for n in (4., -4.)]
        print(f"  n_T={nt} ({int(temps.min())}–{int(temps.max())} °C): concave {r[0]:.2e}, convex {r[1]:.2e}")
    print("=== 各温度の SE(ln k̂) の中央値(Arrhenius, Ea=80, k(25) 固定, 4000 反復)===")
    rng = np.random.default_rng(0)
    for nt in (3, 5):
        temps = np.array([40, 50, 60, 70, 80][:nt], float); k = np.exp(rc.lnk_form(temps, 80., 0.))
        for npts in (3, 4, 6):
            for sigma in (0.01, 0.02, 0.05):
                t = np.array(rc.TIMES[npts]); tc = t - t.mean(); sxx = (tc ** 2).sum()
                y = -k[:, None] * t[None, :] + rng.normal(0, sigma, (4000, nt, len(t)))
                b = (y * tc).sum(-1) / sxx; res = y - y.mean(-1, keepdims=True) - b[..., None] * tc
                se = np.sqrt((res ** 2).sum(-1) / (npts - 2) / sxx) / k[None, :]
                print(f"  n_T={nt} n_pts={npts} σ={sigma}: " + " ".join(f"{int(T)}°C {v:.3f}" for T, v in zip(temps, np.median(se, 0))))
    print("  → 域内残差 ≈ 1.3e-3(n_T=3)に対し SE(ln k̂) は 0.04–1.2(最低温度で最大 = 分解が最も少ない)")


if __name__ == "__main__":
    main()
