#!/usr/bin/env python
"""(a) 加速域内の「折れ」(二段 Arrhenius)を R² > 0.95 は拾えるか。拾えるとして、SN 比と分離できるか。

真の形: ln k(T) は T_b で折れる連続な折れ線。T ≤ T_b で Ea_lo、T > T_b で Ea_hi。
        25 °C の真値 t90 = 61.62 月に較正(k(25) 固定)。
        T_b = 50 °C(加速域内: 40/50/60 の上 1 点が別枝)、T_b = 35 °C(加速域の下 = 保存域との間。
        加速 3 点は全て Ea_hi 枝 → 加速データに痕跡なし、外挿は Ea_hi で行われ真の Ea_lo 枝と食い違う)。
比較対象: 「同じ見かけの Ea を持つ純 Arrhenius」(加速域の OLS 傾きを揃える)。
          これと R² 分布が重なれば、R² は折れと SN 比を区別していない。
出力: R²>0.95 通過率、lack-of-fit χ² 棄却率、25 °C 外挿誤差(t90 の倍率 X = 予測/真値)。

    ~/dev/papers/bayesian-shelf-life-arrhenius/.venv/bin/python ~/dev/generate_papers/docs/audit/r2_break.py
"""
from __future__ import annotations
import numpy as np
from scipy import stats

R = 8.314
K25 = -np.log(0.95) / 30.0
TIMES = np.array([0., 2., 4., 6.])
TSETS = {3: np.array([40., 50., 60.]), 4: np.array([40., 50., 60., 70.])}
NREP = 4000
THRESH = 0.95


def lnk_break(temp_c, ea_lo, ea_hi, tb_c):
    """T ≤ T_b: Ea_lo、T > T_b: Ea_hi。T_b で連続。k(25) = K25。"""
    T = np.asarray(temp_c, float) + 273.15; Tb = tb_c + 273.15
    lo = np.log(K25) + ea_lo * 1000 / R * (1 / 298.15 - 1 / T)
    at_b = np.log(K25) + ea_lo * 1000 / R * (1 / 298.15 - 1 / Tb)
    hi = at_b + ea_hi * 1000 / R * (1 / Tb - 1 / T)
    return np.where(T <= Tb, lo, hi)


def lnk_arr(temp_c, ea, lnk_at, t_at):
    """傾き ea で (t_at, lnk_at) を通る純 Arrhenius。"""
    T = np.asarray(temp_c, float) + 273.15
    return lnk_at + ea * 1000 / R * (1 / (t_at + 273.15) - 1 / T)


def simulate(temps, lnk_true, sigma, rng):
    k = np.exp(lnk_true)
    y = -k[None, :, None] * TIMES[None, None, :] + rng.normal(0, sigma, (NREP, len(temps), len(TIMES)))
    tc = TIMES - TIMES.mean(); sxx = (tc ** 2).sum()
    slope = (y * tc).sum(-1) / sxx
    resid = y - y.mean(-1, keepdims=True) - slope[..., None] * tc
    se = np.sqrt((resid ** 2).sum(-1) / (len(TIMES) - 2) / sxx)
    khat = -slope; ok = (khat > 0).all(-1); khat, se = khat[ok], se[ok]
    lnk = np.log(khat); x = 1 / (temps + 273.15); xc = x - x.mean()
    b = (lnk * xc).sum(-1) / (xc ** 2).sum(); a = lnk.mean(-1) - b * x.mean()
    fit = a[:, None] + b[:, None] * x
    r2 = 1 - ((lnk - fit) ** 2).sum(-1) / ((lnk - lnk.mean(-1, keepdims=True)) ** 2).sum(-1)
    v = (se / khat) ** 2; W = 1 / v
    xw = (W * x).sum(-1) / W.sum(-1); yw = (W * lnk).sum(-1) / W.sum(-1)
    bw = (W * (lnk - yw[:, None]) * (x - xw[:, None])).sum(-1) / (W * (x - xw[:, None]) ** 2).sum(-1)
    chi2 = (((lnk - (yw[:, None] + bw[:, None] * (x - xw[:, None]))) ** 2) / v).sum(-1)
    p_lof = stats.chi2.sf(chi2, len(temps) - 2)
    ea_app = -b * R / 1000
    lnk25_pred = a + b / 298.15
    X = np.exp(-(lnk25_pred - np.log(K25)))            # 予測 t90 / 真値 t90
    return r2, p_lof, ea_app, X


def main():
    rng = np.random.default_rng(20260912)
    print("加速域内・域外の折れに対する R²>0.95 と lack-of-fit の応答。真値 t90(25 °C)=61.62 月、4 点、NREP=4000")
    print("X = 外挿 t90 / 真値(>1 が楽観側)。'同Ea Arrhenius' = 加速域の OLS 傾きを揃えた純 Arrhenius(SN 比を揃えた対照)\n")
    for nt in (3, 4):
        temps = TSETS[nt]
        for sigma in (0.01, 0.02, 0.05):
            print("=" * 118)
            print(f"n_T={nt} ({temps.min():.0f}–{temps.max():.0f} °C), σ={sigma}")
            print(f"{'真の形':<34}{'見かけEa':>8}{'X 中央値':>9} | {'R²>0.95':>8}{'LOF棄却':>8} | {'同Ea Arrhenius: R²>0.95':>24}{'LOF棄却':>8}")
            cases = [("Arrhenius Ea=80", lnk_arr(temps, 80., np.log(K25), 25.))]
            for tb in (50., 35.):
                for lo, hi in ((80., 120.), (80., 160.), (80., 40.)):
                    cases.append((f"折れ T_b={tb:.0f}, {lo:.0f}→{hi:.0f}", lnk_break(temps, lo, hi, tb)))
            for name, lnk in cases:
                r2, p, ea, X = simulate(temps, lnk, sigma, rng)
                # 対照: 見かけの Ea(ノイズなしの OLS 傾き)を持つ純 Arrhenius、加速域の中央で一致させる
                x = 1 / (temps + 273.15); b0 = np.polyfit(x, lnk, 1); ea0 = -b0[0] * R / 1000
                Tm = temps.mean(); ref = lnk_arr(temps, ea0, np.polyval(b0, 1 / (Tm + 273.15)), Tm)
                r2r, pr, _, _ = simulate(temps, ref, sigma, rng)
                print(f"{name:<34}{ea0:>8.1f}{np.median(X):>9.2f} | {np.mean(r2 > THRESH):>8.1%}{np.mean(p < .05):>8.1%} | "
                      f"{np.mean(r2r > THRESH):>24.1%}{np.mean(pr < .05):>8.1%}")
    print("\n読み方: 折れの行と '同Ea Arrhenius' の列が近ければ、R² は折れではなく SN 比(見かけ Ea × 温度幅 ÷ ノイズ)に反応している。")
    print("       T_b=35 の行は加速域に痕跡がなく(R²・LOF とも Arrhenius と同じ)、X が大きくずれる = 害があって見えない型。")


if __name__ == "__main__":
    main()
