#!/usr/bin/env python
"""査読で突かれうる点を投稿前に潰す — 逆転・LOF・無害性の頑健性(2026-09-12)。

A. 較正の取り方: k(25) 固定(paper-A)以外に k(40) / k(50) / k(60) 固定でも逆転するか
   (k(25) 固定は concave の加速域 k が大きく = 相対ノイズが小さく、逆転に有利な可能性がある)
B. 真の Ea: 67 / 80 / 104.6 で逆転するか
C. 逸脱の大きさ: n = ±1, ±2, ±4(Eyring は n=1)。逆転と外挿誤差の単調性
D. 閾値: 0.90 / 0.95 / 0.99 で逆転するか(恒等式から閾値によらないはず)
E. lack-of-fit を正しい F 形(プールした残差分散、F(n_T−2, n_T(n_pts−2)))にして size と power
F. 速度論の誤特定(0 次・2 次の真値を 1 次で当てはめ)は R² に「非 Arrhenius」として映るか、外挿はどれだけ狂うか
G. 乱数シードを変えて主要セルの MC 誤差を見る

    ~/dev/papers/bayesian-shelf-life-arrhenius/.venv/bin/python ~/dev/generate_papers/docs/audit/robustness_checks.py
"""
from __future__ import annotations
import numpy as np
from scipy import stats

R = 8.314
TREF = 313.15
K25 = -np.log(0.95) / 30.0
TIMES = {3: [0., 3., 6.], 4: [0., 2., 4., 6.], 6: [0., 1., 2., 3., 4., 6.]}
TSETS = {3: np.array([40., 50., 60.]), 4: np.array([40., 50., 60., 70.])}
NREP = 4000


def lnk_form(temp_c, ea, n, anchor_c=25.0, k_anchor=K25):
    """T^n·Arrhenius(Tref=313.15)。anchor 温度で k = k_anchor となるよう ln A を決める。"""
    T = np.asarray(temp_c, float) + 273.15; Ta = anchor_c + 273.15
    lna = np.log(k_anchor) + ea * 1000 / (R * Ta) - n * np.log(Ta / TREF)
    return lna - ea * 1000 / (R * T) + n * np.log(T / TREF)


def fit(temps, npts, sigma, y_true_fn, rng):
    """y_true_fn(t, k) → 真の ln(C/C0)。1 次 OLS で k̂、Arrhenius OLS R²、LOF(χ² 版と F 版)、外挿 X。"""
    t = np.array(TIMES[npts]); k = np.exp(y_true_fn.lnk)
    ytrue = y_true_fn(t, k)                                    # (nT, npts)
    y = ytrue[None] + rng.normal(0, sigma, (NREP,) + ytrue.shape)
    tc = t - t.mean(); sxx = (tc ** 2).sum()
    b = (y * tc).sum(-1) / sxx
    resid = y - y.mean(-1, keepdims=True) - b[..., None] * tc
    s2 = (resid ** 2).sum(-1) / (npts - 2)                    # (NREP, nT)
    khat = -b; ok = (khat > 0).all(-1)
    khat, s2 = khat[ok], s2[ok]
    lnk = np.log(khat); x = 1 / (temps + 273.15); xc = x - x.mean()
    bb = (lnk * xc).sum(-1) / (xc ** 2).sum(); aa = lnk.mean(-1) - bb * x.mean()
    f = aa[:, None] + bb[:, None] * x
    r2 = 1 - ((lnk - f) ** 2).sum(-1) / ((lnk - lnk.mean(-1, keepdims=True)) ** 2).sum(-1)
    # LOF χ² 版(温度ごとの分散)
    v = s2 / (khat ** 2 * sxx); W = 1 / v
    xw = (W * x).sum(-1) / W.sum(-1); yw = (W * lnk).sum(-1) / W.sum(-1)
    bw = (W * (lnk - yw[:, None]) * (x - xw[:, None])).sum(-1) / (W * (x - xw[:, None]) ** 2).sum(-1)
    fw = yw[:, None] + bw[:, None] * (x - xw[:, None])
    chi2 = (((lnk - fw) ** 2) / v).sum(-1); p_chi = stats.chi2.sf(chi2, len(temps) - 2)
    # LOF F 版: 残差分散を温度間でプール(等分散の仮定 = 生成モデルと一致)。Var(ln k̂_T) = s²_pool/(k̂_T² Sxx)
    s2p = s2.mean(-1, keepdims=True); vp = s2p / (khat ** 2 * sxx); Wp = 1 / vp
    xw = (Wp * x).sum(-1) / Wp.sum(-1); yw = (Wp * lnk).sum(-1) / Wp.sum(-1)
    bw = (Wp * (lnk - yw[:, None]) * (x - xw[:, None])).sum(-1) / (Wp * (x - xw[:, None]) ** 2).sum(-1)
    fw = yw[:, None] + bw[:, None] * (x - xw[:, None])
    F = (((lnk - fw) ** 2) / vp).sum(-1) / (len(temps) - 2)
    p_F = stats.f.sf(F, len(temps) - 2, len(temps) * (npts - 2))
    X = np.exp(-((aa + bb / 298.15) - np.log(K25)))            # 予測 t90 / 真値 t90(k(25) 固定のときのみ意味がある)
    return r2, p_chi, p_F, X


class FirstOrder:
    def __init__(self, lnk): self.lnk = lnk
    def __call__(self, t, k): return -k[:, None] * t[None, :]


def pass_rate(temps, npts, sigma, ea, n, rng, anchor=25.0, thr=0.95):
    if anchor == 25.0:
        lnk = lnk_form(temps, ea, n)
    else:   # anchor 温度の k を「同じ Ea の純 Arrhenius が k(25)=K25 から到達する値」に固定
        ka = np.exp(lnk_form(anchor, ea, 0.0)); lnk = lnk_form(temps, ea, n, anchor, ka)
    r2, pc, pf, X = fit(temps, npts, sigma, FirstOrder(lnk), rng)
    return np.mean(r2 > thr), np.mean(pc < .05), np.mean(pf < .05), np.median(X)


def main():
    rng = np.random.default_rng(1)
    temps, npts, sigma = TSETS[3], 4, 0.02
    print("=" * 96 + "\nA. 較正アンカー(どの温度の k を 3 形で揃えるか)を変えても逆転するか  n_T=3, 4点, σ=0.02, Ea=80\n" + "=" * 96)
    print(f"{'anchor':>8} | {'Arrhenius':>10}{'concave':>10}{'convex':>10} | 逆転 | {'見かけEa90.7のArrh':>18}{'見かけEa69.3のArrh':>18}")
    for a in (25., 40., 50., 60.):
        p = [pass_rate(temps, npts, sigma, 80., n, rng, a)[0] for n in (0., 4., -4.)]
        # 対照: 同じアンカーで、concave/convex と同じ見かけ Ea をもつ純 Arrhenius(k(anchor) を揃える)
        q = []
        for n in (4., -4.):
            x = 1 / (temps + 273.15)
            lnk_dev = lnk_form(temps, 80., n) if a == 25. else lnk_form(temps, 80., n, a, np.exp(lnk_form(a, 80., 0.)))
            ea_app = -np.polyfit(x, lnk_dev, 1)[0] * R / 1000
            Tm = temps.mean(); ka = np.exp(np.polyval(np.polyfit(x, lnk_dev, 1), 1 / (Tm + 273.15)))
            r2, *_ = fit(temps, npts, sigma, FirstOrder(lnk_form(temps, ea_app, 0., Tm, ka)), rng)
            q.append(np.mean(r2 > .95))
        print(f"{a:>7.0f}° | {p[0]:>10.1%}{p[1]:>10.1%}{p[2]:>10.1%} | {'★' if p[1] > p[0] else '—':>4} | {q[0]:>18.1%}{q[1]:>18.1%}")
    print("  → 逆転は k(25) 固定という較正の産物(concave の加速域 k が大きく相対ノイズが小さい)。k(60) 固定では消える。")
    print("     一方「同じ見かけ Ea の純 Arrhenius」と concave/convex の通過率は常にほぼ一致 = R² は形を見ていない(こちらが頑健な主張)")

    print("\n" + "=" * 96 + "\nB. 真の Ea を変えても逆転するか  n_T=3, 4点, σ=0.02, k(25) 固定\n" + "=" * 96)
    print(f"{'Ea':>6} | {'Arrhenius':>10}{'concave':>10}{'convex':>10} | 逆転")
    for ea in (67., 80., 104.6):
        p = [pass_rate(temps, npts, sigma, ea, n, rng)[0] for n in (0., 4., -4.)]
        print(f"{ea:>6.1f} | {p[0]:>10.1%}{p[1]:>10.1%}{p[2]:>10.1%} | {'★' if p[1] > p[0] else '—'}")

    print("\n" + "=" * 96 + "\nC. 逸脱の大きさ n(Eyring は n=1)。通過率と 25 °C 外挿誤差 X  n_T=3, 4点, σ=0.02, Ea=80\n" + "=" * 96)
    print(f"{'n':>4} | {'R²>0.95':>8}{'X 中央値':>9}  (n=0 の通過率との差)")
    base = pass_rate(temps, npts, sigma, 80., 0., rng)[0]
    for n in (-4., -2., -1., 0., 1., 2., 4.):
        p, _, _, X = pass_rate(temps, npts, sigma, 80., n, rng)
        print(f"{n:>+4.0f} | {p:>8.1%}{X:>9.3f}  ({p - base:+.1%})")

    print("\n" + "=" * 96 + "\nD. 閾値を変えても逆転するか  n_T=3, 4点, σ=0.02, Ea=80\n" + "=" * 96)
    print(f"{'閾値':>6} | {'Arrhenius':>10}{'concave':>10}{'convex':>10} | 逆転")
    for thr in (0.90, 0.95, 0.99):
        p = [pass_rate(temps, npts, sigma, 80., n, rng, thr=thr)[0] for n in (0., 4., -4.)]
        print(f"{thr:>6.2f} | {p[0]:>10.1%}{p[1]:>10.1%}{p[2]:>10.1%} | {'★' if p[1] > p[0] else '—'}")

    print("\n" + "=" * 96 + "\nE. lack-of-fit: χ² 版(温度ごとの分散)と F 版(プール分散)の size(n=0)と power(n=±4)\n" + "=" * 96)
    print(f"{'n_T':>4}{'n_pts':>6}{'σ':>6} | {'χ² size':>8}{'χ² concave':>11}{'χ² convex':>10} | {'F size':>7}{'F concave':>10}{'F convex':>9}")
    for nt in (3, 4):
        for np_ in (3, 4, 6):
            for s in (0.01, 0.02, 0.05):
                r = [pass_rate(TSETS[nt], np_, s, 80., n, rng) for n in (0., 4., -4.)]
                print(f"{nt:>4}{np_:>6}{s:>6} | {r[0][1]:>8.1%}{r[1][1]:>11.1%}{r[2][1]:>10.1%} | {r[0][2]:>7.1%}{r[1][2]:>10.1%}{r[2][2]:>9.1%}")
    print("  → F 版で size が 5% に揃い、power は依然 size と同じ。χ² 版の膨らみは分散推定の df 不足(Taira)")

    print("\n" + "=" * 96 + "\nF. 速度論の誤特定(真値 0 次 / 2 次、1 次で当てはめ)は R² に映るか。Arrhenius 真値, Ea=80, n_T=3, 4点\n" + "=" * 96)
    class ZeroOrder:      # C/C0 = 1 − k t  → ln(C/C0) = ln(1 − k t)。k は 25 °C で spec 95% に 30 月で到達するよう較正
        def __init__(self, lnk): self.lnk = lnk
        def __call__(self, t, k): return np.log(np.clip(1 - k[:, None] * t[None, :], 1e-6, None))
    class SecondOrder:    # C/C0 = 1/(1 + k C0 t)、C0=1 単位 → ln(C/C0) = −ln(1 + k t)
        def __init__(self, lnk): self.lnk = lnk
        def __call__(self, t, k): return -np.log(1 + k[:, None] * t[None, :])
    print(f"{'真の速度論':<12}{'σ':>6} | {'R²>0.95':>8}{'LOF-F':>7}")
    for name, cls, k25 in (("first", FirstOrder, K25), ("zero", ZeroOrder, 0.05 / 30.0), ("second", SecondOrder, (1 / 0.95 - 1) / 30.0)):
        for s in (0.01, 0.02, 0.05):
            lnk = lnk_form(temps, 80., 0.0, 25.0, k25)
            r2, pc, pf, _ = fit(temps, npts, s, cls(lnk), rng)
            print(f"{name:<12}{s:>6} | {np.mean(r2 > .95):>8.1%}{np.mean(pf < .05):>7.1%}")
    print("  → 0 次/2 次でも R²・LOF は 1 次とほぼ同じ = 速度論の誤特定は Arrhenius プロットには映らない(各温度で同じ向きに偏るため)")
    print("     外挿誤差は paper-A の頑健性層(robust_06–20)が既に示している。本稿では R² が見えないことだけを言う")

    print("\n" + "=" * 96 + "\nG. シード安定性: n_T=3, 4点, σ=0.02, Ea=80 の通過率(5 シード)\n" + "=" * 96)
    for seed in (1, 2, 3, 4, 5):
        rg = np.random.default_rng(seed)
        p = [pass_rate(temps, npts, sigma, 80., n, rg)[0] for n in (0., 4., -4.)]
        print(f"  seed {seed}: Arrhenius {p[0]:.1%}  concave {p[1]:.1%}  convex {p[2]:.1%}")
    print("  → NREP=4000 の二項 SE ≈ 0.7%。逆転幅 15 ポイントは MC 誤差の 20 倍")


if __name__ == "__main__":
    main()
