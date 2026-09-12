#!/usr/bin/env python
"""ideas/arrhenius-validity-check.md 「代替案」の中身: 実時間データ併用時の検出力。

問い: 加速外挿の t90 が真値の X 倍ずれているとき、25 °C の実時間データ m ヶ月で
      それを棄却できる確率はいくらか(α=0.05 両側)。

なぜ形(T^n)でなく X で問うか: paper-A 頑健性層の T^{±4} 逸脱は 25 °C 外挿で Δln k = ±0.012
(t90 で 1%)しか動かず無害。困る逸脱は加速域と保存域の間で機構が変わる型で、
加速域内に痕跡がない。ゆえに「形」は問わず「外挿誤差の大きさ X」で語る。

生成モデルは paper-A datagen と同一(ln(C/C0) = -k t + N(0,σ))。
加速側: n_T ∈ {3,4}、4 点、真の Arrhenius Ea=80、25 °C の真値 t90 = 61.62 月に較正。
  各温度 OLS → k̂、Arrhenius OLS → 25 °C へ外挿 → k_pred と SE(k_pred)(delta 法・重みなし OLS の予測分散)。
  ★ 外挿には X を掛ける: k_pred/X が「予測が X 倍長い有効期間を出している」状況。
実時間側: 25 °C、真の k25、ICH 長期スケジュール {0,3,6,9,12,18,24,36} を m で打ち切り(感度: 3 ヶ月毎)。
★ バッチ数 B ∈ {1,3} と反復アッセイ r ∈ {1,2} を加速側・実時間側の両方に入れる(2026-09-12 追加)。
  B バッチは同一の真の k をもつ独立系列(バッチ間変動なし = 検出に有利な仮定)、各時点の傾きは
  バッチごとの OLS 傾きの平均(SE は 1/√B)。反復アッセイは平均を取る(σ → σ/√r)。
  ICH の標準は B=3。r=1 が単回、r=2 が二重測定。
  切片つき OLS → k̂25 と SE(線形スケール。k̂≤0 でも定義される)。
検定: z = (k̂25 − k_pred/X) / sqrt(SE25² + SE_pred²/X²)。臨界値は t(df = 実時間点数 − 2) の 97.5% 点
      (実時間側の SE が少数点から推定されるため。z の 1.96 では m=12 で size が 15% に膨らむ)。
参考列 SE_pred=0: 加速側が完璧だった場合(= 実時間側だけの限界)。

    ~/dev/papers/bayesian-shelf-life-arrhenius/.venv/bin/python \
        ~/dev/generate_papers/docs/audit/realtime_power.py
"""
from __future__ import annotations
import numpy as np
from scipy.stats import t as tdist

R = 8.314
K25 = -np.log(0.95) / 30.0
EA = 80.0
TACC = {3: np.array([40., 50., 60.]), 4: np.array([40., 50., 60., 70.])}
TPTS = np.array([0., 2., 4., 6.])
ICH = np.array([0., 3., 6., 9., 12., 18., 24., 36.])
NREP = 20000
XS = (1.0, 1.25, 1.4, 2.0, 3.0)   # X=1.0 は size(第一種の過誤)の確認列
MS = (12, 18, 24, 36, 48)


def lnk_arr(temp_c):
    T = np.asarray(temp_c) + 273.15
    return np.log(K25) + EA * 1000 / R * (1 / 298.15 - 1 / T)


def ols_slope(t, y):
    """y: (..., len(t))。切片つき OLS の slope と SE(slope)。"""
    tc = t - t.mean(); sxx = (tc ** 2).sum()
    b = (y * tc).sum(-1) / sxx
    resid = y - y.mean(-1, keepdims=True) - b[..., None] * tc
    s2 = (resid ** 2).sum(-1) / (len(t) - 2)
    return b, np.sqrt(s2 / sxx)


def accelerated_pred(nt, sigma, rng, B=1, r=1):
    """加速データ → 25 °C の k_pred と SE(k_pred)。B バッチの傾きを平均、反復 r で σ/√r。"""
    temps = TACC[nt]; k = np.exp(lnk_arr(temps)); sg = sigma / np.sqrt(r)
    y = -k[None, None, :, None] * TPTS[None, None, None, :] + rng.normal(0, sg, (NREP, B, len(temps), len(TPTS)))
    b, se = ols_slope(TPTS, y)                   # (NREP, B, nT)
    b = b.mean(1); se = np.sqrt((se ** 2).sum(1)) / B
    khat = -b
    ok = (khat > 0).all(-1)                      # 加速側で k̂≤0 が出た反復は除外(実務では再測定)
    khat, se = khat[ok], se[ok]
    lnk = np.log(khat); v = (se / khat) ** 2     # delta 法 Var(ln k̂)
    x = 1 / (temps + 273.15); x0 = 1 / 298.15
    X = np.column_stack([np.ones(len(temps)), x])
    A = np.linalg.pinv(X)                        # 重みなし OLS(Skomski の Arrhenius プロットと同じ)
    c = np.array([1.0, x0]) @ A                  # ln k(25) の予測 = c·lnk
    lnpred = lnk @ c
    var_pred = (v * c ** 2).sum(-1)              # 独立な ln k̂ の線形結合
    return np.exp(lnpred), np.exp(lnpred) * np.sqrt(var_pred), ok.sum()


def realtime(m, sigma, quarterly, rng, n, B=1, r=1):
    t = np.arange(0, m + 1, 3.) if quarterly else ICH[ICH <= m]
    y = -K25 * t[None, None, :] + rng.normal(0, sigma / np.sqrt(r), (n, B, len(t)))
    b, se = ols_slope(t, y)
    return -b.mean(1), np.sqrt((se ** 2).sum(1)) / B, len(t)


def batch_table():
    rng = np.random.default_rng(20260913)
    print("\n" + "=" * 100)
    print("★ バッチ数 B × 反復 r の感度(ICH 長期スケジュール、加速 n_T=3・4 点、両側に同じ B, r)")
    print("   '答え合わせ' 列 = 実時間データ単独で申請値(真値の X 倍)を棄却できる確率(加速側の SE を含まない)")
    print("=" * 100)
    for sigma in (0.01, 0.02, 0.05):
        for B, r in ((1, 1), (3, 1), (3, 2), (3, 3)):
            kp, sep, n = accelerated_pred(3, sigma, rng, B, r)
            print(f"\n--- σ={sigma}, B={B}, r={r}   加速側 SE(ln k_pred) 中央値 = {np.median(sep / kp):.2f} ---")
            print(f"{'m':>4}{'SE(ln k̂25)':>12} | 加速との整合検定: " + "".join(f"{'X='+str(X):>7}" for X in XS)
                  + " | 答え合わせ: " + "".join(f"{'X='+str(X):>7}" for X in XS))
            for m in (24, 36, 48):
                k25, se25, npts = realtime(m, sigma, False, rng, n, B, r)
                crit = tdist.ppf(0.975, B * npts - 2)
                row = [np.mean(np.abs((k25 - kp / X) / np.sqrt(se25 ** 2 + (sep / X) ** 2)) > crit) for X in XS]
                ref = [np.mean(np.abs((k25 - K25 / X) / se25) > crit) for X in XS]
                print(f"{m:>4}{np.median(se25) / K25:>12.2f} |                  " + "".join(f"{p:>7.0%}" for p in row)
                      + " |            " + "".join(f"{p:>7.0%}" for p in ref))


def batch_variation_table():
    """reviewer-risks #10: バッチ間変動 τ(ln k のバッチ効果 SD)を入れた '答え合わせ' の検出力(2026-09-12 追加)。

    3 つの一次バッチが実時間に使われる ICH の標準を模す。バッチ b の k は k·exp(τ u_b)、u_b~N(0,1)
    (前指数因子のバッチ効果、Ea は共通)。申請値は製品(バッチ平均 k)の t90 の X 倍。検定は 3 通り:
      (a) within : バッチを反復として扱う(§バッチ数 の計算 = 従来の 61%)。SE は各バッチの回帰残差、df = B·npts − 2
      (b) between: バッチを単位として扱う。SE = sd(バッチ傾き)/√B、df = B − 1(τ を含む。プール不可のときの形)
      (c) Q1E gate: 傾きの同質性を F 検定(α=0.25、切片はバッチ別)。プール可なら共通傾きモデルの残差 SE
                    (df = B·npts − B − 1)、不可なら (b)。実務者が実際に踏む手順に最も近い
    τ=0 で (a) は従来値を再現。τ>0 で (a) の size(X=1 列)が膨らめば従来値は「バッチ交換可能」の上界。
    """
    from scipy.stats import f as fdist
    rng = np.random.default_rng(20260914)
    B, r, m, sigma = 3, 1, 36, 0.02
    t = ICH[ICH <= m]; npts = len(t); tc = t - t.mean(); sxx = (tc ** 2).sum()
    print("\n" + "=" * 100)
    print(f"★ バッチ間変動 τ を入れた '答え合わせ'(σ={sigma}, B={B}, r={r}, ICH 3 年 {npts} 点、NREP={NREP})")
    print("   (a) within = バッチを反復扱い(従来)  (b) between = バッチを単位扱い(SE=sd(傾き)/√B, df=B−1)")
    print("   (c) Q1E gate = 傾き同質性 F 検定 α=0.25 でプール可なら共通傾き、不可なら (b)。'プール率' はその割合")
    print("   X=1.0 列は size。τ は ln k のバッチ効果 SD(τ=0.1 ≈ 速度定数の CV 10%)")
    print("=" * 100)
    hdr = "".join(f"{'X='+str(X):>6}" for X in XS)
    print(f"{'τ':>5} | (a) within{hdr} | (b) between{hdr} | (c) Q1E gate{hdr} | プール率")
    for tau in (0.0, 0.05, 0.1, 0.2, 0.3):
        u = rng.normal(0, tau, (NREP, B, 1))
        kb = K25 * np.exp(u)                                     # (NREP, B, 1)
        y = -kb * t[None, None, :] + rng.normal(0, sigma / np.sqrt(r), (NREP, B, npts))
        b, se = ols_slope(t, y)                                  # (NREP, B)
        khat = -b
        kbar = khat.mean(1)
        se_w = np.sqrt((se ** 2).sum(1)) / B
        se_b = khat.std(1, ddof=1) / np.sqrt(B)
        crit_w = tdist.ppf(0.975, B * npts - 2); crit_b = tdist.ppf(0.975, B - 1)
        # (c) 切片バッチ別・傾き共通モデル。同じ t なら共通傾き = 傾きの平均
        rss_sep = (se ** 2 * sxx * (npts - 2)).sum(1)            # 各バッチ別回帰の RSS 和
        yc = y - y.mean(-1, keepdims=True)
        rss_com = ((yc + kbar[:, None, None] * tc) ** 2).sum((1, 2))
        df1, df2 = B - 1, B * (npts - 2)
        F = ((rss_com - rss_sep) / df1) / (rss_sep / df2)
        pool = fdist.sf(F, df1, df2) > 0.25
        se_c = np.sqrt(rss_com / (B * npts - B - 1) / (B * sxx))
        crit_c = tdist.ppf(0.975, B * npts - B - 1)
        kprod = K25 * np.exp(tau ** 2 / 2)                       # 製品平均 k = E[k·exp(τu)]
        pw = [np.mean(np.abs((kbar - kprod / X) / se_w) > crit_w) for X in XS]
        pb = [np.mean(np.abs((kbar - kprod / X) / se_b) > crit_b) for X in XS]
        pc = [np.mean(np.where(pool, np.abs((kbar - kprod / X) / se_c) > crit_c,
                               np.abs((kbar - kprod / X) / se_b) > crit_b)) for X in XS]
        print(f"{tau:>5.2f} |           " + "".join(f"{p:>6.0%}" for p in pw)
              + " |            " + "".join(f"{p:>6.0%}" for p in pb)
              + " |             " + "".join(f"{p:>6.0%}" for p in pc) + f" | {pool.mean():>6.0%}")
        if tau == 0.0:   # 参考: 片側検定(申請値より速い、の向きだけ棄却)。本文は両側。限界節で一言触れる用
            c1w, c1b, c1c = tdist.ppf(0.95, B * npts - 2), tdist.ppf(0.95, B - 1), tdist.ppf(0.95, B * npts - B - 1)
            ow = [np.mean((kbar - kprod / X) / se_w > c1w) for X in XS]
            ob = [np.mean((kbar - kprod / X) / se_b > c1b) for X in XS]
            oc = [np.mean(np.where(pool, (kbar - kprod / X) / se_c > c1c, (kbar - kprod / X) / se_b > c1b)) for X in XS]
            print(f"{'片側':>5} |           " + "".join(f"{p:>6.0%}" for p in ow)
                  + " |            " + "".join(f"{p:>6.0%}" for p in ob)
                  + " |             " + "".join(f"{p:>6.0%}" for p in oc) + " | (τ=0, α=0.05 片側)")
    print("  → τ=0 で (a) は §バッチ数 の 61% を再現。(b) は df=2 の代償で X=2 が 27%、(c) Q1E 手順は 45%(プール率 75%)。")
    print("     τ≤0.1 では 3 列とも変わらない。τ=0.3 で (a) の size は 13% に膨らみ(名目 5% でない)、(c) は 33%、(b) は 19%。")
    print("     結論: 2 倍の誤りの検出力は「バッチをどう扱うか」で 27–61%、実務手順(Q1E)で 45%。1.4 倍は 13–25%。")


def main():
    rng = np.random.default_rng(20260912)
    print("加速外挿の t90 が真値の X 倍ずれているとき、25 °C 実時間 m ヶ月で棄却できる確率(α=0.05 両側)")
    print(f"NREP={NREP}、真値 t90(25 °C)=61.62 月、Ea=80。'完璧' 列は加速側 SE=0 の参考値。\n")
    for quarterly, lab in ((False, "ICH 長期スケジュール {0,3,6,9,12,18,24,36}"), (True, "3 ヶ月毎(感度)")):
        print("=" * 100 + f"\n実時間サンプリング: {lab}\n" + "=" * 100)
        for sigma in (0.01, 0.02, 0.05):
            for nt in (3, 4):
                kp, sep, n = accelerated_pred(nt, sigma, rng)
                print(f"\n--- σ={sigma}, 加速 n_T={nt}(40–{TACC[nt].max():.0f} °C, 4 点)  "
                      f"外挿 SE(ln k_pred) 中央値 = {np.median(sep / kp):.2f} ---")
                print(f"{'m':>4}{'点':>3}{'SE(ln k̂25)':>12} | " + "".join(f"{'X='+str(X):>8}" for X in XS)
                      + " |" + "".join(f"{'完璧'+str(X):>8}" for X in XS))
                for m in MS:
                    k25, se25, npts = realtime(m, sigma, quarterly, rng, n)
                    crit = tdist.ppf(0.975, npts - 2)
                    row = []
                    for X in XS:
                        z = (k25 - kp / X) / np.sqrt(se25 ** 2 + (sep / X) ** 2)
                        row.append(np.mean(np.abs(z) > crit))
                    ref = []
                    for X in XS:
                        z = (k25 - K25 / X) / se25
                        ref.append(np.mean(np.abs(z) > crit))
                    print(f"{m:>4}{npts:>3}{np.median(se25) / K25:>12.2f} | "
                          + "".join(f"{p:>8.0%}" for p in row) + " |" + "".join(f"{p:>8.0%}" for p in ref))
    batch_table()
    batch_variation_table()
    print("\n参考: 加速データ単独の R²>0.95 が保証する幅は SE(Ea)/Ea<23%(n_T=3)。OLS 予測分散で 25 °C に伝播すると t90 は 1 SE で 1.8 倍、2 SE で 3.3 倍(r2_width.py。40 °C 既知の傾きのみなら 1.4 / 2.0 倍)。")


if __name__ == "__main__":
    main()
