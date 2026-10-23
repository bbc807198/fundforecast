# -*- coding: utf-8 -*-
"""
Step 3: 模型识别与定阶
- PACF 图
- ARIMA(p,1,q) on log(series): p,q in 0..3, trend in {n,c}, 按 AIC/BIC 网格搜索
- SARIMA(p,1,q)(P,1,Q,7) 候选对比
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings, itertools, json, os

warnings.filterwarnings("ignore")
BASE = r"C:\Users\user\Desktop\timeseries"
OUT = os.path.join(BASE, "analysis")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

df = pd.read_csv(os.path.join(OUT, "clean.csv"), encoding="utf-8-sig", index_col=0, parse_dates=True)

# ---------- PACF 图 ----------
def plot_pacf(ax, s, title, nlags=40):
    vals = pacf(s.dropna(), nlags=nlags)
    n = s.dropna().shape[0]
    conf = 1.96 / np.sqrt(n)
    ax.bar(range(len(vals)), vals, width=0.6, color="#8e44ad")
    ax.axhline(conf, ls="--", color="gray", lw=0.8)
    ax.axhline(-conf, ls="--", color="gray", lw=0.8)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title(title, fontsize=10)
    ax.set_ylim(-1.05, 1.05)

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
plot_pacf(axes[0], np.log(df["purchase"]).diff().dropna(), "申购 对数一阶差分 PACF")
plot_pacf(axes[1], np.log(df["redeem"]).diff().dropna(), "赎回 对数一阶差分 PACF")
plt.suptitle("偏自相关函数(PACF)分析", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig7_pacf.png"), dpi=150)
plt.close()

# ---------- ARIMA 网格搜索 ----------
def grid_arima(log_series, p_range=range(0, 4), q_range=range(0, 4), d=1, trends=("n", "c")):
    res = []
    y = np.log(log_series)
    for p, q, tr in itertools.product(p_range, q_range, trends):
        if p == 0 and q == 0:
            continue
        try:
            m = SARIMAX(y, order=(p, d, q), trend=tr,
                        enforce_stationarity=True, enforce_invertibility=True)
            r = m.fit(disp=False)
            res.append({"p": p, "d": d, "q": q, "trend": tr, "nobs": int(r.nobs),
                        "AIC": r.aic, "BIC": r.bic, "HQIC": r.hqic})
        except Exception:
            pass
    return pd.DataFrame(res).sort_values("AIC").reset_index(drop=True)

grids = {}
for col, cn in [("purchase", "申购"), ("redeem", "赎回")]:
    g = grid_arima(df[col])
    g.to_csv(os.path.join(OUT, f"step3_grid_{col}.csv"), index=False, encoding="utf-8-sig")
    grids[col] = g
    print(f"===== {cn} ARIMA(p,1,q) on log  Top5 by AIC =====")
    print(g.head(5).to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"----- Top3 by BIC -----")
    print(g.sort_values("BIC").head(3).to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print()

# ---------- SARIMA 候选 (含7天季节周期) ----------
def grid_sarima(series, orders, seasonal_orders, d=1, D=1, s=7, trends=("n",)):
    res = []
    y = np.log(series)
    for (p, q), (P, Q), tr in itertools.product(orders, seasonal_orders, trends):
        if p == 0 and q == 0 and P == 0 and Q == 0:
            continue
        try:
            m = SARIMAX(y, order=(p, d, q), seasonal_order=(P, D, Q, s), trend=tr,
                        enforce_stationarity=True, enforce_invertibility=True)
            r = m.fit(disp=False)
            res.append({"p": p, "q": q, "P": P, "Q": Q, "trend": tr,
                        "AIC": r.aic, "BIC": r.bic})
        except Exception:
            pass
    return pd.DataFrame(res).sort_values("AIC").reset_index(drop=True)

sar_grids = {}
orders = [(0, 1), (1, 0), (1, 1), (2, 1), (0, 2), (2, 0), (1, 2), (2, 2)]
seasonal_orders = [(0, 1), (1, 0), (1, 1)]
for col, cn in [("purchase", "申购"), ("redeem", "赎回")]:
    g = grid_sarima(df[col], orders, seasonal_orders)
    g.to_csv(os.path.join(OUT, f"step3_grid_sarima_{col}.csv"), index=False, encoding="utf-8-sig")
    sar_grids[col] = g
    print(f"===== {cn} SARIMA(p,1,q)(P,1,Q,7) on log  Top5 by AIC =====")
    print(g.head(5).to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print()
