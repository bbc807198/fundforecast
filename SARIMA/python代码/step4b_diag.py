# -*- coding: utf-8 -*-
"""
Step 4b: 深入诊断
1) SARIMAX 差分初始化会使前若干残差失真, 剔除前14个后再做 Ljung-Box
2) 扩展 SARIMA 网格 (p,q<=3, P,Q<=2), 对Top候选重算诊断
"""
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings, itertools, os

warnings.filterwarnings("ignore")
BASE = r"C:\Users\user\Desktop\timeseries"
OUT = os.path.join(BASE, "analysis")

df = pd.read_csv(os.path.join(OUT, "clean.csv"), encoding="utf-8-sig", index_col=0, parse_dates=True)

def fit_and_diag(y, order, seasonal_order, trend):
    m = SARIMAX(y, order=order, seasonal_order=seasonal_order, trend=trend,
                enforce_stationarity=True, enforce_invertibility=True)
    r = m.fit(disp=False)
    resid = r.resid.iloc[14:]  # 剔除差分初始化影响的残差
    lb10 = acorr_ljungbox(resid, lags=[10], return_df=True)["lb_pvalue"].iloc[0]
    lb20 = acorr_ljungbox(resid, lags=[20], return_df=True)["lb_pvalue"].iloc[0]
    lb7 = acorr_ljungbox(resid, lags=[7], return_df=True)["lb_pvalue"].iloc[0]
    return r, {"AIC": r.aic, "BIC": r.bic, "LB7p": lb7, "LB10p": lb10, "LB20p": lb20,
               "resid_std": float(resid.std())}

# ---------- 1) 对现有模型剔除初始化残差后重算 ----------
print("=== 剔除前14个残差后的诊断 ===")
for col, cn, spec in [
    ("purchase", "申购", dict(order=(1,1,2), seasonal_order=(1,1,1,7), trend="n")),
    ("redeem", "赎回", dict(order=(1,1,1), seasonal_order=(0,1,1,7), trend="n")),
    ("purchase", "申购-ARIMA", dict(order=(3,1,3), trend="n")),
    ("redeem", "赎回-ARIMA", dict(order=(3,1,3), trend="c")),
]:
    y = np.log(df[col])
    r, d = fit_and_diag(y, spec["order"], spec.get("seasonal_order", (0,0,0,0)), spec["trend"])
    print(f"{cn}: AIC={d['AIC']:.2f}  LB7p={d['LB7p']:.4f}  LB10p={d['LB10p']:.4f}  LB20p={d['LB20p']:.4f}")

# ---------- 2) 扩展SARIMA网格 ----------
print("\n=== 扩展SARIMA网格 (剔除初始化残差后按AIC排序) ===")
orders = [(0,0),(0,1),(1,0),(1,1),(2,0),(0,2),(2,1),(1,2),(2,2),(3,0),(0,3),(3,1),(1,3),(3,2),(2,3),(3,3)]
seas = [(0,0),(1,0),(0,1),(1,1),(2,0),(0,2),(2,1),(1,2),(2,2)]

results = {}
for col, cn in [("purchase", "申购"), ("redeem", "赎回")]:
    y = np.log(df[col])
    rows = []
    for (p, q), (P, Q) in itertools.product(orders, seas):
        if p == 0 and q == 0 and P == 0 and Q == 0:
            continue
        try:
            r, d = fit_and_diag(y, (p,1,q), (P,1,Q,7), "n")
            rows.append({"p": p, "q": q, "P": P, "Q": Q, **d})
        except Exception:
            pass
    g = pd.DataFrame(rows).sort_values("AIC").reset_index(drop=True)
    g.to_csv(os.path.join(OUT, f"step4b_grid_sarima_{col}.csv"), index=False, encoding="utf-8-sig")
    results[col] = g
    print(f"\n----- {cn} Top10 by AIC -----")
    print(g.head(10).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
