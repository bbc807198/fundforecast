# -*- coding: utf-8 -*-
"""
Step 4: 模型估计与残差诊断
申购: ARIMA(3,1,3) on log  (AIC/BIC最优, 无漂移)
赎回: ARIMA(3,1,3) on log  (AIC最优, 带漂移c)
对照: SARIMA(1,1,2)(1,1,1,7) 申购; SARIMA(1,1,1)(0,1,1,7) 赎回
诊断: 参数显著性, Ljung-Box(残差白噪声), Jarque-Bera(正态性), 残差ACF图, QQ图
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import jarque_bera
import warnings, os, json

warnings.filterwarnings("ignore")
BASE = r"C:\Users\user\Desktop\timeseries"
OUT = os.path.join(BASE, "analysis")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

df = pd.read_csv(os.path.join(OUT, "clean.csv"), encoding="utf-8-sig", index_col=0, parse_dates=True)

MODELS = {
    "purchase": {
        "cn": "申购",
        "arima": dict(order=(3, 1, 3), trend="n"),
        "sarima": dict(order=(1, 1, 2), seasonal_order=(1, 1, 1, 7), trend="n"),
    },
    "redeem": {
        "cn": "赎回",
        "arima": dict(order=(3, 1, 3), trend="c"),
        "sarima": dict(order=(1, 1, 1), seasonal_order=(0, 1, 1, 7), trend="n"),
    },
}

diag_rows = []
fits = {}

for col, cfg in MODELS.items():
    y = np.log(df[col])
    for kind in ["arima", "sarima"]:
        spec = cfg[kind]
        m = SARIMAX(y, enforce_stationarity=True, enforce_invertibility=True, **spec)
        r = m.fit(disp=False)
        fits[(col, kind)] = r

        # 残差诊断
        resid = r.resid[r.nobs_diffused if hasattr(r, "nobs_diffused") else len(r.resid):]
        resid = r.resid
        lb = acorr_ljungbox(resid, lags=[10, 20], return_df=True)
        jb = jarque_bera(resid)
        arch = het_arch(resid, nlags=10)

        diag_rows.append({
            "序列": cfg["cn"], "模型": kind.upper(),
            "阶数": f"({spec['order'][0]},{spec['order'][1]},{spec['order'][2]})" +
                    (f"({spec.get('seasonal_order',(0,0,0,0))[0]},{spec.get('seasonal_order',(0,0,0,0))[1]},{spec.get('seasonal_order',(0,0,0,0))[2]})₇" if kind == "sarima" else ""),
            "trend": spec["trend"],
            "AIC": r.aic, "BIC": r.bic,
            "LB(10)p": lb["lb_pvalue"].iloc[0], "LB(20)p": lb["lb_pvalue"].iloc[1],
            "JBp": jb[1], "ARCH(10)p": arch[1],
            "残差std": float(np.std(resid)),
        })

        # 保存 summary
        with open(os.path.join(OUT, f"step4_summary_{col}_{kind}.txt"), "w", encoding="utf-8") as f:
            f.write(str(r.summary()))

diag = pd.DataFrame(diag_rows)
diag.to_csv(os.path.join(OUT, "step4_diagnostics.csv"), index=False, encoding="utf-8-sig")
print(diag.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

# ---------- 残差诊断图 ----------
fig, axes = plt.subplots(3, 2, figsize=(13, 10))
row = 0
for col, cfg in MODELS.items():
    for kind in ["arima", "sarima"]:
        r = fits[(col, kind)]
        resid = r.resid
        ax = axes[row % 3, row // 3] if False else axes[row // 2, row % 2]
        ax.plot(resid, lw=0.8, color="#2c3e50")
        ax.axhline(0, color="red", lw=0.6, ls="--")
        ax.set_title(f"{cfg['cn']} {kind.upper()} 残差序列", fontsize=10)
        ax.grid(alpha=0.3)
        row += 1
plt.suptitle("模型残差时序", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig8_resid.png"), dpi=150)
plt.close()

# 残差ACF + QQ (以两个主模型为例)
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.gofplots import qqplot
import scipy.stats as st

fig, axes = plt.subplots(2, 2, figsize=(13, 7))
for i, (col, cfg) in enumerate(MODELS.items()):
    r = fits[(col, "arima")]
    resid = r.resid
    plot_acf(resid, lags=40, ax=axes[i, 0], title=f"{cfg['cn']} ARIMA 残差ACF")
    axes[i, 0].set_ylim(-0.15, 0.15)
    qqplot(resid, line="s", ax=axes[i, 1], fit=True)
    axes[i, 1].set_title(f"{cfg['cn']} ARIMA 残差QQ图", fontsize=10)
plt.suptitle("ARIMA残差诊断: ACF与正态性", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig9_resid_diag.png"), dpi=150)
plt.close()

# 保存拟合值对比图 (原始尺度)
fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
for i, (col, cfg) in enumerate(MODELS.items()):
    ax = axes[i]
    ax.plot(df.index, df[col] / 1e8, color="gray", lw=1, label="实际值", alpha=0.7)
    for kind, color in [("arima", "#c0392b"), ("sarima", "#2980b9")]:
        r = fits[(col, kind)]
        fitted = np.exp(r.fittedvalues)
        ax.plot(fitted.index, fitted / 1e8, color=color, lw=1, alpha=0.85,
                label=f"{kind.upper()}拟合值")
    ax.set_title(f"{cfg['cn']}: 实际值 vs 拟合值", fontsize=11)
    ax.set_ylabel("亿元"); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig10_fitted.png"), dpi=150)
plt.close()

# 保存模型参数表
params_rows = []
for (col, kind), r in fits.items():
    cn = MODELS[col]["cn"]
    for name, val in r.params.items():
        try:
            pv = r.pvalues[name]
        except Exception:
            pv = np.nan
        params_rows.append({"序列": cn, "模型": kind.upper(), "参数": name,
                            "估计值": val, "p值": pv})
pd.DataFrame(params_rows).to_csv(os.path.join(OUT, "step4_params.csv"), index=False, encoding="utf-8-sig")

print("\n参数表:")
print(pd.DataFrame(params_rows).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
