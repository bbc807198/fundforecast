# -*- coding: utf-8 -*-
"""
Step 5b: 修复诊断接口 + 生成回测对比图 + 统一残差诊断表
"""
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import jarque_bera
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings, os

warnings.filterwarnings("ignore")
BASE = r"C:\Users\user\Desktop\timeseries"
OUT = os.path.join(BASE, "analysis")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

df = pd.read_csv(os.path.join(OUT, "clean.csv"), encoding="utf-8-sig", index_col=0, parse_dates=True)

CANDIDATES = {
    "purchase": [
        ("ARIMA(3,1,3)", dict(order=(3,1,3), trend="n", seasonal_order=(0,0,0,0))),
        ("SARIMA(1,1,2)(1,1,1)7", dict(order=(1,1,2), trend="n", seasonal_order=(1,1,1,7))),
        ("SARIMA(1,1,2)(2,1,1)7", dict(order=(1,1,2), trend="n", seasonal_order=(2,1,1,7))),
    ],
    "redeem": [
        ("ARIMA(3,1,3)", dict(order=(3,1,3), trend="c", seasonal_order=(0,0,0,0))),
        ("SARIMA(1,1,1)(0,1,1)7", dict(order=(1,1,1), trend="n", seasonal_order=(0,1,1,7))),
        ("SARIMA(3,1,3)(0,1,1)7", dict(order=(3,1,3), trend="n", seasonal_order=(0,1,1,7))),
    ],
}

def fit_model(col, spec, end=None):
    y = np.log(df.loc[:end, col]) if end else np.log(df[col])
    m = SARIMAX(y, order=spec["order"], seasonal_order=spec["seasonal_order"],
                trend=spec["trend"], enforce_stationarity=True, enforce_invertibility=True)
    return m.fit(disp=False)

# ---------- 统一诊断表 (全样本) ----------
rows = []
fits_full = {}
for col, cands in CANDIDATES.items():
    cn = "申购" if col == "purchase" else "赎回"
    for name, spec in cands:
        r = fit_model(col, spec)
        fits_full[(col, name)] = r
        resid = r.resid.iloc[14:]  # 剔除差分初始化影响
        lb7 = acorr_ljungbox(resid, lags=[7], return_df=True)["lb_pvalue"].iloc[0]
        lb20 = acorr_ljungbox(resid, lags=[20], return_df=True)["lb_pvalue"].iloc[0]
        jb = jarque_bera(resid)
        arch = het_arch(resid, nlags=10)
        rows.append({"序列": cn, "模型": name, "AIC": r.aic, "BIC": r.bic,
                     "残差std": float(resid.std()), "LB(7)p": lb7, "LB(20)p": lb20,
                     "JBp": jb[1], "ARCH(10)p": arch[1]})
diag = pd.DataFrame(rows)
diag.to_csv(os.path.join(OUT, "step5b_diag_full.csv"), index=False, encoding="utf-8-sig")
print("=== 全样本诊断表 ===")
print(diag.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

# ---------- 回测对比图 ----------
def make_bt_chart(bt_name, year, month, n_days, train_end, fname):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    for j, (col, cn) in enumerate([("purchase", "申购"), ("redeem", "赎回")]):
        ax = axes[j]
        bg_start = train_end - pd.Timedelta(days=75)
        bg = df.loc[bg_start:train_end, col]
        ax.plot(bg.index, bg.values / 1e8, color="gray", lw=1, alpha=0.6, label="历史实际")
        colors = ["#c0392b", "#2980b9", "#27ae60"]
        fc_idx = pd.date_range(f"{year}-{month:02d}-01", periods=n_days, freq="D")
        for k, (name, spec) in enumerate(CANDIDATES[col]):
            r = fit_model(col, spec, end=train_end)
            pred = np.exp(r.get_forecast(steps=n_days).predicted_mean.values)
            ax.plot(fc_idx, pred / 1e8, color=colors[k], lw=1.3, marker="o", ms=3, label=name)
        actual = df.loc[fc_idx, col].values
        ax.plot(fc_idx, actual / 1e8, color="black", lw=2.2, label="实际值")
        ax.set_title(f"{bt_name} - {cn}", fontsize=11)
        ax.set_ylabel("亿元"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, fname), dpi=150)
    plt.close()

make_bt_chart("回测1: 预测2013年9月", 2013, 9, 30, pd.Timestamp("2013-08-31"), "fig11_bt2013.png")
make_bt_chart("回测2: 预测2014年8月", 2014, 8, 31, pd.Timestamp("2014-07-31"), "fig12_bt2014.png")
print("\n回测对比图已保存: fig11_bt2013.png, fig12_bt2014.png")
