# -*- coding: utf-8 -*-
"""
Step 5: 模型回测评估
回测1: 训练集 <= 2013-08-31, 预测 2013-09-01 ~ 2013-09-30 (30天)
回测2: 训练集 <= 2014-07-31, 预测 2014-08-01 ~ 2014-08-31 (31天)
指标: RMSE, MAE, MAPE, 月度总量误差
候选:
  申购: ARIMA(3,1,3), SARIMA(1,1,2)(1,1,1)7, SARIMA(1,1,2)(2,1,1)7
  赎回: ARIMA(3,1,3)-c, SARIMA(1,1,1)(0,1,1)7, SARIMA(3,1,3)(0,1,1)7
"""
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings, os

warnings.filterwarnings("ignore")
BASE = r"C:\Users\user\Desktop\timeseries"
OUT = os.path.join(BASE, "analysis")

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

def metrics(actual, pred):
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    err = actual - pred
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    mape = float(np.mean(np.abs(err) / actual) * 100)
    tot_err = float((actual.sum() - pred.sum()) / actual.sum() * 100)
    return {"RMSE": rmse, "MAE": mae, "MAPE%": mape, "月度总量误差%": tot_err}

def backtest(col, spec, train_end, fc_start, fc_end):
    train = np.log(df.loc[:train_end, col])
    test_idx = pd.date_range(fc_start, fc_end, freq="D")
    actual = df.loc[test_idx, col]
    m = SARIMAX(train, order=spec["order"], seasonal_order=spec["seasonal_order"],
                trend=spec["trend"], enforce_stationarity=True, enforce_invertibility=True)
    r = m.fit(disp=False)
    fc = r.get_forecast(steps=len(test_idx))
    pred = np.exp(fc.predicted_mean.values)
    return actual.values, pred, r

rows = []
preds_store = {}
for bt_name, train_end, fc_start, fc_end in [
    ("回测1: 预测2013年9月", "2013-08-31", "2013-09-01", "2013-09-30"),
    ("回测2: 预测2014年8月", "2014-07-31", "2014-08-01", "2014-08-31"),
]:
    for col, cands in CANDIDATES.items():
        cn = "申购" if col == "purchase" else "赎回"
        for name, spec in cands:
            actual, pred, r = backtest(col, spec, train_end, fc_start, fc_end)
            mt = metrics(actual, pred)
            rows.append({"回测": bt_name, "序列": cn, "模型": name, **mt})
            preds_store[(bt_name, cn, name)] = (actual, pred)

bt = pd.DataFrame(rows)
bt.to_csv(os.path.join(OUT, "step5_backtest.csv"), index=False, encoding="utf-8-sig")
print(bt.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))

# ---------- 最终全样本模型的规范诊断 (statsmodels内置检验) ----------
print("\n=== 全样本拟合的残差序列相关检验 (statsmodels test_serial_correlation) ===")
diag_rows = []
FINAL = {
    "purchase": CANDIDATES["purchase"],
    "redeem": CANDIDATES["redeem"],
}
for col, cands in FINAL.items():
    cn = "申购" if col == "purchase" else "赎回"
    y = np.log(df[col])
    for name, spec in cands:
        m = SARIMAX(y, order=spec["order"], seasonal_order=spec["seasonal_order"],
                    trend=spec["trend"], enforce_stationarity=True, enforce_invertibility=True)
        r = m.fit(disp=False)
        lb = r.test_serial_correlation(lags=20, method="ljungbox")
        lb7 = r.test_serial_correlation(lags=7, method="ljungbox")
        diag_rows.append({"序列": cn, "模型": name, "AIC": r.aic, "BIC": r.bic,
                          "LB7p": lb7.pvalue, "LB20p": lb.pvalue})
        print(f"{cn} {name}: AIC={r.aic:.2f} BIC={r.bic:.2f} LB7p={lb7.pvalue:.4f} LB20p={lb.pvalue:.4f}")

pd.DataFrame(diag_rows).to_csv(os.path.join(OUT, "step5_final_diag.csv"), index=False, encoding="utf-8-sig")

# ---------- 回测对比图 ----------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

for bt_name in ["回测1: 预测2013年9月", "回测2: 预测2014年8月"]:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    for j, (col, cn) in enumerate([("purchase", "申购"), ("redeem", "赎回")]):
        ax = axes[j]
        # 实际值(含回测期前60天背景)
        bg_start = pd.Timestamp("2013-07-01" if "2013" in bt_name else "2014-06-01")
        bg = df.loc[bg_start:pd.Timestamp(bt_name[-11:]) if False else None, col]
        ax.plot(bg.index, bg.values / 1e8, color="gray", lw=1, alpha=0.6, label="历史实际")
        colors = ["#c0392b", "#2980b9", "#27ae60", "#8e44ad"]
        for k, name in enumerate([c[0] for c in CANDIDATES[col]]):
            actual, pred = preds_store[(bt_name, cn, name)]
            idx = pd.date_range("2013-09-01" if "2013" in bt_name else "2014-08-01",
                                periods=len(actual), freq="D")
            ax.plot(idx, pred / 1e8, color=colors[k], lw=1.4, marker="o", ms=3, label=name)
        # 实际值高亮
        actual0, _ = preds_store[(bt_name, cn, CANDIDATES[col][0][0])]
        idx0 = pd.date_range("2013-09-01" if "2013" in bt_name else "2014-08-01",
                             periods=len(actual0), freq="D")
        ax.plot(idx0, actual0 / 1e8, color="black", lw=2.2, label="实际值")
        ax.set_title(f"{bt_name} - {cn}", fontsize=11)
        ax.set_ylabel("亿元"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    fname = "fig11_bt2013.png" if "2013" in bt_name else "fig12_bt2014.png"
    plt.savefig(os.path.join(OUT, fname), dpi=150)
    plt.close()

print("\n回测对比图已保存")
