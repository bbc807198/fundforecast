# -*- coding: utf-8 -*-
"""
Step 6: 最终模型全样本拟合 + 2014年9月预测
推荐模型(经诊断与回测选择):
  申购: SARIMA(1,1,2)(2,1,1)_7  on log
  赎回: SARIMA(1,1,1)(0,1,1)_7  on log
对照(用户要求的非季节性ARIMA):
  申购/赎回: ARIMA(3,1,3) on log
"""
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
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

FINAL = {
    "purchase": {
        "cn": "申购",
        "rec": dict(name="SARIMA(1,1,2)(2,1,1)₇", order=(1,1,2), seasonal_order=(2,1,1,7), trend="n"),
        "base": dict(name="ARIMA(3,1,3)", order=(3,1,3), seasonal_order=(0,0,0,0), trend="n"),
    },
    "redeem": {
        "cn": "赎回",
        "rec": dict(name="SARIMA(1,1,1)(0,1,1)₇", order=(1,1,1), seasonal_order=(0,1,1,7), trend="n"),
        "base": dict(name="ARIMA(3,1,3)", order=(3,1,3), seasonal_order=(0,0,0,0), trend="c"),
    },
}

fc_idx = pd.date_range("2014-09-01", "2014-09-30", freq="D")
H = len(fc_idx)

results = {}
for col, cfg in FINAL.items():
    y = np.log(df[col])
    for key in ["rec", "base"]:
        spec = cfg[key]
        m = SARIMAX(y, order=spec["order"], seasonal_order=spec["seasonal_order"],
                    trend=spec["trend"], enforce_stationarity=True, enforce_invertibility=True)
        r = m.fit(disp=False)
        fc = r.get_forecast(steps=H)
        mean = np.exp(fc.predicted_mean.values)
        ci = fc.conf_int(alpha=0.05)
        lo = np.exp(ci.iloc[:, 0].values)
        hi = np.exp(ci.iloc[:, 1].values)
        results[(col, key)] = {"spec": spec, "res": r, "mean": mean, "lo": lo, "hi": hi}

# ---------- 汇总预测表 ----------
tbl = pd.DataFrame(index=fc_idx)
tbl.index.name = "日期"
for col, cfg in FINAL.items():
    cn = cfg["cn"]
    r_rec = results[(col, "rec")]
    r_base = results[(col, "base")]
    tbl[f"{cn}_预测"] = np.round(r_rec["mean"]).astype("int64")
    tbl[f"{cn}_预测下限95"] = np.round(r_rec["lo"]).astype("int64")
    tbl[f"{cn}_预测上限95"] = np.round(r_rec["hi"]).astype("int64")
    tbl[f"{cn}_ARIMA预测"] = np.round(r_base["mean"]).astype("int64")
    tbl[f"{cn}_净流入"] = tbl[f"{cn}_预测"] - tbl[f"{cn}_预测"]
tbl["净流入_预测"] = tbl["申购_预测"] - tbl["赎回_预测"]
tbl.to_csv(os.path.join(OUT, "step6_forecast_sep2014.csv"), encoding="utf-8-sig")

print("=== 2014年9月预测 (推荐模型) ===")
print(tbl.to_string(float_format=lambda v: f"{v:,.0f}"))
print()
for col, cfg in FINAL.items():
    cn = cfg["cn"]
    tot_rec = results[(col, "rec")]["mean"].sum()
    tot_base = results[(col, "base")]["mean"].sum()
    aug = df.loc["2014-08", col].sum()
    print(f"{cn}: 9月预测总量={tot_rec/1e8:.2f}亿元 (ARIMA对照 {tot_base/1e8:.2f}亿元), "
          f"8月实际={aug/1e8:.2f}亿元, 环比={(tot_rec-aug)/aug*100:+.1f}%")

# ---------- 预测图 ----------
fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
for i, (col, cfg) in enumerate(FINAL.items()):
    ax = axes[i]
    cn = cfg["cn"]
    hist = df.loc["2014-06-01":, col]
    ax.plot(hist.index, hist.values / 1e8, color="gray", lw=1, alpha=0.75, label="历史实际(6-8月)")
    rec = results[(col, "rec")]
    ax.plot(fc_idx, rec["mean"] / 1e8, color="#c0392b" if col == "purchase" else "#27ae60",
            lw=1.8, marker="o", ms=4, label=f"{cfg['rec']['name']} 预测")
    ax.fill_between(fc_idx, rec["lo"] / 1e8, rec["hi"] / 1e8,
                    color="#c0392b" if col == "purchase" else "#27ae60", alpha=0.15, label="95%预测区间")
    base = results[(col, "base")]
    ax.plot(fc_idx, base["mean"] / 1e8, color="#7f8c8d", lw=1.3, ls="--",
            marker="s", ms=3, label=f"{cfg['base']['name']} 预测(对照)")
    ax.axvline(pd.Timestamp("2014-08-31"), color="black", lw=0.8, ls=":")
    ax.set_title(f"{cn}: 2014年9月预测", fontsize=11)
    ax.set_ylabel("亿元"); ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig13_forecast.png"), dpi=150)
plt.close()

# ---------- 预测汇总 ----------
summ_rows = []
for col, cfg in FINAL.items():
    cn = cfg["cn"]
    rec = results[(col, "rec")]
    base = results[(col, "base")]
    aug = df.loc["2014-08", col].sum()
    summ_rows.append({
        "序列": cn, "模型": cfg["rec"]["name"],
        "9月预测总量(亿元)": rec["mean"].sum() / 1e8,
        "ARIMA对照(亿元)": base["mean"].sum() / 1e8,
        "8月实际(亿元)": aug / 1e8,
        "环比%": (rec["mean"].sum() - aug) / aug * 100,
        "预测区间下限(亿元)": rec["lo"].sum() / 1e8,
        "预测区间上限(亿元)": rec["hi"].sum() / 1e8,
    })
summ = pd.DataFrame(summ_rows)
summ.to_csv(os.path.join(OUT, "step6_forecast_summary.csv"), index=False, encoding="utf-8-sig")
print("\n=== 月度汇总 ===")
print(summ.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

# 保存模型参数
param_rows = []
for (col, key), v in results.items():
    cn = FINAL[col]["cn"]
    for name, val in v["res"].params.items():
        param_rows.append({"序列": cn, "模型": v["spec"]["name"], "参数": name, "估计值": val})
pd.DataFrame(param_rows).to_csv(os.path.join(OUT, "step6_final_params.csv"), index=False, encoding="utf-8-sig")
print("\n最终模型参数:")
print(pd.DataFrame(param_rows).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print("\n预测图已保存: fig13_forecast.png")
