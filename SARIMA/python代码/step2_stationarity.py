# -*- coding: utf-8 -*-
"""
Step 2: 平稳性检验 (ADF / KPSS) 与 ACF 分析, 确定差分阶数
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss, acf
import json, os

BASE = r"C:\Users\user\Desktop\timeseries"
OUT = os.path.join(BASE, "analysis")

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

df = pd.read_csv(os.path.join(OUT, "clean.csv"), encoding="utf-8-sig", index_col=0, parse_dates=True)

def adf_test(s, name):
    r = adfuller(s.dropna(), autolag="AIC", result_object=True)
    return {"series": name, "ADF统计量": r.statistic, "p值": r.pvalue, "滞后阶": r.lags,
            "1%临界": r.critical_values["1%"], "5%临界": r.critical_values["5%"], "10%临界": r.critical_values["10%"],
            "结论": "平稳" if r.pvalue < 0.05 else "不平稳"}

def kpss_test(s, name):
    try:
        r = kpss(s.dropna(), regression="c", nlags="auto")
        return {"series": name, "KPSS统计量": r[0], "p值": r[1], "结论": "不平稳" if r[1] < 0.05 else "平稳"}
    except Exception as e:
        return {"series": name, "KPSS统计量": np.nan, "p值": np.nan, "结论": f"计算失败:{e}"}
rows = []
for col, cn in [("purchase", "申购"), ("redeem", "赎回")]:
    x = df[col]
    lx = np.log(x)
    d1 = x.diff().dropna()
    dl1 = lx.diff().dropna()
    d7 = x.diff(7).dropna()
    dl7 = lx.diff(7).dropna()
    for s, nm in [(x, f"{cn}-原序列"), (lx, f"{cn}-对数"), (d1, f"{cn}-一阶差分"),
                  (dl1, f"{cn}-对数一阶差分"), (d7, f"{cn}-7步差分"), (dl7, f"{cn}-对数7步差分")]:
        rows.append(adf_test(s, nm))

adf_tbl = pd.DataFrame(rows)
adf_tbl.to_csv(os.path.join(OUT, "step2_adf.csv"), index=False, encoding="utf-8-sig")

krows = []
for col, cn in [("purchase", "申购"), ("redeem", "赎回")]:
    x = df[col]; lx = np.log(x)
    krows.append(kpss_test(x, f"{cn}-原序列"))
    krows.append(kpss_test(lx, f"{cn}-对数"))
    krows.append(kpss_test(lx.diff().dropna(), f"{cn}-对数一阶差分"))
kpss_tbl = pd.DataFrame(krows)
kpss_tbl.to_csv(os.path.join(OUT, "step2_kpss.csv"), index=False, encoding="utf-8-sig")

# ---------- ACF 图: 原序列 / 一阶差分 / 对数一阶差分 ----------
def plot_acf(ax, s, title, nlags=40):
    vals = acf(s.dropna(), nlags=nlags, fft=True)
    n = s.dropna().shape[0]
    conf = 1.96 / np.sqrt(n)
    ax.bar(range(len(vals)), vals, width=0.6, color="#2980b9")
    ax.axhline(conf, ls="--", color="gray", lw=0.8)
    ax.axhline(-conf, ls="--", color="gray", lw=0.8)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title(title, fontsize=10)
    ax.set_ylim(-1.05, 1.05)

fig, axes = plt.subplots(3, 2, figsize=(13, 9))
for j, (col, cn) in enumerate([("purchase", "申购"), ("redeem", "赎回")]):
    x = df[col]; lx = np.log(x)
    plot_acf(axes[0, j], x, f"{cn}原序列 ACF")
    plot_acf(axes[1, j], x.diff().dropna(), f"{cn}一阶差分 ACF")
    plot_acf(axes[2, j], np.log(x).diff().dropna(), f"{cn}对数一阶差分 ACF")
plt.suptitle("自相关函数(ACF)分析", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig5_acf.png"), dpi=150)
plt.close()

# ---------- 差分后序列图 ----------
fig, axes = plt.subplots(2, 2, figsize=(13, 7))
for i, (col, cn) in enumerate([("purchase", "申购"), ("redeem", "赎回")]):
    axes[i, 0].plot(df.index, df[col].diff(), color="#8e44ad", lw=1)
    axes[i, 0].set_title(f"{cn}一阶差分", fontsize=10); axes[i, 0].grid(alpha=0.3)
    axes[i, 1].plot(df.index, np.log(df[col]).diff(), color="#16a085", lw=1)
    axes[i, 1].set_title(f"{cn}对数一阶差分", fontsize=10); axes[i, 1].grid(alpha=0.3)
plt.suptitle("差分处理后的序列", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig6_diff.png"), dpi=150)
plt.close()

# 汇总关键统计: 差分序列的描述
summ = []
for col, cn in [("purchase", "申购"), ("redeem", "赎回")]:
    dl = np.log(df[col]).diff().dropna()
    summ.append(f"{cn} 对数一阶差分: mean={dl.mean():.5f}, std={dl.std():.5f}, "
                f"min={dl.min():.4f}, max={dl.max():.4f}")
with open(os.path.join(OUT, "step2_summary.txt"), "w", encoding="utf-8") as f:
    f.write("ADF检验结果\n" + adf_tbl.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    f.write("\n\nKPSS检验结果\n" + kpss_tbl.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    f.write("\n\n" + "\n".join(summ))

print(adf_tbl.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print()
print(kpss_tbl.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print()
print("\n".join(summ))
