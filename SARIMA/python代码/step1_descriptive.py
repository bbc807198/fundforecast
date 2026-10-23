# -*- coding: utf-8 -*-
"""
Step 1: 数据探索与描述性统计
基金申购(purchase)与赎回(redeem)日度序列, 2013-07-01 ~ 2014-08-31
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import json, os

BASE = r"C:\Users\user\Desktop\timeseries"
OUT = os.path.join(BASE, "analysis")
os.makedirs(OUT, exist_ok=True)

# ---------- 中文字体检测 ----------
def pick_cn_font():
    candidates = ["Microsoft YaHei", "SimHei", "SimSun", "STXihei", "DengXian"]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for c in candidates:
        if c in installed:
            return c
    return None

CN_FONT = pick_cn_font()
if CN_FONT:
    plt.rcParams["font.sans-serif"] = [CN_FONT, "DejaVu Sans"]
else:
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
with open(os.path.join(OUT, "font_info.json"), "w", encoding="utf-8") as f:
    json.dump({"cn_font": CN_FONT}, f, ensure_ascii=False)

# ---------- 读取数据 ----------
df = pd.read_csv(os.path.join(BASE, "daily_series.csv"), encoding="utf-8-sig")
df["report_date"] = pd.to_datetime(df["report_date"].astype(str), format="%Y%m%d")
df = df.set_index("report_date").sort_index()
for c in ["purchase", "redeem"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

n = len(df)
date_min, date_max = df.index.min(), df.index.max()

# 日期连续性检查
full_idx = pd.date_range(date_min, date_max, freq="D")
missing_dates = full_idx.difference(df.index)

# 描述统计
def desc_stats(s: pd.Series) -> dict:
    q = s.quantile([0.25, 0.5, 0.75])
    return {
        "count": int(s.count()), "mean": float(s.mean()), "std": float(s.std()),
        "cv": float(s.std() / s.mean()), "min": float(s.min()), "p25": float(q.loc[0.25]),
        "median": float(q.loc[0.5]), "p75": float(q.loc[0.75]), "max": float(s.max()),
        "range": float(s.max() - s.min()), "skew": float(s.skew()), "kurt": float(s.kurt()),
    }

stats = {c: desc_stats(df[c]) for c in ["purchase", "redeem"]}

# 月度汇总
monthly = df.resample("ME").agg(["mean", "sum", "count"])
monthly.columns = ["_".join(c) for c in monthly.columns]

# 星期效应
wd = df.index.dayofweek
wd_names = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
weekday_stats = df.groupby(wd).agg(["mean", "std", "count"])
weekday_stats.index = wd_names
weekday_stats.columns = ["_".join(c) for c in weekday_stats.columns]

# 净流入
df["net"] = df["purchase"] - df["redeem"]

report = []
report.append("=" * 60)
report.append("一、数据概览")
report.append("=" * 60)
report.append(f"样本区间: {date_min.date()} ~ {date_max.date()}")
report.append(f"观测数: {n} 天; 缺失日期: {len(missing_dates)} 天")
report.append(f"缺失值: purchase={int(df['purchase'].isna().sum())}, redeem={int(df['redeem'].isna().sum())}")
report.append("")
report.append("二、描述性统计 (单位: 元)")
report.append("-" * 60)
stat_df = pd.DataFrame(stats).T
report.append(stat_df.to_string(float_format=lambda x: f"{x:,.2f}"))
report.append("")
report.append("三、月度汇总")
report.append("-" * 60)
report.append(monthly.to_string(float_format=lambda x: f"{x:,.0f}"))
report.append("")
report.append("四、按星期的统计 (weekday effect)")
report.append("-" * 60)
report.append(weekday_stats.to_string(float_format=lambda x: f"{x:,.0f}"))
report.append("")
report.append("五、净流入(purchase - redeem)描述")
report.append("-" * 60)
ns = desc_stats(df["net"])
report.append(f"mean={ns['mean']:,.0f}  std={ns['std']:,.0f}  min={ns['min']:,.0f}  max={ns['max']:,.0f}")
report.append(f"净流入>0 的天数: {int((df['net']>0).sum())} / {n}")

report_text = "\n".join(report)
with open(os.path.join(OUT, "step1_descriptive.txt"), "w", encoding="utf-8") as f:
    f.write(report_text)

# 保存描述统计表供报告使用
stat_df.to_csv(os.path.join(OUT, "desc_stats.csv"), encoding="utf-8-sig")
monthly.to_csv(os.path.join(OUT, "monthly.csv"), encoding="utf-8-sig")
weekday_stats.to_csv(os.path.join(OUT, "weekday.csv"), encoding="utf-8-sig")
df.to_csv(os.path.join(OUT, "clean.csv"), encoding="utf-8-sig")

# ---------- 图1: 时序图 ----------
fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
axes[0].plot(df.index, df["purchase"] / 1e8, color="#c0392b", lw=1.2)
axes[0].set_ylabel("申购 (亿元)")
axes[0].set_title("基金申购金额日度时序 (2013-07 ~ 2014-08)")
axes[0].grid(alpha=0.3)
axes[1].plot(df.index, df["redeem"] / 1e8, color="#27ae60", lw=1.2)
axes[1].set_ylabel("赎回 (亿元)")
axes[1].set_title("基金赎回金额日度时序 (2013-07 ~ 2014-08)")
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig1_timeseries.png"), dpi=150)
plt.close()

# ---------- 图2: 直方图 ----------
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(df["purchase"] / 1e8, bins=40, color="#c0392b", alpha=0.75, edgecolor="white")
axes[0].set_title("申购金额分布")
axes[0].set_xlabel("亿元"); axes[0].set_ylabel("频数")
axes[0].grid(alpha=0.3)
axes[1].hist(df["redeem"] / 1e8, bins=40, color="#27ae60", alpha=0.75, edgecolor="white")
axes[1].set_title("赎回金额分布")
axes[1].set_xlabel("亿元"); axes[1].set_ylabel("频数")
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig2_hist.png"), dpi=150)
plt.close()

# ---------- 图3: 月度均值趋势 ----------
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(monthly.index, monthly["purchase_mean"] / 1e8, marker="o", color="#c0392b", label="申购月均")
ax.plot(monthly.index, monthly["redeem_mean"] / 1e8, marker="s", color="#27ae60", label="赎回月均")
ax.set_title("月度日均申购/赎回趋势")
ax.set_ylabel("亿元")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig3_monthly.png"), dpi=150)
plt.close()

# ---------- 图4: 星期效应 ----------
fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(7)
ax.bar(x - 0.2, weekday_stats["purchase_mean"] / 1e8, width=0.4, color="#c0392b", label="申购")
ax.bar(x + 0.2, weekday_stats["redeem_mean"] / 1e8, width=0.4, color="#27ae60", label="赎回")
ax.set_xticks(x); ax.set_xticklabels(wd_names)
ax.set_ylabel("亿元")
ax.set_title("按星期几的平均申购/赎回")
ax.legend(); ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig4_weekday.png"), dpi=150)
plt.close()

print(report_text)
