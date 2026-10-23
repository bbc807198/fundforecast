"""
画图。全部用 Agg 后端（不弹窗），直接存 png，VS Code 里跑完打开看即可。
如果系统里找不到中文字体，标签会自动回退成英文，避免出现"豆腐块"。
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import Config
from .data_prep import PreparedData
from .utils import ensure_dir, setup_matplotlib


def _plt(ok: bool):
    import matplotlib.pyplot as plt

    def L(zh: str, en: str) -> str:
        return zh if ok else en

    return plt, L


def plot_forecast(
    ds: PreparedData,
    pred_df: pd.DataFrame,
    cfg: Config,
    path: str,
    title_note: str = "",
) -> None:
    """
    历史走势 + 9 月预测。

    阴影区间来自回测的残差分布（由调用方算好传进来），不是模型自己瞎猜的方差，
    所以它更接近真实的预测不确定性。
    """
    ok = setup_matplotlib(True)
    plt, L = _plt(ok)
    ensure_dir(os.path.dirname(path) or ".")

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    hist_start = ds.dates[0]
    zoom_start = ds.dates[-1] - pd.Timedelta(days=150)

    for ax, target in zip(axes, ds.target_names):
        y = ds.raw[:, ds.target_names.index(target)] / 1e6
        ax.plot(ds.dates, y, color="#8c8c8c", lw=0.9, label=L("历史实际", "actual"))
        ax.plot(ds.dates, ds.level[:, ds.target_names.index(target)] / 1e6,
                color="#1f77b4", lw=1.2, alpha=0.9, label=L("7日水平", "7d level"))

        if pred_df is not None and f"{target}_pred" in pred_df.columns:
            d = pd.to_datetime(pred_df["date"])
            p = pred_df[f"{target}_pred"] / 1e6
            ax.plot(d, p, color="#d62728", lw=2.0, marker="o", ms=3,
                    label=L("9月预测", "forecast"))
            if f"{target}_lo" in pred_df.columns:
                ax.fill_between(d, pred_df[f"{target}_lo"] / 1e6, pred_df[f"{target}_hi"] / 1e6,
                                color="#d62728", alpha=0.15, label=L("80% 经验区间", "80% band"))
            ax.axvline(d.iloc[0], color="#d62728", ls="--", lw=0.8, alpha=0.6)

        ax.set_ylabel(L("金额（百万）", "value (million)"))
        ax.set_title(f"{target}  {title_note}", fontsize=11)
        ax.grid(alpha=0.25)
        ax.legend(loc="upper left", fontsize=9)
        ax.axvspan(zoom_start, ds.dates[-1], color="#ffd400", alpha=0.06)

    axes[-1].set_xlabel(L("日期", "date"))
    tag = pd.Timestamp(cfg.forecast_start).strftime("%Y-%m")
    fig.suptitle(
        L(f"申购 / 赎回：历史与 {tag} 预测", f"Purchase / Redeem: history and {tag} forecast"),
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_recent_zoom(ds: PreparedData, pred_df: pd.DataFrame, cfg: Config, path: str, days: int = 90) -> None:
    """最近 N 天 + 预测的放大图，方便肉眼检查周内节奏对不对。"""
    ok = setup_matplotlib(True)
    plt, L = _plt(ok)
    ensure_dir(os.path.dirname(path) or ".")

    start = ds.dates[-1] - pd.Timedelta(days=days)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    for ax, target in zip(axes, ds.target_names):
        m = ds.dates >= start
        ti = ds.target_names.index(target)
        ax.plot(ds.dates[m], ds.raw[m, ti] / 1e6, "o-", color="#8c8c8c", ms=3, lw=0.9,
                label=L("实际", "actual"))
        if pred_df is not None:
            d = pd.to_datetime(pred_df["date"])
            ax.plot(d, pred_df[f"{target}_pred"] / 1e6, "o-", color="#d62728", ms=4, lw=1.6,
                    label=L("预测", "forecast"))
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)
        ax.set_title(target)
        ax.set_ylabel(L("百万", "million"))
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_training_curves(hists: Dict[str, List[float]], best_epochs: Optional[Dict[str, int]], path: str) -> None:
    """训练/验证损失曲线。验证损失见底后继续下降的训练损失，就是过拟合的信号。"""
    ok = setup_matplotlib(True)
    plt, L = _plt(ok)
    ensure_dir(os.path.dirname(path) or ".")

    n = max(1, len(hists))
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4), squeeze=False)
    for i, (name, hist) in enumerate(hists.items()):
        ax = axes[0][i]
        ax.plot(hist["train"], label=L("训练", "train"))
        ax.plot(hist["val"], label=L("验证", "val"))
        if best_epochs and name in best_epochs:
            ax.axvline(best_epochs[name], color="#888", ls="--", lw=0.8)
        ax.set_title(f"{name} loss")
        ax.set_xlabel(L("epoch", "epoch"))
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_backtest(per_day: pd.DataFrame, cfg: Config, path: str) -> None:
    """回测期内 实际 vs LSTM vs 最强基线，按目标分面板。"""
    ok = setup_matplotlib(True)
    plt, L = _plt(ok)
    ensure_dir(os.path.dirname(path) or ".")

    targets = list(per_day["target"].unique())
    fig, axes = plt.subplots(len(targets), 1, figsize=(13, 4 * len(targets)), squeeze=False)
    for i, target in enumerate(targets):
        ax = axes[i][0]
        d = per_day[per_day["target"] == target].sort_values("date")
        ax.plot(d["date"], d["actual"] / 1e6, color="#444", lw=1.4, label=L("实际", "actual"))
        ax.plot(d["date"], d["lstm"] / 1e6, color="#d62728", lw=1.4, alpha=0.9,
                label=L("LSTM 集成", "LSTM ensemble"))
        if "weekday_profile" in d.columns:
            ax.plot(d["date"], d["weekday_profile"] / 1e6, color="#2ca02c", lw=1.1, ls="--",
                    alpha=0.9, label=L("周内形状×水平 基线", "weekday-profile baseline"))
        for fold in d["fold"].unique():
            m = d["fold"] == fold
            ax.axvspan(d.loc[m, "date"].min(), d.loc[m, "date"].max(), alpha=0.05, color="#1f77b4")
        ax.set_title(f"{target} — {L('滚动回测（每段为一个月，蓝色底色=不同折）', 'walk-forward backtest (one month per fold)')}")
        ax.set_ylabel(L("百万", "million"))
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
