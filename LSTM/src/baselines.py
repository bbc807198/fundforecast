"""
三条不依赖 LSTM 的基线。基线是"判断模型是否真的有用"的标尺：
如果 LSTM 连"上周同一天"或"星期几均值 × 水平"都打不过，那说明模型没有学到东西，
应该先回去检查特征/归一化，而不是继续调网络。

注意：三条基线都只用 origin_idx（含）之前的数据，不会偷看未来。
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .config import Config
from .data_prep import PreparedData, forecast_level_path


def baseline_seasonal_naive(
    values: np.ndarray, origin_idx: int, horizon: int
) -> np.ndarray:
    """
    季节朴素法：下一个周一 = 上一个周一的实际值，以此类推。
    这是时间序列里最强的小成本基线之一，很多"复杂模型"都打不过它。
    """
    v = np.asarray(values, dtype=np.float64)
    pred = np.empty(horizon, dtype=np.float64)
    for h in range(1, horizon + 1):
        j = origin_idx + h - 7
        if j < 0:
            pred[h - 1] = v[: origin_idx + 1].mean()
        else:
            pred[h - 1] = v[j]
    return pred


def baseline_mean_same_weekday(
    ds: PreparedData, origin_idx: int, dates: pd.DatetimeIndex, target_idx: int, weeks: int = 4
) -> np.ndarray:
    """
    最近 weeks 周"同一个星期几"的均值。比季节朴素法平滑，抗单日异常。
    """
    values = ds.raw[: origin_idx + 1, target_idx].astype(np.float64)
    past_dates = ds.dates[: origin_idx + 1]
    dow_past = past_dates.dayofweek.values
    future_dow = pd.DatetimeIndex(dates).dayofweek.values

    pred = np.empty(len(future_dow), dtype=np.float64)
    for i, dw in enumerate(future_dow):
        idx = np.where(dow_past == dw)[0][-weeks:]
        pred[i] = values[idx].mean() if len(idx) else values.mean()
    return pred


def baseline_weekday_profile_level(
    cfg: Config,
    ds: PreparedData,
    origin_idx: int,
    dates: pd.DatetimeIndex,
    target_idx: int,
    weeks: int = 8,
) -> np.ndarray:
    """
    "周内形状 × 水平"基线：和 LSTM 模型用的是同一套水平外推，
    只是周内形状直接用历史统计（每个星期几的 值/水平 之比）来估，不用网络。

    这条基线特别重要：它把"水平外推是否正确"和"LSTM 是否比简单统计更好地
    刻画周内形状"两件事分开了。LSTM 只有打赢它，才说明学习是有效的。
    """
    n_hist = min(len(ds), weeks * 7)
    lo = origin_idx + 1 - n_hist
    if lo < 0:
        lo = 0

    y = ds.raw[lo: origin_idx + 1, target_idx].astype(np.float64)
    lvl = ds.level[lo: origin_idx + 1, target_idx].astype(np.float64)
    dow = ds.dates[lo: origin_idx + 1].dayofweek.values

    # 用 值之和 / 水平之和 估每个星期几的因子（比"比值的均值"更稳）
    profile = np.ones(7, dtype=np.float64)
    global_ratio = float(y.sum() / max(lvl.sum(), 1e-9))
    for d in range(7):
        m = dow == d
        profile[d] = float(y[m].sum() / lvl[m].sum()) if m.sum() >= 2 and lvl[m].sum() > 0 else global_ratio

    lp = forecast_level_path(ds.raw[: origin_idx + 1, target_idx], len(dates), cfg)
    future_dow = pd.DatetimeIndex(dates).dayofweek.values
    return profile[future_dow] * lp.path


def all_baselines(
    cfg: Config, ds: PreparedData, origin_idx: int, dates: pd.DatetimeIndex, target: str
) -> Dict[str, np.ndarray]:
    """一次性算好三条基线，返回 {名字: 预测值数组}。"""
    ti = ds.target_names.index(target)
    values = ds.raw[:, ti].astype(np.float64)
    H = len(dates)
    return {
        "seasonal_naive": baseline_seasonal_naive(values, origin_idx, H),
        "mean_same_dow_4w": baseline_mean_same_weekday(ds, origin_idx, dates, ti, weeks=4),
        "weekday_profile": baseline_weekday_profile_level(cfg, ds, origin_idx, dates, ti, weeks=8),
    }
