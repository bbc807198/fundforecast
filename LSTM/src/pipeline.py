"""
把"数据 -> 训练 -> 预测"串起来的一层，供 forecast.py（9 月预测）和
backtest.py（滚动回测）共用。

这样做的好处：回测和最终预测走的是同一条代码路径，回测分数才有参考价值。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

from .config import Config, future_head_active, net_out_len, window_horizon
from .data_prep import (
    FeatureBuilder,
    PreparedData,
    build_windows,
    calendar_rows_for,
    forecast_level_path,
    split_train_val,
    valid_anchors,
)
from .model import count_parameters
from .train import History, fit_target_scaler, predict_ensemble, train_one
from .utils import log


# ---------------------------------------------------------------------------
# 单个目标的训练产物
# ---------------------------------------------------------------------------
@dataclass
class TrainedTarget:
    target: str
    target_idx: int
    builder: FeatureBuilder
    models: List[torch.nn.Module]
    y_mu: float
    y_sd: float
    hist: History
    train_end_idx: int
    n_train: int
    n_val: int
    horizon: int


@dataclass
class Prediction:
    dates: "np.ndarray"
    ratio: Dict[str, np.ndarray] = field(default_factory=dict)
    value: Dict[str, np.ndarray] = field(default_factory=dict)
    level: Dict[str, np.ndarray] = field(default_factory=dict)
    slope: Dict[str, float] = field(default_factory=dict)
    drift: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------
def train_target(
    cfg: Config,
    ds: PreparedData,
    target: str,
    train_end_idx: int,
    device: str = "cpu",
    verbose: bool = False,
) -> TrainedTarget:
    """
    训练一个目标（申购 or 赎回）的 LSTM 集成。

    只用 train_end_idx（含）之前的数据：
      - 特征窗口不能跨过 train_end_idx
      - 目标也必须全部落在 train_end_idx 之前，所以锚点上限是 train_end_idx - horizon
      - 标准化参数只用训练段统计
    """
    tidx = ds.target_names.index(target)
    horizon = window_horizon(cfg)
    use_future = future_head_active(cfg)
    head = "future" if use_future else "multistep"

    anchors = valid_anchors(len(ds), cfg.lookback, horizon)
    anchors = anchors[anchors <= train_end_idx - horizon]
    if len(anchors) < 30:
        raise ValueError(
            f"{target}: 训练锚点只有 {len(anchors)} 个，历史太短，请把回测起点往后移或减小 lookback/out_len"
        )

    tr, va = split_train_val(anchors, cfg)

    builder = FeatureBuilder(cfg, ds, target)
    # 标准化只用"训练样本覆盖到的天"，不碰验证段
    scaler_hi = int(tr[-1]) + cfg.lookback
    builder.fit(0, min(max(scaler_hi, 10), len(ds)))

    X = builder.build(np.arange(len(ds)))
    # 目标：默认用"相对窗口均值"的 ratio，消除长期漂移；raw 则直接用 ratio
    target_series = ds.rel_ratio if cfg.target_norm == "window_mean" else ds.ratio
    win = build_windows(X, target_series, tr, tidx, cfg.lookback, horizon,
                        head=head, cal=ds.cal)
    val = build_windows(X, target_series, va, tidx, cfg.lookback, horizon,
                        head=head, cal=ds.cal)
    if use_future:
        Xtr, Ytr, Ftr = win
        Xva, Yva, Fva = val
    else:
        (Xtr, Ytr), (Xva, Yva) = win, val
        Ftr = Fva = None

    y_mu, y_sd = fit_target_scaler(Ytr)
    Ytr_n = (Ytr - y_mu) / y_sd
    Yva_n = (Yva - y_mu) / y_sd

    models: List[torch.nn.Module] = []
    first_hist: Optional[History] = None
    for k in range(cfg.ensemble):
        seed = cfg.seed + 1000 * k
        model, hist = train_one(
            cfg, Xtr, Ytr_n, Xva, Yva_n, seed=seed, device=device,
            verbose=verbose and k == 0, Ftr=Ftr, Fva=Fva,
        )
        models.append(model)
        if k == 0:
            first_hist = hist

    assert first_hist is not None
    if verbose:
        log(
            f"  {target}: 样本 {len(tr)}/{len(va)}（训练/验证），"
            f"参数 {count_parameters(models[0])}，最佳 epoch {first_hist.best_epoch}，"
            f"val_loss {first_hist.best_val:.4f}"
        )
    return TrainedTarget(
        target=target,
        target_idx=tidx,
        builder=builder,
        models=models,
        y_mu=y_mu,
        y_sd=y_sd,
        hist=first_hist,
        train_end_idx=int(train_end_idx),
        n_train=len(tr),
        n_val=len(va),
        horizon=horizon,
    )


# ---------------------------------------------------------------------------
# 直接多步预测
# ---------------------------------------------------------------------------
def predict_direct(
    cfg: Config,
    ds: PreparedData,
    trained: Dict[str, TrainedTarget],
    origin_idx: int,
    dates: "np.ndarray",
    device: str = "cpu",
) -> Prediction:
    """
    direct 模式：模型一次输出 out_len 天的 ratio，截取前 H 天。

    优点：不存在误差逐步累积的问题（30 步递归会把每一步的误差放大 30 次）。
    代价：输出维度固定，月初/月末的差异只能靠日历特征自己去学。
    """
    H = len(dates)
    rows = np.arange(origin_idx - cfg.lookback + 1, origin_idx + 1)
    pred = Prediction(dates=np.asarray(dates))

    # future 头需要"未来每天的日历"，这是已知信息，可以从日期直接算出来
    fut_all = build_future_features(cfg, ds, dates) if future_head_active(cfg) else None

    for target, tm in trained.items():
        if tm.horizon != window_horizon(cfg):
            raise RuntimeError("训练与预测的 horizon 不一致，请检查 mode / out_len 配置")

        X = build_input_batch(cfg, tm, rows, H)
        y_hat = np.asarray(predict_ensemble(tm.models, X, tm.y_mu, tm.y_sd, device, fut=fut_all))
        y_hat = y_hat[0]                     # batch=1：(1, H) 或 (1, out_len) -> 一维
        if len(y_hat) < H:
            raise ValueError(f"模型输出长度 {len(y_hat)} 小于预测天数 {H}，请把 out_len 调大")

        # 反标准化 -> 乘回"窗口内 ratio 均值" -> 得到真实的 ratio
        scale = float(ds.scale[origin_idx, tm.target_idx]) if cfg.target_norm == "window_mean" else 1.0
        ratio = y_hat[:H] * scale
        lp = forecast_level_path(ds.raw[: origin_idx + 1, tm.target_idx], H, cfg)

        pred.ratio[target] = ratio
        pred.level[target] = lp.path
        pred.value[target] = ratio * lp.path
        pred.slope[target] = lp.slope
        pred.drift[target] = lp.total_drift
    return pred


def build_input_batch(cfg: Config, tm: "TrainedTarget", rows: np.ndarray, H: int) -> np.ndarray:
    """构造预测用的输入：就是锚点前 lookback 天的特征窗口，shape (1, L, F)。"""
    window = tm.builder.build(rows)
    return window[None, :, :]


def build_future_features(cfg: Config, ds: PreparedData, dates) -> np.ndarray:
    """
    构造"已知的未来协变量"：未来每天的日历特征（周几 one-hot / 节假日 / 月内位置）
    + 第几天/总天数。shape (1, H, C+1)，batch 维广播给所有样本。

    注意分母用 window_horizon(cfg)（训练时的固定长度 31），不是 H，
    否则 30 天的月份和 31 天的月份，"第 30 天"会对应不同的特征值。
    """
    cal = calendar_rows_for(dates, cfg, ds.cal_names)
    hs = (np.arange(1, len(dates) + 1) / window_horizon(cfg)).astype(np.float32)[:, None]
    fut = np.concatenate([cal.astype(np.float32), hs], axis=1)
    return fut[None, :, :]


# ---------------------------------------------------------------------------
# 递归单步预测
# ---------------------------------------------------------------------------
def predict_recursive(
    cfg: Config,
    ds: PreparedData,
    trained: Dict[str, TrainedTarget],
    origin_idx: int,
    dates: "np.ndarray",
    device: str = "cpu",
) -> Prediction:
    """
    recursive 模式：每次只预测下一天，再把预测值塞回输入窗口，滚动 H 次。

    注意两个模型是"互为输入"的（申购模型输入里含赎回 ratio），
    所以必须同步滚动：每一步先把两个目标的当天预测都算出来，再一起回填。

    优点：预测长度不限，可解释性好（能看到逐日演化）。
    缺点：误差会累积，30 天后容易"跑偏"，通常会比 direct 差。
          只有当你需要预测长度远大于训练时的 out_len 时才建议用。
    """
    H = len(dates)
    if net_out_len(cfg) != 1:
        raise ValueError("recursive 模式要求单步输出（out_len=1）")

    cal_future = calendar_rows_for(dates, cfg, ds.cal_names)
    pred = Prediction(dates=np.asarray(dates))

    # 每个目标的水平外推路径（只依赖历史，与 LSTM 无关）
    level_paths: Dict[str, np.ndarray] = {}
    for target, tm in trained.items():
        lp = forecast_level_path(ds.raw[: origin_idx + 1, tm.target_idx], H, cfg)
        level_paths[target] = lp.path
        pred.level[target] = lp.path
        pred.slope[target] = lp.slope
        pred.drift[target] = lp.total_drift

    # 每个目标的输入窗口缓存（已标准化的特征行）
    rows = np.arange(origin_idx - cfg.lookback + 1, origin_idx + 1)
    bufs: Dict[str, List[np.ndarray]] = {
        t: [r for r in tm.builder.build(rows)] for t, tm in trained.items()
    }

    ratio_pred: Dict[str, np.ndarray] = {t: np.zeros(H) for t in trained}
    value_pred: Dict[str, np.ndarray] = {t: np.zeros(H) for t in trained}
    # 反标准化用的 scale（锚点当天的"窗口内 ratio 均值"），整个 horizon 固定
    scales: Dict[str, float] = {
        t: (float(ds.scale[origin_idx, tm.target_idx]) if cfg.target_norm == "window_mean" else 1.0)
        for t, tm in trained.items()
    }

    for h in range(H):
        # 1) 用当前窗口预测"第 h 天"的 ratio（相对量）
        for target, tm in trained.items():
            X = np.stack(bufs[target][-cfg.lookback:], axis=0)[None, :, :]
            rel = float(predict_ensemble(tm.models, X, tm.y_mu, tm.y_sd, device)[0, 0])
            ratio_pred[target][h] = rel * scales[target]

        # 2) ratio * 水平 = 原始值
        for target in trained:
            value_pred[target][h] = ratio_pred[target][h] * level_paths[target][h]

        # 3) 把这一天回填进每个目标的输入窗口
        for target, tm in trained.items():
            used = tm.builder.used
            r_vec = np.array([[ratio_pred[u][h] for u in used]], dtype=np.float32)
            l_vec = np.array([[level_paths[u][h] for u in used]], dtype=np.float32)
            row = tm.builder.build_from_values(r_vec, l_vec, cal_future[h: h + 1])
            bufs[target].append(row[0])

    for target in trained:
        pred.ratio[target] = ratio_pred[target]
        pred.value[target] = value_pred[target]
    return pred


# ---------------------------------------------------------------------------
# 统一入口
# ---------------------------------------------------------------------------
def predict_horizon(
    cfg: Config,
    ds: PreparedData,
    trained: Dict[str, TrainedTarget],
    origin_idx: int,
    dates,
    device: str = "cpu",
) -> Prediction:
    """按配置选择 direct / recursive 预测。"""
    if cfg.mode == "direct":
        return predict_direct(cfg, ds, trained, origin_idx, dates, device)
    if cfg.mode == "recursive":
        return predict_recursive(cfg, ds, trained, origin_idx, dates, device)
    raise ValueError(f"未知 mode: {cfg.mode!r}（应为 direct 或 recursive）")
