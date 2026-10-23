"""
数据准备：读取 -> 清洗 -> 构造"水平 + 季节指数" -> 日历特征 -> 训练窗口。

这里最关键的三个概念，先讲清楚再读代码：

1) 水平 level_t
   把原始值里"量级"的部分剥出来。定义 level_t = mean(y[t-7 .. t-1])，
   即"不含当天的过去 7 天均值"（因果，不用未来信息）。
   这样做的好处：2013-07 到 2014-02 数据涨了 10 倍，但 ratio 一直稳定在 0.7~1.3，
   模型学到的规律不会因为量级漂移而失效。

2) 季节指数 ratio_t
   ratio_t = y_t / level_t。它约等于"星期几因子 + 短期扰动"，
   均值≈1，取值范围可控，是 LSTM 真正要预测的东西。

3) 最终预测
   y_hat(T+h) = ratio_hat(h) * level_hat(T+h)
   其中 level_hat 由"对数空间加权最小二乘 + 阻尼外推"给出（见 forecast_level_path）。
   LSTM 负责形状（周内效应、节假日、短期动量），水平外推交给一个可解释的线性模型。
   这种拆法在样本量小（本数据只有 427 天）时比让 LSTM 直接吐原始值稳得多。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import Config, HOLIDAYS, MAKEUP_WORKDAYS


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------
@dataclass
class PreparedData:
    """清洗并对齐后的日频数据，所有数组第 i 行对应 dates[i]。"""

    dates: pd.DatetimeIndex      # 日期
    raw: np.ndarray              # (N, T) 原始值：申购、赎回
    level: np.ndarray            # (N, T) 因果水平（过去 level_ma 天均值，不含当天）
    ratio: np.ndarray            # (N, T) 季节指数 = raw / level
    rel_ratio: np.ndarray        # (N, T) 相对季节指数 = ratio / 输入窗口内 ratio 均值
    scale: np.ndarray            # (N, T) 上面那个"窗口内 ratio 均值"（预测时乘回去用）
    cal: np.ndarray              # (N, C) 日历特征（0/1 或 sin/cos）
    cal_names: List[str]
    target_names: List[str]

    def __len__(self) -> int:
        return len(self.dates)

    def index_of(self, date) -> int:
        """返回某日期在数组中的下标；找不到时抛异常（不要静默用 -1）。"""
        ts = pd.Timestamp(date)
        idx = int(np.searchsorted(self.dates.values, ts.to_datetime64(), side="left"))
        if idx >= len(self.dates) or self.dates[idx] != ts:
            raise ValueError(f"日期 {ts.date()} 不在数据集中（范围 {self.dates[0].date()} ~ {self.dates[-1].date()}）")
        return idx


# ---------------------------------------------------------------------------
# 读取与清洗
# ---------------------------------------------------------------------------
def load_daily(cfg: Config) -> pd.DataFrame:
    """
    读 csv，统一成"每个自然日一行"的日频表。

    兼容两种日期格式：20130701 / 2013-07-01。
    如果原始数据缺某些天（周末没记录等），用时间插值补齐——本数据集是满的，
    但换成别的品种时这一步很重要，否则 rolling 会错位。
    """
    df = pd.read_csv(cfg.csv_path)
    if cfg.date_col not in df.columns:
        raise ValueError(f"csv 里找不到日期列 {cfg.date_col!r}，实际列名：{list(df.columns)}")
    for t in cfg.targets:
        if t not in df.columns:
            raise ValueError(f"csv 里找不到目标列 {t!r}，实际列名：{list(df.columns)}")

    s = df[cfg.date_col].astype(str).str.strip()
    if s.str.fullmatch(r"\d{8}").all():
        dates = pd.to_datetime(s, format="%Y%m%d")
    else:
        dates = pd.to_datetime(s, errors="coerce")

    out = pd.DataFrame({t: pd.to_numeric(df[t], errors="coerce") for t in cfg.targets})
    out.index = pd.DatetimeIndex(dates).normalize()
    out = out[~out.index.isna()]

    # 同一天出现多条时求和（多数场景下是重复上报；如果你的数据是分渠道，这里语义要对）
    out = out.groupby(level=0).sum().sort_index()

    # 补齐日历
    full = pd.date_range(out.index.min(), out.index.max(), freq="D")
    out = out.reindex(full)

    n_missing = int(out.isna().any(axis=1).sum())
    if n_missing:
        out = out.interpolate(method="time", limit_direction="both")

    out.index.name = "date"
    out = out.astype(float)
    return out


# ---------------------------------------------------------------------------
# 日历特征
# ---------------------------------------------------------------------------
def calendar_features(dates: pd.DatetimeIndex, cfg: Config) -> Tuple[np.ndarray, List[str]]:
    """
    构造日历特征。全部是"事先已知"的变量，预测 9 月时可以直接算出来，不存在泄漏。

    - dow_0..dow_6 : 星期几 one-hot（周一=0，周日=6）
    - is_holiday   : 法定节假日
    - is_makeup    : 调休上班的周末
    - dom_sin/cos  : 月内位置（月初/月末效应），用 sin/cos 避免 1 号和 31 号被当成两端
    """
    n = len(dates)
    dow = dates.dayofweek.values
    onehot = np.zeros((n, 7), dtype=np.float32)
    onehot[np.arange(n), dow] = 1.0

    day = dates.normalize()
    holiday_set = pd.DatetimeIndex(list(HOLIDAYS))
    makeup_set = pd.DatetimeIndex(list(MAKEUP_WORKDAYS))
    is_holiday = day.isin(holiday_set).astype(np.float32)
    is_makeup = day.isin(makeup_set).astype(np.float32)

    dom = dates.day.values.astype(np.float32)
    dom_sin = np.sin(2 * np.pi * (dom - 1) / 31.0).astype(np.float32)
    dom_cos = np.cos(2 * np.pi * (dom - 1) / 31.0).astype(np.float32)

    names = [f"dow_{i}" for i in range(7)]
    if cfg.use_holiday_feature:
        blocks = [onehot, is_holiday[:, None], is_makeup[:, None], dom_sin[:, None], dom_cos[:, None]]
        names += ["is_holiday", "is_makeup", "dom_sin", "dom_cos"]
    else:
        blocks = [onehot, dom_sin[:, None], dom_cos[:, None]]
        names += ["dom_sin", "dom_cos"]

    cal = np.concatenate(blocks, axis=1).astype(np.float32)
    return cal, names


# ---------------------------------------------------------------------------
# 准备
# ---------------------------------------------------------------------------
def prepare(cfg: Config, df: Optional[pd.DataFrame] = None) -> PreparedData:
    """把原始日频表变成 PreparedData。"""
    if df is None:
        df = load_daily(cfg)

    target_names = list(cfg.targets)
    y = df[target_names].to_numpy(dtype=np.float64)

    # 水平：过去 level_ma 天（不含当天）的均值
    level = pd.DataFrame(y, index=df.index).shift(1).rolling(cfg.level_ma, min_periods=cfg.level_ma).mean()
    level = level.to_numpy(dtype=np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = y / level
    ratio = np.clip(ratio, 0.0, cfg.ratio_clip)

    # 目标归一化用的分母：输入窗口内（含当天）的 ratio 均值。
    # 它是一个"当下常态水平"的估计，未来 30 天的目标都除以它，
    # 这样模型学的是"相对现在的形状"，不会把 2013 年上涨期的 ratio 均值带进来。
    win = max(int(cfg.lookback), 7)
    ratio_df = pd.DataFrame(ratio, index=df.index)
    scale = ratio_df.rolling(win, min_periods=max(7, win // 2)).mean()
    scale = scale.bfill().fillna(1.0).to_numpy(dtype=np.float64)
    scale = np.maximum(scale, 1e-3)
    rel_ratio = np.clip(ratio / scale, 0.0, cfg.ratio_clip)

    cal, cal_names = calendar_features(df.index, cfg)

    # 丢掉开头 level 还没满的几天，保证后面所有数组都是有效值
    first = cfg.level_ma
    ds = PreparedData(
        dates=df.index[first:],
        raw=y[first:].astype(np.float32),
        level=level[first:].astype(np.float32),
        ratio=ratio[first:].astype(np.float32),
        rel_ratio=rel_ratio[first:].astype(np.float32),
        scale=scale[first:].astype(np.float32),
        cal=cal[first:],
        cal_names=cal_names,
        target_names=target_names,
    )
    return ds


# ---------------------------------------------------------------------------
# 特征拼装 + 标准化
# ---------------------------------------------------------------------------
class FeatureBuilder:
    """
    负责把 (ratio, level, calendar) 拼成模型输入，并做标准化。

    为什么单独抽一个类：训练、回测、滚动预测三处必须用完全一致的列顺序和
    标准化参数，否则"回测看着好、预测结果离谱"这种 bug 极难查。
    """

    def __init__(self, cfg: Config, ds: PreparedData, target: str):
        self.cfg = cfg
        self.ds = ds
        self.target = target

        # 单目标模型 or 双目标模型
        used = list(ds.target_names) if cfg.use_both_targets else [target]
        self.used = used
        self.tidx = [ds.target_names.index(t) for t in used]

        names = [f"ratio_{t}" for t in used]
        if cfg.use_level_feature:
            names += [f"lvl_{t}" for t in used]
        names += ds.cal_names
        self.names = names

        self.mu = np.zeros(len(names), dtype=np.float32)
        self.sd = np.ones(len(names), dtype=np.float32)
        self._fitted = False

    # -- 原始（未标准化）特征 ------------------------------------------------
    def raw(self, rows: np.ndarray) -> np.ndarray:
        """给定行下标，返回未标准化特征 (len(rows), F)。"""
        rows = np.asarray(rows)
        r = self.ds.ratio[np.ix_(rows, self.tidx)]
        blocks = [r]
        if self.cfg.use_level_feature:
            lv = np.log1p(np.maximum(self.ds.level[np.ix_(rows, self.tidx)], 1.0))
            blocks.append(lv)
        blocks.append(self.ds.cal[rows])
        return np.concatenate(blocks, axis=1).astype(np.float32)

    def raw_from_values(self, ratio_used: np.ndarray, level_used: np.ndarray, cal: np.ndarray) -> np.ndarray:
        """
        用显式给定的 ratio/level 构造特征行（滚动预测时用）。
        ratio_used / level_used: (n, n_used)；cal: (n, n_cal)
        """
        blocks = [np.asarray(ratio_used, dtype=np.float32)]
        if self.cfg.use_level_feature:
            blocks.append(np.log1p(np.maximum(np.asarray(level_used, dtype=np.float32), 1.0)))
        blocks.append(np.asarray(cal, dtype=np.float32))
        return np.concatenate(blocks, axis=1).astype(np.float32)

    # -- 标准化 --------------------------------------------------------------
    def fit(self, lo: int, hi: int) -> "FeatureBuilder":
        """用 [lo, hi) 这些天（只应该来自训练区间！）统计均值/标准差。"""
        X = self.raw(np.arange(lo, hi))
        self.mu = X.mean(axis=0).astype(np.float32)
        sd = X.std(axis=0).astype(np.float32)
        sd[sd < 1e-8] = 1.0
        self.sd = sd
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("FeatureBuilder 还没 fit，标准化参数未初始化")
        return ((X - self.mu) / self.sd).astype(np.float32)

    def build(self, rows: np.ndarray) -> np.ndarray:
        return self.transform(self.raw(rows))

    def build_from_values(self, ratio_used, level_used, cal) -> np.ndarray:
        return self.transform(self.raw_from_values(ratio_used, level_used, cal))


def feature_names_for(cfg: Config, ds: PreparedData, target: str) -> List[str]:
    """返回某个目标模型的特征名列（用于打印/保存，方便核对）。"""
    return FeatureBuilder(cfg, ds, target).names


# ---------------------------------------------------------------------------
# 窗口
# ---------------------------------------------------------------------------
def valid_anchors(n_days: int, lookback: int, horizon: int) -> np.ndarray:
    """
    可用的"锚点"：锚点 t 的第 t 天是输入窗口的最后一天，
    预测 t+1 .. t+horizon。
    """
    last = n_days - horizon - 1
    if last < lookback - 1:
        return np.array([], dtype=int)
    return np.arange(lookback - 1, last + 1, dtype=int)


def build_windows(
    X: np.ndarray,
    ratio: np.ndarray,
    anchors: np.ndarray,
    target_col: int,
    lookback: int,
    horizon: int,
    head: str = "multistep",
    cal: Optional[np.ndarray] = None,
):
    """
    生成监督学习样本。

    X      : (N, F) 每日特征（已标准化）
    ratio  : (N, T) 目标序列（ratio 或 相对 ratio）
    cal    : (N, C) 未标准化的日历特征（future 头需要，作为"已知的未来协变量"）
    anchors: 锚点下标数组

    返回：
        multistep: (Xw (n, L, F), Yw (n, H))
        future   : (Xw, Yw, Fw (n, H, C+1))，最后一列是 第几天/总天数

    说明：窗口逐日滑动，样本之间高度重叠，所以切分训练/验证集时必须留 gap，
    不然验证集的输入里就包含了训练集的目标，指标会虚高。
    """
    n = len(anchors)
    Xw = np.empty((n, lookback, X.shape[1]), dtype=np.float32)
    Yw = np.empty((n, horizon), dtype=np.float32)

    use_future = head == "future" and cal is not None and horizon > 1
    if use_future:
        Fw = np.empty((n, horizon, cal.shape[1] + 1), dtype=np.float32)
        hs = (np.arange(1, horizon + 1) / horizon).astype(np.float32)
    else:
        Fw = None

    for i, t in enumerate(anchors):
        Xw[i] = X[t - lookback + 1: t + 1]
        Yw[i] = ratio[t + 1: t + 1 + horizon, target_col]
        if Fw is not None:
            Fw[i, :, :-1] = cal[t + 1: t + 1 + horizon]
            Fw[i, :, -1] = hs

    return (Xw, Yw, Fw) if Fw is not None else (Xw, Yw)


def split_train_val(anchors: np.ndarray, cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    """按时间顺序切训练/验证，并在中间留 out_len 天的空档。"""
    n = len(anchors)
    n_val = max(int(cfg.min_val), int(round(n * cfg.val_ratio)))
    if n - n_val < 10:
        raise ValueError(f"样本太少（{n} 条），无法切出训练集与验证集")
    val_start = int(anchors[n - n_val])
    cut = val_start - max(cfg.val_gap, 1)
    tr = anchors[anchors <= cut]
    va = anchors[anchors >= val_start]
    if len(tr) < 10:
        raise ValueError("留 gap 后训练样本不足，请减小 val_ratio / val_gap / out_len")
    return tr, va


# ---------------------------------------------------------------------------
# 水平外推
# ---------------------------------------------------------------------------
@dataclass
class LevelPath:
    """未来 horizon 天的水平估计。"""

    path: np.ndarray      # (H,) 每天的水平
    level_now: float      # 锚点当天的水平估计
    slope: float          # log 空间日斜率（已截断）
    total_drift: float    # horizon 天累计漂移（log 空间）


def _weighted_linear_fit(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> Tuple[float, float]:
    """加权最小二乘 y = a + b*x，返回 (b, a)。"""
    sw = w.sum()
    if sw <= 0 or len(x) < 2:
        return 0.0, float(np.mean(y)) if len(y) else 0.0
    xb = float((w * x).sum() / sw)
    yb = float((w * y).sum() / sw)
    sxx = float((w * (x - xb) ** 2).sum())
    if sxx < 1e-12:
        return 0.0, yb
    sxy = float((w * (x - xb) * (y - yb)).sum())
    b = sxy / sxx
    a = yb - b * xb
    return b, a


def forecast_level_path(values: np.ndarray, horizon: int, cfg: Config) -> LevelPath:
    """
    预测未来 horizon 天的"水平"（即日均值量级）。

    两种模式：
    - flat  （默认）：EWMA 估一个当前水平，整段保持不变。
            这份数据 2014-03 之后水平基本横盘，任何趋势外推都会系统性高估
            （回测里趋势外推 WAPE≈31%，flat≈20%）。1 个月尺度上，均值回复
            比趋势延续更靠谱。
    - trend        ：log 空间加权最小二乘 + 阻尼外推，适合真的有持续趋势的品种。

    参数 values 是"截至锚点当天（含）"的原始日值序列。
    """
    v = np.asarray(values, dtype=np.float64)
    v = np.maximum(v, 0.0)
    n_all = len(v)
    if n_all == 0 or np.allclose(v, 0):
        return LevelPath(path=np.zeros(horizon), level_now=0.0, slope=0.0, total_drift=0.0)

    if cfg.level_mode == "flat":
        # 指数加权平均：越近的天权重越高，比"简单 28 日均值"更跟得上水平切换
        n = int(min(n_all, 180))
        y = v[-n:]
        w = np.power(0.5, ((n - 1) - np.arange(n, dtype=np.float64)) / max(cfg.level_ewma_halflife, 1e-6))
        level = float((w * y).sum() / w.sum())
        return LevelPath(path=np.full(horizon, level, dtype=np.float64), level_now=level, slope=0.0, total_drift=0.0)

    if cfg.level_mode != "trend":
        raise ValueError(f"未知 level_mode: {cfg.level_mode!r}（应为 flat 或 trend）")

    w_len = int(min(max(cfg.level_window, 14), n_all))
    y = np.log1p(v[-w_len:])
    x = np.arange(w_len, dtype=np.float64)

    # 越近的样本权重越大（指数衰减，半衰期 level_halflife 天）
    age = (w_len - 1) - x
    w = np.power(0.5, age / max(cfg.level_halflife, 1e-6))

    b, a = _weighted_linear_fit(x, y, w)
    b = float(np.clip(b, -cfg.max_daily_slope, cfg.max_daily_slope))

    fitted = a + b * x
    fitted_end = float(a + b * (w_len - 1))

    # 锚定：让拟合曲线最近 level_anchor 天的均值，等于实际观测的均值。
    # 这一步解决"拟合线整体偏高/偏低"造成的系统性偏差（log 空间做加法修正）。
    m = int(min(max(cfg.level_anchor, 1), w_len))
    offset = float(y[-m:].mean() - fitted[-m:].mean())
    fitted_end += offset

    # 阻尼外推
    k = np.arange(1, horizon + 1, dtype=np.float64)
    step = b * np.power(cfg.damp, k)
    drift = np.cumsum(step)
    drift = np.clip(drift, -cfg.max_total_drift, cfg.max_total_drift)

    path = np.expm1(fitted_end + drift)
    path = np.maximum(path, 0.0)
    return LevelPath(path=path, level_now=float(np.expm1(fitted_end)), slope=float(b), total_drift=float(drift[-1]))


# ---------------------------------------------------------------------------
# 回测折
# ---------------------------------------------------------------------------
@dataclass
class Fold:
    """一次"用 train_end 之前的数据训练，预测 test_start~test_end"的回测。"""

    name: str
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    @property
    def is_month(self) -> bool:
        return self.test_start.day == 1


def make_month_folds(ds: PreparedData, cfg: Config) -> List[Fold]:
    """
    按月构造 walk-forward 折："预测整个月 M" —— 与 9 月的任务完全同构。

    每折的训练数据只到 M 月的上一天（含），不碰 M 月任何一天，避免泄漏。
    """
    start = pd.Timestamp(cfg.backtest_start).normalize().replace(day=1)
    end = pd.Timestamp(cfg.backtest_end).normalize()
    folds: List[Fold] = []

    for month_start in pd.date_range(start, end, freq="MS"):
        month_end = (month_start + pd.offsets.MonthEnd(0)).normalize()
        test_start = max(month_start, ds.dates[0])
        test_end = min(month_end, ds.dates[-1])
        if test_end < test_start:
            continue
        train_end = month_start - pd.Timedelta(days=1)
        if train_end < ds.dates[0] or train_end > ds.dates[-1]:
            continue
        # 训练天数不够就跳过（前几个月历史太短，结果没有参考价值）
        n_train = int(np.sum(ds.dates <= train_end))
        if n_train < cfg.min_train_days:
            continue
        folds.append(
            Fold(
                name=f"{month_start:%Y-%m}",
                train_end=pd.Timestamp(train_end),
                test_start=pd.Timestamp(test_start),
                test_end=pd.Timestamp(test_end),
            )
        )
    return folds


def forecast_dates(cfg: Config) -> pd.DatetimeIndex:
    """预测区间的日期序列（2014-09-01 ~ 2014-09-30），完全由配置决定。"""
    start = pd.Timestamp(cfg.forecast_start).normalize()
    end = pd.Timestamp(cfg.forecast_end).normalize()
    if end < start:
        raise ValueError("forecast_end 早于 forecast_start")
    return pd.date_range(start, end, freq="D")


def calendar_rows_for(dates: pd.DatetimeIndex, cfg: Config, cal_names: Sequence[str]) -> np.ndarray:
    """
    给定未来日期，返回日历特征（列顺序与训练时一致）。
    未来日期的日历是确定的，所以这里没有任何泄漏问题。
    """
    cal, names = calendar_features(dates, cfg)
    if list(names) != list(cal_names):
        # 理论上不会发生；真发生了说明改过特征开关，直接报错比静默错位好
        raise ValueError(f"日历特征列不一致：{names} vs {cal_names}")
    return cal
