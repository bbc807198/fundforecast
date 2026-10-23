"""
评估指标与统计检验。

为什么不用单一指标：
  - MAE/RMSE       受量级影响，跨月份不可比（1 月和 8 月的量级差 10 倍）
  - MAPE           在数值小的日子会被放大，2013 年那一段数据会主导结果
  - WAPE（推荐）   sum(|误差|)/sum(实际值)，等价于"总量视角下的误差率"，
                   对量级不敏感、不会被小数值日期带偏，和业务上"月度总量准不准"最贴近
  - total_err_pct  月总量误差，最直观的考核口径
  - 偏差 bias      平均误差的符号，能看出系统性高估/低估
"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np


EPS = 1e-9


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """一次性算齐常用指标（百分比指标以 % 为单位，例如 12.3 表示 12.3%）。"""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(f"形状不一致: {y_true.shape} vs {y_pred.shape}")
    if len(y_true) == 0:
        raise ValueError("空数组无法计算指标")

    err = y_pred - y_true
    abs_err = np.abs(err)

    mae = float(abs_err.mean())
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mape = float(np.mean(abs_err / (np.abs(y_true) + EPS)) * 100.0)
    smape = float(np.mean(2.0 * abs_err / (np.abs(y_true) + np.abs(y_pred) + EPS)) * 100.0)
    wape = float(abs_err.sum() / (np.abs(y_true).sum() + EPS) * 100.0)
    bias = float(err.mean() / (np.abs(y_true).mean() + EPS) * 100.0)
    total_true = float(y_true.sum())
    total_pred = float(y_pred.sum())
    total_err_pct = float((total_pred - total_true) / (abs(total_true) + EPS) * 100.0)

    return {
        "n": int(len(y_true)),
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "smape": smape,
        "wape": wape,
        "bias_pct": bias,
        "total_true": total_true,
        "total_pred": total_pred,
        "total_err_pct": total_err_pct,
    }


def paired_block_bootstrap(
    y_true: np.ndarray,
    err_a: np.ndarray,
    err_b: np.ndarray,
    block: int = 7,
    n_boot: int = 2000,
    seed: int = 0,
) -> Dict[str, float]:
    """
    配对分块 bootstrap：判断模型 A 的 WAPE 是否显著优于模型 B。

    为什么分块：相邻日期的误差高度相关（周末一起低、节前一起高），
    按天独立重采样会把方差低估，容易得出"显著"的假结论。这里按周（7 天）分块重采样。

    返回
    ----
    diff_mean : E[WAPE_A - WAPE_B]（负数表示 A 更好）
    ci90      : 差值的 90% 置信区间
    p_a_better: 重采样中 A 优于 B 的比例（≈ P(A 更好)）
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    err_a = np.asarray(err_a, dtype=np.float64).ravel()
    err_b = np.asarray(err_b, dtype=np.float64).ravel()
    n = len(y_true)
    if n == 0:
        return {"diff_mean": np.nan, "ci90_lo": np.nan, "ci90_hi": np.nan, "p_a_better": np.nan}

    block = max(1, int(block))
    n_blocks = int(np.ceil(n / block))
    starts = np.arange(n_blocks) * block
    rng = np.random.default_rng(seed)

    diffs = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        pick = rng.integers(0, n_blocks, size=n_blocks)
        idx = np.concatenate([np.arange(s, min(s + block, n)) for s in starts[pick]])
        denom = y_true[idx].sum() + EPS
        wape_a = err_a[idx].sum() / denom
        wape_b = err_b[idx].sum() / denom
        diffs[b] = wape_a - wape_b

    return {
        "diff_mean": float(np.mean(diffs)),
        "ci90_lo": float(np.percentile(diffs, 5)),
        "ci90_hi": float(np.percentile(diffs, 95)),
        "p_a_better": float(np.mean(diffs < 0)),
    }


def skill_score(model_wape: float, base_wape: float) -> float:
    """技能分 = 1 - 模型WAPE/基线WAPE。0 表示和基线一样，0.2 表示误差降低 20%。"""
    if base_wape <= 0:
        return float("nan")
    return 1.0 - model_wape / base_wape


def format_table(rows: Iterable[Dict], columns: Iterable[Tuple[str, str, str]]) -> str:
    """
    简单的等宽表输出。
    columns: [(字段名, 表头, 格式)]，例如 ("wape", "WAPE%", ".2f")
    """
    cols = list(columns)
    header = [c[1] for c in cols]
    body = []
    for r in rows:
        line = []
        for key, _, fmt in cols:
            v = r.get(key, "")
            if isinstance(v, (int, np.integer)):
                line.append(str(v).ljust(0))
            elif isinstance(v, (float, np.floating)):
                line.append(format(float(v), fmt))
            else:
                line.append(str(v))
        body.append(line)

    widths = [max(len(header[i]), *(len(b[i]) for b in body)) if body else len(header[i]) for i in range(len(cols))]
    out = [" | ".join(header[i].ljust(widths[i]) for i in range(len(cols)))]
    out.append("-+-".join("-" * widths[i] for i in range(len(cols))))
    for b in body:
        out.append(" | ".join(b[i].ljust(widths[i]) for i in range(len(cols))))
    return "\n".join(out)
