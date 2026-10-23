"""
滚动回测（walk-forward）：判断"模型是不是真的变好了"的唯一可靠方式。

做法：对 2014-04 ~ 2014-08 的每个月 M：
    用 M 月之前的所有数据训练 -> 预测整个 M 月 -> 和真实值比
这跟"用 8 月及之前的数据预测 9 月"完全同构，所以回测的分数才有参考价值。

输出三张表：
    1) 分折明细（每个月、每个模型、每个目标）
    2) 汇总（均值/最差折/月总量误差/相对最强基线的技能分）
    3) 配对分块 bootstrap（"变好"是不是偶然）
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .baselines import all_baselines
from .config import Config
from .data_prep import Fold, PreparedData, make_month_folds
from .metrics import (
    format_table,
    paired_block_bootstrap,
    regression_metrics,
    skill_score,
)
from .pipeline import predict_horizon, train_target
from .plots import plot_backtest
from .utils import ensure_dir, log

LSTM_NAME = "lstm_ensemble"
PROFILE_NAME = "weekday_profile"


@dataclass
class BacktestResult:
    detail: pd.DataFrame                 # 分折 x 目标 x 模型的指标
    summary: pd.DataFrame                # 汇总表
    per_day: pd.DataFrame                # 逐日预测（画图 + bootstrap 用）
    bootstrap: pd.DataFrame              # 显著性检验
    rel_err_std: Dict[str, float]        # 各目标相对误差标准差（给预测区间用）
    folds: List[Fold] = field(default_factory=list)

    def best_model(self, target: str) -> str:
        sub = self.summary[self.summary["target"] == target]
        return str(sub.sort_values("wape_mean").iloc[0]["model"])


def run_backtest(cfg: Config, ds: PreparedData, device: str = "cpu", verbose: bool = True) -> BacktestResult:
    folds = make_month_folds(ds, cfg)
    if not folds:
        raise ValueError("没有可用的回测折，请检查 backtest_start/backtest_end/min_train_days")

    if verbose:
        log(f"回测折 {len(folds)} 个：" + ", ".join(f.name for f in folds))

    detail_rows: List[Dict] = []
    per_day_rows: List[Dict] = []

    for fold in folds:
        train_end_idx = ds.index_of(fold.train_end)
        test_dates = pd.date_range(fold.test_start, fold.test_end, freq="D")
        test_idx = ds.index_of(test_dates[0]) + np.arange(len(test_dates))

        if verbose:
            log(f"[{fold.name}] 训练至 {fold.train_end.date()}，预测 {len(test_dates)} 天")

        # ---- 训练两个 LSTM ----
        trained = {}
        for target in ds.target_names:
            trained[target] = train_target(
                cfg, ds, target, train_end_idx, device=device, verbose=False
            )

        # ---- LSTM 预测 ----
        pred = predict_horizon(cfg, ds, trained, train_end_idx, test_dates, device)

        for target in ds.target_names:
            ti = ds.target_names.index(target)
            actual = ds.raw[test_idx, ti].astype(np.float64)

            preds: Dict[str, np.ndarray] = {LSTM_NAME: np.asarray(pred.value[target], dtype=np.float64)}
            preds.update(all_baselines(cfg, ds, train_end_idx, test_dates, target))

            for model_name, p in preds.items():
                m = regression_metrics(actual, p)
                detail_rows.append({
                    "fold": fold.name,
                    "target": target,
                    "model": model_name,
                    "wape": m["wape"],
                    "mape": m["mape"],
                    "smape": m["smape"],
                    "mae": m["mae"],
                    "rmse": m["rmse"],
                    "total_err_pct": m["total_err_pct"],
                    "bias_pct": m["bias_pct"],
                })

                # 逐日明细：所有模型都存下来，bootstrap 才能两两比较
                for d, a, v in zip(test_dates, actual, p):
                    per_day_rows.append({
                        "date": d,
                        "fold": fold.name,
                        "target": target,
                        "model": model_name,
                        "actual": float(a),
                        "pred": float(v),
                    })

    detail = pd.DataFrame(detail_rows)
    per_day = pd.DataFrame(per_day_rows)

    # ---- 汇总 ----
    detail = detail.assign(abs_total_err=detail["total_err_pct"].abs())
    g = detail.groupby(["target", "model"], sort=True)
    summary = pd.DataFrame({
        "wape_mean": g["wape"].mean(),
        "wape_std": g["wape"].std(ddof=0),
        "wape_worst": g["wape"].max(),
        "mape_mean": g["mape"].mean(),
        "total_err_abs_mean": g["abs_total_err"].mean(),
        "bias_mean": g["bias_pct"].mean(),
        "folds": g["fold"].nunique(),
    }).reset_index()

    # 相对"周内形状×水平"基线（同一套水平外推，只是形状用统计量估）的技能分
    summary["skill_vs_profile"] = np.nan
    for target, sub in summary.groupby("target"):
        base_rows = sub[sub["model"] == PROFILE_NAME]
        if base_rows.empty:
            continue
        base = float(base_rows["wape_mean"].iloc[0])
        summary.loc[sub.index, "skill_vs_profile"] = 1.0 - sub["wape_mean"] / base

    # ---- 配对 bootstrap ----
    boot_rows = []
    rel_err_std: Dict[str, float] = {}
    if not per_day.empty:
        for target in ds.target_names:
            d = per_day[per_day["target"] == target]
            piv = d.pivot_table(index="date", columns="model", values=["actual", "pred"], aggfunc="first")
            piv = piv.sort_index()
            actual = piv[("actual", LSTM_NAME)].to_numpy(dtype=np.float64)
            lstm_pred = piv[("pred", LSTM_NAME)].to_numpy(dtype=np.float64)
            rel_err_std[target] = float(np.std((lstm_pred - actual) / np.maximum(actual, 1e-9)))

            err_lstm = np.abs(lstm_pred - actual)
            for model_name in d["model"].unique():
                if model_name == LSTM_NAME:
                    continue
                other_pred = piv[("pred", model_name)].to_numpy(dtype=np.float64)
                err_other = np.abs(other_pred - actual)
                bt = paired_block_bootstrap(
                    actual, err_lstm, err_other,
                    block=cfg.block, n_boot=cfg.bootstrap, seed=cfg.seed,
                )
                boot_rows.append({
                    "target": target,
                    "compare": f"vs {model_name}",
                    "diff_wape_pp": bt["diff_mean"] * 100,      # 百分点
                    "ci90_lo_pp": bt["ci90_lo"] * 100,
                    "ci90_hi_pp": bt["ci90_hi"] * 100,
                    "p_better": bt["p_a_better"],
                })
    bootstrap = pd.DataFrame(boot_rows)

    # ---- 画图 ----
    if cfg.plots and not per_day.empty:
        try:
            wide = per_day.pivot_table(
                index=["date", "fold", "target"], columns="model", values="pred", aggfunc="first"
            ).reset_index()
            act = per_day[per_day["model"] == LSTM_NAME][["date", "fold", "target", "actual"]]
            merged = act.merge(wide, on=["date", "fold", "target"], how="left")
            merged = merged.rename(columns={LSTM_NAME: "lstm"})
            p = os.path.join(ensure_dir(cfg.out_dir), "backtest.png")
            plot_backtest(merged, cfg, p)
        except Exception as e:
            log(f"⚠ 回测图绘制失败：{e}")

    return BacktestResult(
        detail=detail,
        summary=summary,
        per_day=per_day,
        bootstrap=bootstrap,
        rel_err_std=rel_err_std,
        folds=folds,
    )


def print_backtest(res: BacktestResult, cfg: Config) -> None:
    """控制台打印三张表。"""
    detail = res.detail
    print("\n=== 1. 分折明细（每折一个整月，WAPE 越小越好）===")
    rows = []
    for _, r in detail.sort_values(["target", "fold", "model"]).iterrows():
        rows.append({
            "target": r["target"],
            "fold": r["fold"],
            "model": r["model"],
            "wape": r["wape"],
            "total_err": r["total_err_pct"],
            "bias": r["bias_pct"],
        })
    print(format_table(rows, [
        ("target", "目标", "s"), ("fold", "月份", "s"), ("model", "模型", "s"),
        ("wape", "WAPE%", ".2f"), ("total_err", "月总量误差%", "+.2f"), ("bias", "偏差%", "+.2f"),
    ]))

    print("\n=== 2. 汇总（按目标看，行按 WAPE 排序）===")
    for target in sorted(res.summary["target"].unique()):
        sub = res.summary[res.summary["target"] == target].sort_values("wape_mean")
        rows = []
        for _, r in sub.iterrows():
            rows.append({
                "model": r["model"],
                "wape_mean": r["wape_mean"],
                "wape_std": r["wape_std"],
                "wape_worst": r["wape_worst"],
                "total_err_abs": r["total_err_abs_mean"],
                "bias": r["bias_mean"],
                "skill": r["skill_vs_profile"],
            })
        print(f"\n-- {target} --")
        print(format_table(rows, [
            ("model", "模型", "s"),
            ("wape_mean", "WAPE均值%", ".2f"),
            ("wape_std", "WAPE标准差", ".2f"),
            ("wape_worst", "最差折%", ".2f"),
            ("total_err_abs", "月总量平均绝对误差%", ".2f"),
            ("bias", "平均偏差%", "+.2f"),
            ("skill", "技能分(vs基线)", "+.3f"),
        ]))

    if not res.bootstrap.empty:
        print("\n=== 3. 配对分块 bootstrap（LSTM 是否显著优于基线）===")
        rows = []
        for _, r in res.bootstrap.iterrows():
            rows.append({
                "target": r["target"],
                "compare": r["compare"],
                "diff": r["diff_wape_pp"],
                "lo": r["ci90_lo_pp"],
                "hi": r["ci90_hi_pp"],
                "p": r["p_better"],
            })
        print(format_table(rows, [
            ("target", "目标", "s"), ("compare", "对比", "s"),
            ("diff", "WAPE差(pp)", "+.2f"), ("lo", "90%CI下", "+.2f"),
            ("hi", "90%CI上", "+.2f"), ("p", "P(LSTM更好)", ".2f"),
        ]))
        print("解读：WAPE差 < 0 且 90%CI 不跨 0 → LSTM 确实更好；"
              "差值在 0 附近、CI 跨 0 → 只能说'打平'，别硬说提升。")

    print("\n=== 4. 各折胜负 ===")
    win_rows = []
    for target in sorted(detail["target"].unique()):
        d = detail[detail["target"] == target]
        lstm = d[d["model"] == LSTM_NAME].set_index("fold")["wape"]
        for model_name in sorted(d["model"].unique()):
            if model_name == LSTM_NAME:
                continue
            other = d[d["model"] == model_name].set_index("fold")["wape"]
            common = lstm.index.intersection(other.index)
            win_rows.append({
                "target": target,
                "compare": model_name,
                "wins": f"{int((lstm[common] < other[common]).sum())}/{len(common)}",
            })
    print(format_table(win_rows, [
        ("target", "目标", "s"), ("compare", "对手", "s"), ("wins", "LSTM 胜出折数", "s"),
    ]))
