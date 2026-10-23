"""
最终任务：预测 2014 年 9 月整月的申购 / 赎回。

流程：
    1) 用 2013-07-01 ~ 2014-08-31 全部数据，分别为申购、赎回训练 LSTM 集成
    2) 从 2014-08-31 这个锚点往后推 30 天
    3) 预测值 = 模型输出的季节指数 × 水平外推路径
    4) 落盘：逐日预测 csv、月度汇总 csv、图
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .config import Config
from .data_prep import PreparedData, forecast_dates
from .pipeline import Prediction, TrainedTarget, predict_horizon, train_target
from .plots import plot_forecast, plot_recent_zoom, plot_training_curves
from .utils import ensure_dir, log


@dataclass
class ForecastResult:
    pred_df: pd.DataFrame
    summary: pd.DataFrame
    trained: Dict[str, TrainedTarget]
    pred: Prediction
    paths: Dict[str, str]
    tag: str = ""


def run_forecast(
    cfg: Config,
    ds: PreparedData,
    device: str = "cpu",
    rel_err_std: Optional[Dict[str, float]] = None,
    verbose: bool = True,
) -> ForecastResult:
    """
    训练 + 预测 + 落盘。

    rel_err_std: 各目标在回测中"相对误差"的标准差，用来画经验区间。
                 传 None 时用 0.20 作为默认值（并会在日志里提示这是拍脑袋的数）。
    """
    out_dir = ensure_dir(cfg.out_dir)
    dates = forecast_dates(cfg)
    if rel_err_std is None:
        log("⚠ 没有回测残差可用，预测区间按固定的 ±20% 相对误差画（跑 --task all 会用回测残差）")

    origin_idx = len(ds) - 1
    origin_date = ds.dates[origin_idx]
    expected_start = origin_date + pd.Timedelta(days=1)
    if dates[0] != expected_start:
        log(f"⚠ 预测起点 {dates[0].date()} 与数据最后一天 {origin_date.date()} 不连续，请确认这是有意的")
    if len(dates) > cfg.out_len and cfg.mode == "direct":
        raise ValueError(f"预测 {len(dates)} 天但 out_len={cfg.out_len}，请调大 out_len")

    log(f"预测区间 {dates[0].date()} ~ {dates[-1].date()}（{len(dates)} 天），锚点 {origin_date.date()}")

    # ---- 1. 两个目标分别训练 ----
    trained: Dict[str, TrainedTarget] = {}
    for target in ds.target_names:
        trained[target] = train_target(cfg, ds, target, origin_idx, device=device, verbose=verbose)

    # ---- 2. 预测 ----
    pred = predict_horizon(cfg, ds, trained, origin_idx, dates, device)
    for target, tm in trained.items():
        log(
            f"  {target}: 水平 {pred.level[target][0] / 1e6:.1f}M → {pred.level[target][-1] / 1e6:.1f}M，"
            f"log 日斜率 {pred.slope[target]:+.4f}，累计漂移 {pred.drift[target]:+.3f}"
        )

    # ---- 3. 组装逐日表 ----
    pred_df = pd.DataFrame({"date": dates.date})
    pred_df["date"] = pd.to_datetime(pred_df["date"])
    z = 1.2816  # 80% 区间
    for target in ds.target_names:
        v = pred.value[target]
        r = pred.ratio[target]
        lv = pred.level[target]
        pred_df[f"{target}_pred"] = np.round(v, 0)
        pred_df[f"{target}_ratio"] = np.round(r, 4)
        pred_df[f"{target}_level"] = np.round(lv, 0)
        sd = float((rel_err_std or {}).get(target, 0.20))
        pred_df[f"{target}_lo"] = np.round(v * (1 - z * sd), 0)
        pred_df[f"{target}_hi"] = np.round(v * (1 + z * sd), 0)

    # 参考基线：周内形状 × 水平（不涉及 LSTM，用来看模型有没有明显跑偏）
    from .baselines import baseline_weekday_profile_level  # 延迟导入，避免循环依赖

    for target in ds.target_names:
        ti = ds.target_names.index(target)
        pred_df[f"{target}_baseline"] = np.round(
            baseline_weekday_profile_level(cfg, ds, origin_idx, dates, ti, weeks=8), 0
        )

    # ---- 4. 月度汇总 ----
    rows = []
    for target in ds.target_names:
        v = pred_df[f"{target}_pred"]
        base = pred_df[f"{target}_baseline"]
        ti = ds.target_names.index(target)
        aug_mask = ds.dates >= (dates[0] - pd.Timedelta(days=1)).replace(day=1)
        aug_total = float(ds.raw[aug_mask, ti].sum())
        peak_i = int(np.argmax(v.values))
        low_i = int(np.argmin(v.values))
        rows.append({
            "target": target,
            "month_total": float(v.sum()),
            "baseline_total": float(base.sum()),
            "daily_mean": float(v.mean()),
            "peak_date": pred_df["date"].iloc[peak_i].date(),
            "peak_value": float(v.iloc[peak_i]),
            "low_date": pred_df["date"].iloc[low_i].date(),
            "low_value": float(v.iloc[low_i]),
            "aug_total": aug_total,
            "mom_pct": (float(v.sum()) - aug_total) / aug_total * 100 if aug_total else np.nan,
        })
    summary = pd.DataFrame(rows)

    # ---- 5. 落盘 ----
    tag = pd.Timestamp(cfg.forecast_start).strftime("%Y-%m")
    paths: Dict[str, str] = {}
    daily_path = os.path.join(out_dir, f"forecast_{tag}_daily.csv")
    pred_df.to_csv(daily_path, index=False, encoding="utf-8-sig")
    paths["daily"] = daily_path

    summary_path = os.path.join(out_dir, f"forecast_{tag}_summary.csv")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    paths["summary"] = summary_path

    # ---- 5b. 提交格式：无表头，每行 "YYYYMMDD,申购预测值,赎回预测值" ----
    # 这是给下游/业务方直接用的最简格式，日期用 20140901 这种紧凑写法，数值取整。
    sub_path = os.path.join(out_dir, f"submission_{tag}.csv")
    sub_cols = [pd.to_datetime(pred_df["date"]).dt.strftime("%Y%m%d")]
    for target in ds.target_names:
        sub_cols.append(pred_df[f"{target}_pred"].round(0).astype("int64"))
    pd.concat(sub_cols, axis=1).to_csv(sub_path, index=False, header=False, encoding="utf-8")
    paths["submission"] = sub_path

    # 训练曲线（只用第一个种子，够看趋势）
    try:
        hists = {
            t: {"train": tm.hist.train_loss, "val": tm.hist.val_loss}
            for t, tm in trained.items()
        }
        best = {t: tm.hist.best_epoch for t, tm in trained.items()}
        p = os.path.join(out_dir, "training_curves.png")
        plot_training_curves(hists, best, p)
        paths["training_curves"] = p
    except Exception as e:  # 画图失败不应该影响主流程
        log(f"⚠ 训练曲线绘制失败：{e}")

    if cfg.plots:
        try:
            p = os.path.join(out_dir, f"forecast_{tag}.png")
            plot_forecast(ds, pred_df, cfg, p, title_note=f"{tag} forecast")
            paths["forecast"] = p

            p2 = os.path.join(out_dir, "forecast_zoom_90d.png")
            plot_recent_zoom(ds, pred_df, cfg, p2, days=90)
            paths["zoom"] = p2
        except Exception as e:
            log(f"⚠ 绘图失败：{e}")

    if cfg.save_models:
        try:
            import torch

            model_dir = ensure_dir(os.path.join(out_dir, "models"))
            for t, tm in trained.items():
                for k, m in enumerate(tm.models):
                    torch.save(
                        {
                            "state_dict": m.state_dict(),
                            "n_features": m.lstm.input_size,
                            "hidden": cfg.hidden,
                            "out_len": cfg.out_len,
                            "y_mu": tm.y_mu,
                            "y_sd": tm.y_sd,
                            "feature_names": tm.builder.names,
                            "feat_mu": tm.builder.mu,
                            "feat_sd": tm.builder.sd,
                        },
                        os.path.join(model_dir, f"{t}_seed{k}.pt"),
                    )
            paths["models"] = model_dir
        except Exception as e:
            log(f"⚠ 模型保存失败：{e}")

    return ForecastResult(pred_df=pred_df, summary=summary, trained=trained, pred=pred,
                          paths=paths, tag=tag)


def print_summary(res: ForecastResult) -> None:
    """在控制台打印月度汇总，单位换算成亿，方便读。"""
    from .metrics import format_table

    rows = []
    for _, r in res.summary.iterrows():
        rows.append({
            "target": r["target"],
            "month_total_yi": r["month_total"] / 1e8,
            "baseline_yi": r["baseline_total"] / 1e8,
            "daily_mean_wan": r["daily_mean"] / 1e4,
            "peak": f"{r['peak_date']} ({r['peak_value'] / 1e8:.2f}亿)",
            "low": f"{r['low_date']} ({r['low_value'] / 1e8:.2f}亿)",
            "aug_yi": r["aug_total"] / 1e8,
            "mom_pct": r["mom_pct"],
        })
    print()
    print(f"=== {res.tag or '预测'} 预测汇总 ===")
    print(format_table(rows, [
        ("target", "目标", "s"),
        ("month_total_yi", "预测月总量(亿)", ".2f"),
        ("baseline_yi", "基线月总量(亿)", ".2f"),
        ("daily_mean_wan", "日均(万)", ".0f"),
        ("peak", "峰值日", "s"),
        ("low", "低谷日", "s"),
        ("aug_yi", "8月实际(亿)", ".1f"),
        ("mom_pct", "环比%", "+.1f"),
    ]))
