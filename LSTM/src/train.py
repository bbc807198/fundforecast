"""
训练逻辑：单个模型的训练循环 + 早停 + 多随机种子集成。

训练/验证的切分方式非常重要：
    时间序列不能随机打乱后再切分，否则验证集里会混进"未来"的信息，
    指标会虚高，然后上线就崩。这里一律按时间顺序切，并且训练集和验证集之间
    留出 val_gap 天，避免窗口重叠（相邻两天的样本共享 27/28 的输入）。 
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config
from .model import build_model, count_parameters
from .utils import set_seed


# ---------------------------------------------------------------------------
# 训练记录
# ---------------------------------------------------------------------------
@dataclass
class History:
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val: float = float("inf")
    n_params: int = 0


# ---------------------------------------------------------------------------
# 目标值标准化
# ---------------------------------------------------------------------------
def fit_target_scaler(y: np.ndarray) -> Tuple[float, float]:
    """对 ratio 目标做 z-score。返回 (mu, sd)。"""
    mu = float(np.mean(y))
    sd = float(np.std(y))
    if sd < 1e-8:
        sd = 1.0
    return mu, sd


# ---------------------------------------------------------------------------
# 训练单个模型
# ---------------------------------------------------------------------------
def train_one(
    cfg: Config,
    Xtr: np.ndarray,
    Ytr: np.ndarray,
    Xva: np.ndarray,
    Yva: np.ndarray,
    seed: int,
    device: str = "cpu",
    verbose: bool = False,
    Ftr: Optional[np.ndarray] = None,
    Fva: Optional[np.ndarray] = None,
) -> Tuple[nn.Module, History]:
    """
    训练一个 LSTM 并返回（恢复成最优权重的）模型与训练曲线。

    参数
    ----
    Xtr/Ytr : 训练集窗口，Y 已经标准化过
    Xva/Yva : 验证集窗口
    Ftr/Fva : future 头的"已知未来协变量"，其它结构传 None
    seed    : 随机种子（集成时每个成员用不同种子）
    """
    set_seed(seed)
    hist = History()

    n_future = 0 if Ftr is None else Ftr.shape[-1]
    model = build_model(cfg, Xtr.shape[-1], out_len=Ytr.shape[1], n_future=n_future).to(device)
    hist.n_params = count_parameters(model)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(cfg.epochs, 1), eta_min=cfg.lr * 0.02)

    xtr = torch.from_numpy(np.ascontiguousarray(Xtr)).to(device)
    ytr = torch.from_numpy(np.ascontiguousarray(Ytr)).to(device)
    xva = torch.from_numpy(np.ascontiguousarray(Xva)).to(device)
    yva = torch.from_numpy(np.ascontiguousarray(Yva)).to(device)
    ftr = torch.from_numpy(np.ascontiguousarray(Ftr)).to(device) if Ftr is not None else None
    fva = torch.from_numpy(np.ascontiguousarray(Fva)).to(device) if Fva is not None else None

    # 样本权重：默认全 1；sample_decay<1 时越近的样本权重越大（应对制度切换）
    n = len(xtr)
    if cfg.sample_decay < 1.0:
        w = np.power(cfg.sample_decay, (n - 1) - np.arange(n))
        w = w / w.mean()
    else:
        w = np.ones(n, dtype=np.float32)
    w_t = torch.from_numpy(w.astype(np.float32)).to(device)

    def _loss(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        if cfg.loss == "huber":
            per = F.smooth_l1_loss(pred, target, reduction="none", beta=1.0).mean(dim=1)
        else:
            per = F.mse_loss(pred, target, reduction="none").mean(dim=1)
        return (per * weight).mean()

    best_state: Optional[Dict[str, torch.Tensor]] = None
    bad = 0
    bs = max(1, min(cfg.batch_size, n))
    rng = np.random.default_rng(seed)

    for epoch in range(cfg.epochs):
        model.train()
        perm = rng.permutation(n)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i: i + bs]
            xb = xtr[idx]
            yb = ytr[idx]
            wb = w_t[idx]
            fb = ftr[idx] if ftr is not None else None

            # 输入加噪：一种便宜的数据增强，能让模型对单日异常值不那么敏感
            if cfg.input_noise > 0:
                xb = xb + torch.randn_like(xb) * cfg.input_noise

            opt.zero_grad(set_to_none=True)
            pred = model(xb, fb) if fb is not None else model(xb)
            loss = _loss(pred, yb, wb)
            loss.backward()
            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            tot += float(loss.detach()) * len(idx)
        sched.step()
        hist.train_loss.append(tot / n)

        model.eval()
        with torch.no_grad():
            vp = model(xva, fva) if fva is not None else model(xva)
            vloss = float(_loss(vp, yva, torch.ones(len(xva), device=device)))
        hist.val_loss.append(vloss)

        if vloss < hist.best_val - 1e-7:
            hist.best_val = vloss
            hist.best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break

        if verbose and (epoch % 50 == 0 or epoch == cfg.epochs - 1):
            print(f"    epoch {epoch:3d}  train={hist.train_loss[-1]:.4f}  val={vloss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, hist


# ---------------------------------------------------------------------------
# 推理
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict_array(
    model: nn.Module,
    X: np.ndarray,
    device: str = "cpu",
    batch: int = 512,
    fut: Optional[np.ndarray] = None,
) -> np.ndarray:
    """对已标准化的特征矩阵做批量推理，返回 (n, out_len) 或 (n, H) 的标准化预测值。"""
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(np.ascontiguousarray(X[i: i + batch])).to(device)
        fb = None
        if fut is not None:
            fb = torch.from_numpy(np.ascontiguousarray(fut[i: i + batch])).to(device)
        out.append((model(xb, fb) if fb is not None else model(xb)).cpu().numpy())
    if not out:  # 理论上不会发生，兜底避免 np.concatenate 报错
        return np.zeros((0, getattr(model, "out_len", 1)), dtype=np.float32)
    return np.concatenate(out, axis=0)


def predict_ensemble(
    models,
    X: np.ndarray,
    y_mu: float,
    y_sd: float,
    device: str = "cpu",
    fut: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    集成预测：各成员输出取平均，再反标准化。

    小数据场景下"多训几个种子再平均"通常比调结构更划算：方差降下来，
    回测的分月指标会明显更稳（均值改善不一定大，但最差那一折会好很多）。
    """
    if len(models) == 0:
        raise ValueError("models 为空")
    preds = [predict_array(m, X, device, fut=fut) for m in models]
    p = np.mean(preds, axis=0) * y_sd + y_mu
    return p
