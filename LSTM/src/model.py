"""
LSTM 网络结构。

输入  : (batch, lookback, n_features)  过去 28 天的逐日特征
输出  : (batch, out_len)               未来 out_len 天的季节指数 ratio
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .config import Config


class LSTMForecaster(nn.Module):
    """
    单目标的多步预测 LSTM，支持两种输出头。

    结构：
        LSTM(输入 -> hidden) 取最后时刻隐状态 enc
        multistep: Linear(enc) -> out_len 个数
        future   : 把 enc 复制 H 份，和"未来第 h 天的日历特征 + h/H"拼起来，
                   过一个共享的小 MLP，逐天输出 1 个数

    为什么默认用 future 头：
        未来 30 天里"哪天是周一、哪天是中秋"是已知的，这是这份数据里最强的信号。
        multistep 头要求网络自己从 28 天窗口里推断"输出第 3 个数对应周三"，
        在只有 300 个样本的情况下很难学准；future 头把这件事显式喂给模型，
        参数还更少（共享 MLP），小数据下明显更稳。

    为什么这么浅：数据只有 420 天，LSTM 参数一多就过拟合。
    先把正则化、早停、集成做好，比堆深网络有效得多。
    """

    def __init__(
        self,
        n_features: int,
        hidden: int = 48,
        layers: int = 1,
        dropout: float = 0.1,
        out_len: int = 31,
        n_future: int = 0,
    ):
        super().__init__()
        self.n_features = n_features
        self.hidden = hidden
        self.layers = layers
        self.out_len = out_len
        self.n_future = n_future          # >0 表示用 future 头
        self.use_future = n_future > 0

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        mid = max(hidden // 2, 8)

        if self.use_future:
            self.decoder = nn.Sequential(
                nn.Linear(hidden + n_future, mid),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(mid, 1),
            )
            self.head = None
        else:
            self.decoder = None
            self.head = nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, mid),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(mid, out_len),
            )
        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier/正交初始化 + 遗忘门 bias 置 1（小数据上收敛更快更稳）。"""
        for name, p in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(p)
            elif "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

        with torch.no_grad():
            for layer in range(self.layers):
                # PyTorch 的 LSTM 每个方向有 bias_ih / bias_hh 两组，各长 4*hidden，
                # 排布是 [i, f, g, o]。遗忘门（f）合计 bias 设为 1.0：
                # 这是小数据上很常用的技巧，能让模型"默认记住"，缓解梯度消失。
                b_ih = getattr(self.lstm, f"bias_ih_l{layer}")
                b_hh = getattr(self.lstm, f"bias_hh_l{layer}")
                b_ih[self.hidden: 2 * self.hidden].fill_(0.0)
                b_hh[self.hidden: 2 * self.hidden].fill_(1.0)

    def forward(self, x: torch.Tensor, fut: torch.Tensor = None) -> torch.Tensor:
        """
        x   : (B, lookback, n_features)
        fut : (B, H, n_future)，只有在 future 头时传
        返回: (B, out_len) 或 (B, H)
        """
        out, _ = self.lstm(x)          # (B, L, hidden)
        enc = out[:, -1, :]            # 取最后一个时间步

        if not self.use_future:
            return self.head(enc)      # (B, out_len)

        if fut is None:
            raise ValueError("future 头必须提供未来协变量 fut")
        b, h, c = fut.shape
        z = torch.cat([enc.unsqueeze(1).expand(b, h, -1), fut], dim=-1)  # (B, H, hidden+C)
        return self.decoder(z).squeeze(-1)                              # (B, H)


def build_model(cfg: Config, n_features: int, out_len: Optional[int] = None,
                n_future: int = 0) -> LSTMForecaster:
    """根据配置构造模型。out_len / n_future 由调用方按输出头结构传入。"""
    return LSTMForecaster(
        n_features=n_features,
        hidden=cfg.hidden,
        layers=cfg.layers,
        dropout=cfg.dropout,
        out_len=cfg.out_len if out_len is None else int(out_len),
        n_future=n_future,
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
