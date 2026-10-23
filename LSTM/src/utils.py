"""
零碎工具：随机种子、目录、日志、matplotlib 中文字体、JSON 读写。
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from typing import Any, Dict

import numpy as np


# ---------------------------------------------------------------------------
# 随机性控制
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """把 python / numpy / torch 的随机种子全部固定，保证同一份配置可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # 关闭 cudnn 自动调优，换取可复现性（CPU 上无影响）
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:  # pragma: no cover
        pass


def pick_device(name: str = "auto") -> str:
    """选择计算设备。默认 auto：有 CUDA 用 CUDA，否则 CPU。"""
    import torch

    if name != "auto":
        return name
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# 目录 / 日志
# ---------------------------------------------------------------------------
def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def log(msg: str) -> None:
    """带时间戳的打印，方便看每一步耗时。"""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def save_json(obj: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# matplotlib 中文字体
# ---------------------------------------------------------------------------
def setup_matplotlib(enable: bool = True) -> bool:
    """
    配置 matplotlib：无界面后端 + 中文字体。

    返回：是否找到了中文字体。找不到时调用方应回退到英文标签，
    否则图上会出现一排"豆腐块"方框。
    """
    if not enable:
        return False
    import matplotlib

    matplotlib.use("Agg")  # 不弹窗，直接存 png；VS Code 里也能正常运行
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    candidates = [
        "Microsoft YaHei", "SimHei", "Microsoft JhengHei", "PingFang SC",
        "Hiragino Sans GB", "Noto Sans CJK SC", "Source Han Sans SC",
        "WenQuanYi Micro Hei", "Arial Unicode MS",
    ]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    found = [c for c in candidates if c in installed]

    if found:
        plt.rcParams["font.sans-serif"] = found + ["DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        return True

    # Windows 上偶尔 fontManager 缓存没刷新，这里兜底直接检查系统字体文件
    if sys.platform.startswith("win"):
        for name in ("msyh.ttc", "simhei.ttf"):
            p = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", name)
            if os.path.exists(p):
                try:
                    font_manager.fontManager.addfont(p)
                    prop = font_manager.FontProperties(fname=p)
                    plt.rcParams["font.sans-serif"] = [prop.get_name(), "DejaVu Sans"]
                    plt.rcParams["axes.unicode_minus"] = False
                    return True
                except Exception:
                    pass

    plt.rcParams["axes.unicode_minus"] = False
    return False
