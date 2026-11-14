"""
viz_utils.py —— 绘图公共工具：中文字体、浅色主题配色、滑动平均。
"""
from __future__ import annotations

import numpy as np
from matplotlib import font_manager as fm
import matplotlib.pyplot as plt

# 浅色主题（与 IDE light theme 一致）
BG = "#ffffff"
PANEL = "#f6f8fc"
GRID = "#dde3ed"
TEXT = "#1f2937"
MUTED = "#6b7280"
BLUE = "#2563eb"
RED = "#dc2626"
GREEN = "#059669"
ORANGE = "#ea580c"
PURPLE = "#7c3aed"
TEAL = "#0891b2"


def setup_font():
    """自动挑选可用的中文字体；找不到时退回 DejaVu Sans（英文标签仍可读）。"""
    available = {f.name for f in fm.fontManager.ttflist}
    for name in ("Microsoft YaHei", "Noto Sans SC", "SimHei", "DengXian", "SimSun"):
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            break
    else:
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = BG
    plt.rcParams["axes.facecolor"] = PANEL
    plt.rcParams["text.color"] = TEXT
    plt.rcParams["axes.labelcolor"] = TEXT
    plt.rcParams["xtick.color"] = MUTED
    plt.rcParams["ytick.color"] = MUTED


def moving_average(x, window: int = 50) -> np.ndarray:
    """居中/截断式滑动平均，长度与输入相同；前段用已有样本平均。"""
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return x
    out = np.empty_like(x)
    for i in range(len(x)):
        lo = max(0, i - window + 1)
        out[i] = x[lo:i + 1].mean()
    return out


def style_axis(ax, title: str, xlabel: str = "", ylabel: str = ""):
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, color=GRID, linewidth=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.tick_params(labelsize=8)
