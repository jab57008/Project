"""problem1 可视化层：生成说明中要求的图表。"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .data import PROBLEM1_DIR


FIGURES_DIR = PROBLEM1_DIR / "figures"


def ensure_figures_dir():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def plot_weights(weights, save_path=None):
    """权重向量柱状图，突出前 10 最大与最小维度。"""
    ensure_figures_dir()
    fig, ax = plt.subplots(figsize=(14, 5))
    n = len(weights)
    colors = ["#4c78a8"] * n

    top10 = set(np.argpartition(weights, -10)[-10:])
    bottom10 = set(np.argpartition(weights, 10)[:10])
    for i in range(n):
        if i in top10:
            colors[i] = "#e45756"
        elif i in bottom10:
            colors[i] = "#f58518"

    ax.bar(range(n), weights, color=colors, edgecolor="none")
    ax.set_xlabel("Dimension index")
    ax.set_ylabel("Weight")
    ax.set_title(r"Estimated weights $\hat{w}$ (top/bottom 10 highlighted)")
    ax.set_xlim(-1, n)
    fig.tight_layout()

    if save_path is None:
        save_path = FIGURES_DIR / "weights.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_residual_histogram(residuals, a, save_path=None):
    """残差直方图，叠加均匀分布与正态分布密度。"""
    ensure_figures_dir()
    fig, ax = plt.subplots(figsize=(8, 5))

    n, bins, patches = ax.hist(residuals, bins=40, density=True, color="#4c78a8", alpha=0.7, edgecolor="white")

    x = np.linspace(residuals.min(), residuals.max(), 500)
    # 均匀分布密度 U(-a, a)
    uniform_density = np.where((x >= -a) & (x <= a), 1.0 / (2 * a), 0.0)
    # 正态分布密度（与残差同均值同标准差）
    sigma = residuals.std(ddof=1)
    normal_density = (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - residuals.mean()) / sigma) ** 2)

    ax.plot(x, uniform_density, color="#e45756", lw=2, label=f"Uniform $U(-{a:.2f}, {a:.2f})$")
    ax.plot(x, normal_density, color="#54a24b", lw=2, linestyle="--", label="Normal (same mean, std)")
    ax.set_xlabel("Residual")
    ax.set_ylabel("Density")
    ax.set_title("Residual histogram with uniform and normal densities")
    ax.legend()
    fig.tight_layout()

    if save_path is None:
        save_path = FIGURES_DIR / "residual_histogram.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_residual_vs_fitted(fitted, residuals, save_path=None):
    """残差-拟合值散点图。"""
    ensure_figures_dir()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(fitted, residuals, s=8, alpha=0.4, color="#4c78a8")
    ax.axhline(0, color="#e45756", linestyle="--", lw=1.5)
    ax.set_xlabel("Fitted value")
    ax.set_ylabel("Residual")
    ax.set_title("Residual vs fitted value")
    fig.tight_layout()

    if save_path is None:
        save_path = FIGURES_DIR / "residual_vs_fitted.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def generate_all_plots(weights, residuals, fitted, a):
    """生成所有图表并返回保存路径列表。"""
    paths = []
    paths.append(plot_weights(weights))
    paths.append(plot_residual_histogram(residuals, a))
    paths.append(plot_residual_vs_fitted(fitted, residuals))
    return paths
