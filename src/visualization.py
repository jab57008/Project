import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy import stats


def _ensure_dir(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_weights_bar(coef: np.ndarray, output_path: Path, top_k: int = 10):
    """权重分布柱状图，高亮 top_k 最大/最小权重。"""
    output_path = _ensure_dir(output_path)
    coef = np.asarray(coef).ravel()
    n = len(coef)

    indices = np.arange(n)
    top_idx = np.argsort(coef)[-top_k:]
    bottom_idx = np.argsort(coef)[:top_k]

    colors = ["#1f77b4"] * n
    for i in top_idx:
        colors[i] = "#d62728"
    for i in bottom_idx:
        colors[i] = "#2ca02c"

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(indices, coef, color=colors, edgecolor="none")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Dimension Index")
    ax.set_ylabel("Weight")
    ax.set_title(f"Weight Distribution (Top/Bottom {top_k} Highlighted)")
    ax.set_xlim(-1, n)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_residual_histogram(
    residuals: np.ndarray,
    output_path: Path,
    uniform_a: float | None = None,
    bins: int = 50,
):
    """残差直方图，叠加正态与均匀分布密度。"""
    output_path = _ensure_dir(output_path)
    resid = np.asarray(residuals).ravel()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        resid,
        bins=bins,
        density=True,
        color="#1f77b4",
        alpha=0.6,
        edgecolor="white",
    )

    x = np.linspace(resid.min(), resid.max(), 500)

    std = np.std(resid, ddof=1)
    normal_pdf = stats.norm.pdf(x, loc=0, scale=std) if std > 0 else np.zeros_like(x)
    ax.plot(x, normal_pdf, "r--", linewidth=2, label="Normal(0, std²)")

    if uniform_a is not None:
        uniform_pdf = np.where(
            (x >= -uniform_a) & (x <= uniform_a),
            1 / (2 * uniform_a),
            0,
        )
        ax.plot(x, uniform_pdf, "g-", linewidth=2, label=f"U(-{uniform_a:.2f}, {uniform_a:.2f})")

    ax.set_xlabel("Residual")
    ax.set_ylabel("Density")
    ax.set_title("Residual Histogram with Distribution Overlay")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_residual_vs_fitted(fitted: np.ndarray, residuals: np.ndarray, output_path: Path):
    """残差-拟合值散点图。"""
    output_path = _ensure_dir(output_path)
    fitted = np.asarray(fitted).ravel()
    resid = np.asarray(residuals).ravel()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(fitted, resid, alpha=0.3, s=8, c="#1f77b4")
    ax.axhline(0, color="red", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Fitted Value")
    ax.set_ylabel("Residual")
    ax.set_title("Residual vs Fitted Value")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
