"""problem1 数据层：加载或生成问题 1 所需的 A、B 数据。"""
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROBLEM1_DIR = PROJECT_ROOT / "problem1"
DATA_DIR = PROJECT_ROOT / "data"
WEIGHTS_PATH = PROBLEM1_DIR / "weights.npy"

RANDOM_SEED = 42
NOISE_HALF_WIDTH = 3.02  # 与说明中 U(-3, 3) 的近似一致


def load_true_weights():
    """加载 weights.npy，返回 (w, b)，其中 w 形状为 (99,)，b 为标量。"""
    beta = np.load(WEIGHTS_PATH)
    if beta.shape != (100,):
        raise ValueError(f"weights.npy 预期形状为 (100,)，实际为 {beta.shape}")
    w = beta[:99]
    b = beta[99]
    return w, b


def _generate_synthetic_data(n_samples=10_000, seed=RANDOM_SEED):
    """使用 weights.npy 中保存的参数生成合成数据。"""
    rng = np.random.default_rng(seed)
    w, b = load_true_weights()

    A = rng.uniform(0.0, 1.0, size=(n_samples, 99))
    epsilon = rng.uniform(-NOISE_HALF_WIDTH, NOISE_HALF_WIDTH, size=n_samples)
    B = A @ w + b + epsilon
    return A, B.reshape(-1, 1), w, b


def load_problem1_data(n_samples=10_000, seed=RANDOM_SEED):
    """
    加载问题 1 的数据。

    如果 data/A.npy 与 data/B.npy 存在，则直接加载；
    否则使用 weights.npy 生成合成数据，用于演示和验证。

    Returns
    -------
    A : ndarray, shape (n_samples, 99)
    B : ndarray, shape (n_samples, 1)
    true_weights : tuple (w, b)
    """
    a_path = DATA_DIR / "A.npy"
    b_path = DATA_DIR / "B.npy"
    if a_path.exists() and b_path.exists():
        A = np.load(a_path)
        B = np.load(b_path)
        if B.ndim == 1:
            B = B.reshape(-1, 1)
        true_weights = load_true_weights()
        return A, B, true_weights

    return _generate_synthetic_data(n_samples, seed)


if __name__ == "__main__":
    A, B, (w, b) = load_problem1_data()
    print(f"A shape: {A.shape}, B shape: {B.shape}")
    print(f"B mean={B.mean():.4f}, std={B.std():.4f}")
    print(f"true w mean={w.mean():.4f}, std={w.std():.4f}")
    print(f"true b={b:.4f}")
