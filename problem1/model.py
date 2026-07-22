"""problem1 建模层：最小二乘拟合与指标计算。"""
import numpy as np


def make_design_matrix(A):
    """构造增广设计矩阵 X = [A, 1]。"""
    n = A.shape[0]
    return np.hstack([A, np.ones((n, 1))])


def fit_ols(A, B):
    """
    对 B = A @ w + b + ε 做最小二乘估计。

    Parameters
    ----------
    A : ndarray, shape (n, 99)
    B : ndarray, shape (n, 1) 或 (n,)

    Returns
    -------
    beta : ndarray, shape (100,)
        beta[:99] 为 w，beta[99] 为 b。
    """
    X = make_design_matrix(A)
    y = B.reshape(-1)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def predict(A, beta):
    """使用 beta 预测 B。"""
    X = make_design_matrix(A)
    return X @ beta


def compute_metrics(B, y_pred, n_features):
    """
    计算拟合指标。

    Parameters
    ----------
    B : ndarray, shape (n, 1) 或 (n,)
    y_pred : ndarray, shape (n,)
    n_features : int
        原始特征数 99，用于计算调整 R²。

    Returns
    -------
    dict
    """
    y = B.reshape(-1)
    n = y.shape[0]
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)

    r2 = 1.0 - ss_res / ss_tot
    adj_r2 = 1.0 - (ss_res / (n - n_features - 1)) / (ss_tot / (n - 1))
    mse = ss_res / n
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y - y_pred))

    return {
        "r2": r2,
        "adj_r2": adj_r2,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "bias": y_pred.mean() - y.mean(),  # 拟合值均值与真实均值之差，接近 0
    }


def fit_metrics(A, B):
    """一次性拟合并返回 beta 与指标。"""
    beta = fit_ols(A, B)
    y_pred = predict(A, beta)
    metrics = compute_metrics(B, y_pred, n_features=A.shape[1])
    return beta, y_pred, metrics
