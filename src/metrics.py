import numpy as np


def residuals(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """返回残差 y_true - y_pred，展平为一维"""
    return (y_true - y_pred).ravel()


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """均方误差"""
    return float(np.mean(residuals(y_true, y_pred) ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """均方根误差"""
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """平均绝对误差"""
    return float(np.mean(np.abs(residuals(y_true, y_pred))))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    R² 决定系数。
    R² = 1 - SS_res / SS_tot
    """
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    if ss_tot == 0:
        return 1.0  # y_true 为常数
    
    return float(1 - ss_res / ss_tot)


def adjusted_r2_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_features: int,
    fit_intercept: bool = True,
) -> float:
    """
    调整 R²。
    
    Parameters
    ----------
    n_features : int
        特征数 p（问题 1 中 p=99）
    fit_intercept : bool
        是否包含偏置项。若 True，p 为特征数；若 False，p 为参数总数。
        公式中分母为 n - p - 1，其中 +1 对应偏置项。
    """
    y_true = y_true.ravel()
    n_samples = y_true.shape[0]
    
    r2 = r2_score(y_true, y_pred)
    
    # p 是模型参数个数（不含偏置时 p=n_features，含偏置时 p=n_features+1）
    # 但公式中的 p 通常指特征数，+1 单独对应 intercept
    # 按题目：adjusted_r2 = 1 - (1 - R²) * (n - 1) / (n - p - 1)
    # 其中 p 是特征数（99），注意分母的 -1 是 intercept 的惩罚
    
    p = n_features  # 特征数
    
    if fit_intercept:
        denominator = n_samples - p - 1
    else:
        denominator = n_samples - p
    
    if denominator <= 0:
        return float("nan")
    
    return float(1 - (1 - r2) * (n_samples - 1) / denominator)