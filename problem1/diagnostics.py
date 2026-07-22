"""problem1 诊断层：误差来源分析与统计检验。"""
import numpy as np

from .model import fit_ols, predict, make_design_matrix


def quadratic_test(A, B):
    """
    加入 A 的二次项 A**2，拟合扩展模型并返回 R²。
    用于判断线性假设是否充分。
    """
    X_quad = np.hstack([A, A ** 2, np.ones((A.shape[0], 1))])
    y = B.reshape(-1)
    beta_quad, *_ = np.linalg.lstsq(X_quad, y, rcond=None)
    y_pred_quad = X_quad @ beta_quad
    ss_res = np.sum((y - y_pred_quad) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1.0 - ss_res / ss_tot


def durbin_watson(residuals):
    """Durbin-Watson 统计量，≈2 表示无自相关。"""
    diff = np.diff(residuals)
    return np.sum(diff ** 2) / np.sum(residuals ** 2)


def skewness(x):
    """样本偏度。"""
    n = x.shape[0]
    mu = x.mean()
    m2 = np.mean((x - mu) ** 2)
    m3 = np.mean((x - mu) ** 3)
    # 无偏调整
    return np.sqrt(n * (n - 1)) / (n - 2) * m3 / (m2 ** 1.5)


def kurtosis_excess(x):
    """样本超额峰度。"""
    n = x.shape[0]
    mu = x.mean()
    m2 = np.mean((x - mu) ** 2)
    m4 = np.mean((x - mu) ** 4)
    # G2 无偏调整
    g2 = m4 / (m2 ** 2) - 3.0
    return ((n - 1) * ((n + 1) * g2 + 6)) / ((n - 2) * (n - 3))


def jarque_bera(residuals):
    """Jarque-Bera 正态性检验。"""
    n = residuals.shape[0]
    s = skewness(residuals)
    k = kurtosis_excess(residuals)
    jb = n / 6.0 * (s ** 2 + k ** 2 / 4)
    return jb


def ks_test_normal(residuals):
    """
     Kolmogorov-Smirnov 正态性检验：与标准差相同、均值相同的正态分布比较。
    返回 KS 统计量。
    """
    mu = residuals.mean()
    sigma = residuals.std(ddof=1)
    x_sorted = np.sort(residuals)
    n = x_sorted.shape[0]
    cdf_emp = np.arange(1, n + 1) / n
    # 正态分布 CDF
    from math import erf

    def norm_cdf(x):
        return 0.5 * (1.0 + erf((x - mu) / (sigma * np.sqrt(2.0))))

    cdf_theo = np.array([norm_cdf(x) for x in x_sorted])
    # 考虑左连续与右连续
    ks = np.max(np.abs(cdf_emp - cdf_theo))
    ks = max(ks, np.max(np.abs(np.arange(0, n) / n - cdf_theo)))
    return ks


def breusch_pagan(residuals, X):
    """
    Breusch-Pagan 同方差检验（LM 形式）。
    用残差平方对 X 做辅助回归，LM = n * R²。
    """
    n = X.shape[0]
    e2 = residuals ** 2
    e2_mean = e2.mean()
    # 辅助回归
    gamma, *_ = np.linalg.lstsq(X, e2, rcond=None)
    e2_pred = X @ gamma
    ss_res = np.sum((e2 - e2_pred) ** 2)
    ss_tot = np.sum((e2 - e2_mean) ** 2)
    r2_aux = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    lm = n * r2_aux
    return lm


def uniform_fit_test(residuals, a):
    """
    对 U(-a, a) 做卡方拟合优度检验。
    返回卡方统计量与自由度。
    """
    n = residuals.shape[0]
    # 使用 10 个等宽区间覆盖 [-a, a]
    k = 10
    bin_edges = np.linspace(-a, a, k + 1)
    observed, _ = np.histogram(residuals, bins=bin_edges)
    expected = n / k
    # 合并期望频数小于 5 的尾部区间（简单处理：合并到第一个/最后一个）
    chi2 = np.sum((observed - expected) ** 2 / expected)
    return chi2, k - 1


def residual_tests(residuals, fitted, X):
    """汇总所有残差诊断检验。"""
    return {
        "durbin_watson": durbin_watson(residuals),
        "corr_residual_fitted": np.corrcoef(residuals, fitted)[0, 1],
        "skewness": skewness(residuals),
        "kurtosis_excess": kurtosis_excess(residuals),
        "jarque_bera": jarque_bera(residuals),
        "ks_normal": ks_test_normal(residuals),
        "breusch_pagan": breusch_pagan(residuals, X),
    }
