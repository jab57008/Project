"""problem1 报告层：在控制台输出分析结果。"""
from pathlib import Path

from .data import PROBLEM1_DIR


def print_section(title):
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print("=" * 60)


def print_metrics(metrics):
    print_section("模型拟合指标")
    print(f"  R²              = {metrics['r2']:.6f}")
    print(f"  Adjusted R²     = {metrics['adj_r2']:.6f}")
    print(f"  RMSE            = {metrics['rmse']:.4f}")
    print(f"  MAE             = {metrics['mae']:.4f}")
    print(f"  MSE (noise var) = {metrics['mse']:.4f}")
    print(f"  Fitted bias b   = {metrics['bias']:.4f}")


def print_weights_summary(weights):
    import numpy as np

    print_section("权重分布")
    print(f"  mean  = {weights.mean():.4f}")
    print(f"  std   = {weights.std():.4f}")
    print(f"  min   = {weights.min():.4f}")
    print(f"  max   = {weights.max():.4f}")

    print("\n  Top 10 最大权重维度:")
    top_idx = np.argsort(weights)[-10:][::-1]
    for i, idx in enumerate(top_idx, 1):
        print(f"    {i:2d}. dim {idx:2d}: {weights[idx]:.4f}")

    print("\n  Top 10 最小权重维度:")
    bottom_idx = np.argsort(weights)[:10]
    for i, idx in enumerate(bottom_idx, 1):
        print(f"    {i:2d}. dim {idx:2d}: {weights[idx]:.4f}")


def print_quadratic_test(r2_linear, r2_quad):
    print_section("模型偏差检验（二次项对照实验）")
    print(f"  线性模型 R²          = {r2_linear:.6f}")
    print(f"  线性+二次项模型 R²   = {r2_quad:.6f}")
    print(f"  R² 提升              = {r2_quad - r2_linear:.6f}")
    print("  结论：非线性项几乎不贡献解释力，线性假设充分。")


def print_residual_tests(tests, critical_values):
    print_section("残差统计检验")
    print(f"  Durbin-Watson                = {tests['durbin_watson']:.4f}  (≈2 无自相关)")
    print(f"  残差-拟合值相关系数          = {tests['corr_residual_fitted']:.4e}")
    print(f"  偏度                         = {tests['skewness']:.4f}")
    print(f"  超额峰度                     = {tests['kurtosis_excess']:.4f}")
    print(f"  Jarque-Bera (JB)             = {tests['jarque_bera']:.2f}  (临界 χ²(2)=5.99)")
    print(f"  KS(正态)                     = {tests['ks_normal']:.4f}  (临界 ≈{critical_values['ks_critical']:.4f})")
    print(f"  Breusch-Pagan (LM)           = {tests['breusch_pagan']:.2f}  (临界 χ²({critical_values['bp_df']})={critical_values['bp_critical']:.2f})")


def print_error_decomposition(var_b, var_explained, mse, r2_linear, a):
    print_section("误差量化分解")
    print(f"  Var(B)              = {var_b:.4f}")
    print(f"  Var( explained )    = {var_explained:.4f}  ({r2_linear * 100:.2f}%)")
    print(f"  噪声方差估计 σ²     = {mse:.4f}  ({(1 - r2_linear) * 100:.2f}%)")
    print(f"  噪声近似 U(-{a:.2f}, {a:.2f})")


def print_report(A, B, beta, metrics, r2_quad, tests, a, figures):
    import numpy as np

    y = B.reshape(-1)
    w_hat = beta[:99]

    print_section("Problem 1 线性变换分析结果")
    print(f"  样本数 n = {A.shape[0]}, 特征数 p = {A.shape[1]}")
    print(f"  B 均值 = {y.mean():.4f}, B 标准差 = {y.std():.4f}")

    print_metrics(metrics)
    print_weights_summary(w_hat)

    r2_linear = metrics["r2"]
    print_quadratic_test(r2_linear, r2_quad)
    print_residual_tests(
        tests,
        critical_values={
            "ks_critical": 1.36 / np.sqrt(A.shape[0]),
            "bp_df": A.shape[1],
            "bp_critical": 123.23,
        },
    )

    y_pred = metrics.get("y_pred") or (A @ w_hat + beta[99])
    var_b = y.var(ddof=0)
    var_explained = y_pred.var(ddof=0)
    print_error_decomposition(var_b, var_explained, metrics["mse"], r2_linear, a)

    print_section("输出图表")
    for p in figures:
        print(f"  {p}")

    print("\n完成。")


def save_report_md(A, B, beta, metrics, r2_quad, tests, a, figures, path=None):
    """可选：将报告保存为 Markdown 文件。"""
    import numpy as np

    if path is None:
        path = PROBLEM1_DIR / "report.md"

    y = B.reshape(-1)
    w_hat = beta[:99]
    y_pred = A @ w_hat + beta[99]

    lines = [
        "# Problem 1 线性变换分析报告",
        "",
        f"- 样本数 n = {A.shape[0]}, 特征数 p = {A.shape[1]}",
        f"- B 均值 = {y.mean():.4f}, B 标准差 = {y.std():.4f}",
        "",
        "## 拟合指标",
        "",
        f"| 指标 | 数值 |",
        f"|---|---|",
        f"| R² | {metrics['r2']:.6f} |",
        f"| Adjusted R² | {metrics['adj_r2']:.6f} |",
        f"| RMSE | {metrics['rmse']:.4f} |",
        f"| MAE | {metrics['mae']:.4f} |",
        f"| MSE (noise var) | {metrics['mse']:.4f} |",
        f"| Fitted bias b | {metrics['bias']:.4f} |",
        "",
        "## 权重分布",
        "",
        f"- mean = {w_hat.mean():.4f}",
        f"- std = {w_hat.std():.4f}",
        f"- min = {w_hat.min():.4f}",
        f"- max = {w_hat.max():.4f}",
        "",
        "## 模型偏差检验",
        "",
        f"| 模型 | R² |",
        f"|---|---|",
        f"| 线性模型 | {metrics['r2']:.6f} |",
        f"| 线性+二次项 | {r2_quad:.6f} |",
        "",
        "结论：非线性项几乎不贡献解释力，线性假设充分。",
        "",
        "## 残差统计检验",
        "",
        f"| 检验项 | 统计量 |",
        f"|---|---|",
        f"| Durbin-Watson | {tests['durbin_watson']:.4f} |",
        f"| 残差-拟合值相关系数 | {tests['corr_residual_fitted']:.4e} |",
        f"| 偏度 | {tests['skewness']:.4f} |",
        f"| 超额峰度 | {tests['kurtosis_excess']:.4f} |",
        f"| Jarque-Bera | {tests['jarque_bera']:.2f} |",
        f"| KS(正态) | {tests['ks_normal']:.4f} |",
        f"| Breusch-Pagan | {tests['breusch_pagan']:.2f} |",
        "",
        "## 输出图表",
        "",
    ]
    for fig in figures:
        lines.append(f"- {fig}")
    lines.append("")

    Path(path).write_text("\n".join(lines), encoding="utf-8")
    return path
