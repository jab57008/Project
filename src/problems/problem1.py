from pathlib import Path
import numpy as np

from src.data_loader import load_problem1_data
from src.linear_model import LinearRegression
from src import metrics
from src import diagnostics
from src import visualization
from src.report import ReportBuilder


def run_problem1(
    data_dir: Path | str = "problem1/附件1",
    output_dir: Path | str = "outputs/problem1",
) -> dict:
    """
    问题 1 完整分析流程：A → B 的线性变换建模、误差诊断与报告生成。
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    # 加载数据
    A, B = load_problem1_data(data_dir)
    y = B.ravel()
    n_samples, n_features = A.shape

    # 拟合
    model = LinearRegression(fit_intercept=True)
    model.fit(A, y)
    assert model.coef_ is not None  # 断言了fit后coef_一定不为None，以应对pylance类型检查
    y_pred = model.predict(A)
    resid = metrics.residuals(y, y_pred)

    # 3. 指标
    mse = metrics.mse(y, y_pred)
    rmse = metrics.rmse(y, y_pred)
    mae = metrics.mae(y, y_pred)
    r2 = metrics.r2_score(y, y_pred)
    adj_r2 = metrics.adjusted_r2_score(
        y, y_pred, n_features=n_features, fit_intercept=True
    )

    # 4. 诊断
    diag = diagnostics.run_all_diagnostics(resid, y_pred, A, mse)

    # 5. 绘图
    visualization.plot_weights_bar(model.coef_, output_dir / "weights_bar.png")
    visualization.plot_residual_histogram(
        resid, output_dir / "residual_hist.png", uniform_a=diag["noise_half_width"]
    )
    visualization.plot_residual_vs_fitted(
        y_pred, resid, output_dir / "residual_fitted.png"
    )

    # 6. 生成报告
    report = ReportBuilder(title="Problem 1: Linear Transformation from A to B")

    report.add_table(
        "拟合结果",
        ["指标", "数值"],
        [
            ["R²", str(r2)],
            ["Adjusted R²", str(adj_r2)],
            ["RMSE", str(rmse)],
            ["MAE", str(mae)],
            ["MSE (噪声方差估计)", str(mse)],
            ["偏置 b", str(model.intercept_)],
            ["权重均值", str(float(np.mean(model.coef_)))],
            ["权重标准差", str(float(np.std(model.coef_, ddof=1)))],
            ["权重最小值", str(float(np.min(model.coef_)))],
            ["权重最大值", str(float(np.max(model.coef_)))],
        ],
    )

    report.add_table(
        "残差诊断",
        ["检验项", "统计量", "p 值 / 备注"],
        [
            ["Durbin-Watson", diag["durbin_watson"], "≈2 无自相关"],
            ["残差-拟合值相关性", diag["correlation_with_fitted"], "OLS 理论 ≈0"],
            ["偏度", diag["skewness"], "≈0 对称"],
            ["超额峰度", diag["kurtosis"], "正态 0，均匀 -1.2"],
            ["Jarque-Bera", diag["jarque_bera"]["statistic"], diag["jarque_bera"]["p_value"]],
            ["Breusch-Pagan LM", diag["breusch_pagan"]["lm_statistic"], diag["breusch_pagan"]["lm_pvalue"]],
            ["Koenker BP LM", diag["koenker_breusch_pagan"]["lm_statistic"], diag["koenker_breusch_pagan"]["lm_pvalue"]],
            ["KS 对均匀分布", diag["ks_uniform"]["statistic"], diag["ks_uniform"]["p_value"]],
            ["卡方拟合优度（均匀）", diag["chi2_uniform"]["statistic"], f"df={diag['chi2_uniform']['df']}"],
            ["噪声半宽估计", diag["noise_half_width"], f"U(-a, a)"],
            ["越界比例", diag["outlier_ratio"], f"outside (-a, a)"],
        ],
    )

    report.save(output_dir / "report.md")

    return {
        "model": model,
        "A": A,
        "B": B,
        "predicted": y_pred,
        "residuals": resid,
        "metrics": {
            "r2": r2,
            "adjusted_r2": adj_r2,
            "rmse": rmse,
            "mae": mae,
            "mse": mse,
        },
        "diagnostics": diag,
        "output_dir": output_dir,
    }


if __name__ == "__main__":
    result = run_problem1()
    print("Problem 1 completed.")
    print("Metrics:")
    for k, v in result["metrics"].items():
        print(f"  {k}: {v:.6f}")
    print(f"Output saved to: {result['output_dir']}")