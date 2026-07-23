from pathlib import Path
import numpy as np
import pandas as pd

from src import metrics
from src.report import ReportBuilder


def solve_problem2(
    data_dir: Path | str = "problem2/附件2(Attachment 2)",
    output_dir: Path | str = "outputs/problem2",
) -> dict:
    """问题 2：PCA 降维压缩与还原。在 MSE ≤ 0.005 的前提下最大化压缩效率。"""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载
    X = pd.read_excel(data_dir / "Data.xlsx", header=None).values.astype(np.float64)
    n, p = X.shape

    # 2. 中心化
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu

    # 3. 一次 SVD（耗时几分钟），之后全部从 s/Vt 推导
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)

    # 4. 用 Eckart-Young 从奇异值直接推出每个 k 的 MSE
    s_sq = s ** 2
    total_var = s_sq.sum()
    cum = np.cumsum(s_sq)
    records = []
    for k in range(1, len(s) + 1):
        mse = float((total_var - cum[k - 1]) / (n * p))
        comp_ratio = (n * p) / (n * k + p * k + p)
        storage_save = (1 - 1 / comp_ratio) * 100
        var_exp = float(cum[k - 1] / total_var * 100)
        records.append({"k": k, "var_explained": var_exp, "comp_ratio": comp_ratio,
                        "storage_save": storage_save, "mse": mse})
        if mse <= 0.005:
            break

    # 5. 最优 k 下的还原
    k_star = records[-1]["k"]
    Z_star = U[:, :k_star] * s[:k_star]
    V_star = Vt[:k_star, :].T
    X_hat = Z_star @ V_star.T + mu
    residuals = (X - X_hat).ravel()
    mse_final = records[-1]["mse"]
    col_r2 = np.array([metrics.r2_score(X[:, j], X_hat[:, j]) for j in range(p)])

    # 6. 图表 + 报告
    _make_plots(s, total_var, records, residuals, mse_final, col_r2, k_star, output_dir)
    _generate_report(records, k_star, mse_final, float(np.sqrt(mse_final)),
                     float(np.mean(np.abs(residuals))), col_r2, residuals, output_dir)

    return {"k_star": k_star, "mse": mse_final,
            "comp_ratio": records[-1]["comp_ratio"],
            "storage_save": records[-1]["storage_save"],
            "records": records, "output_dir": output_dir}


def _make_plots(s, total_var, records, residuals, mse_final, col_r2, k_star, out):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    ks = [r["k"] for r in records]
    mses = [r["mse"] for r in records]
    crs = [r["comp_ratio"] for r in records]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(1, 16), s[:15]**2/total_var*100, color="steelblue")
    ax.axvline(k_star, color="red", linestyle="--")
    ax.set_xlabel("主成分序号"); ax.set_ylabel("方差解释比(%)")
    ax.set_title("碎石图"); fig.tight_layout(); fig.savefig(out/"scree.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ks, mses, "o-", color="steelblue")
    ax.axhline(0.005, color="red", linestyle="--"); ax.axvline(k_star, color="green", linestyle="--")
    ax.set_xlabel("k"); ax.set_ylabel("MSE"); ax.set_title("MSE vs k")
    fig.tight_layout(); fig.savefig(out/"mse_vs_k.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ks, crs, "s-", color="darkorange")
    ax.axvline(k_star, color="green", linestyle="--")
    ax.set_xlabel("k"); ax.set_ylabel("压缩比"); ax.set_title("压缩比 vs k")
    fig.tight_layout(); fig.savefig(out/"cr_vs_k.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(residuals, bins=60, density=True, alpha=0.7, color="gray", edgecolor="black")
    ax.axvline(0, color="red", linestyle="--")
    ax.set_xlabel("还原误差"); ax.set_title(f"残差分布 k={k_star} MSE={mse_final:.6f}")
    fig.tight_layout(); fig.savefig(out/"residual_hist.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(1, len(col_r2)+1), col_r2, color="steelblue")
    ax.axhline(np.mean(col_r2), color="red", linestyle="--")
    ax.set_xlabel("列序号"); ax.set_ylabel("R²"); ax.set_title("逐列还原 R²")
    fig.tight_layout(); fig.savefig(out/"col_r2.png", dpi=150); plt.close(fig)
    plt.close("all")


def _generate_report(records, k_star, mse, rmse, mae, col_r2, residuals, out):
    report = ReportBuilder(title="Problem 2: 数据降维压缩与还原")
    report.add_table("k 值扫描结果", ["k", "方差解释比(%)", "压缩比", "存储节省率(%)", "MSE"],
                     [[f"{r['k']}", f"{r['var_explained']:.2f}", f"{r['comp_ratio']:.2f}",
                       f"{r['storage_save']:.1f}", f"{r['mse']:.6f}"] for r in records])
    report.add_table("最优方案", ["指标", "数值"],
                     [["k*", str(k_star)],
                      ["方差解释比", f"{records[-1]['var_explained']:.2f}%"],
                      ["压缩比", f"{records[-1]['comp_ratio']:.2f}"],
                      ["存储节省率", f"{records[-1]['storage_save']:.1f}%"],
                      ["MSE", f"{mse:.6f} ≤ 0.005"],
                      ["RMSE", f"{rmse:.6f}"], ["MAE", f"{mae:.6f}"]])
    report.add_table("误差统计", ["统计量", "数值"],
                     [["均值", f"{np.mean(residuals):.6f}"],
                      ["标准差", f"{np.std(residuals, ddof=1):.6f}"],
                      ["偏度", f"{pd.Series(residuals).skew():.6f}"],
                      ["峰度", f"{pd.Series(residuals).kurtosis():.6f}"],
                      ["最小值", f"{np.min(residuals):.6f}"],
                      ["最大值", f"{np.max(residuals):.6f}"]])
    report.add_table("逐列 R²", ["统计量", "数值"],
                     [["最大值", f"{np.max(col_r2):.6f}"],
                      ["最小值", f"{np.min(col_r2):.6f}"],
                      ["均值", f"{np.mean(col_r2):.6f}"],
                      ["标准差", f"{np.std(col_r2, ddof=1):.6f}"]])
    report.save(out / "report.md")


if __name__ == "__main__":
    r = solve_problem2()
    print(f"k*={r['k_star']}  MSE={r['mse']:.6f}  CR={r['comp_ratio']:.2f}")
    print(f"Output: {r['output_dir']}")