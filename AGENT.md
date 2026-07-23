# Project Agent Guide

本文件汇总项目结构、核心方法、诊断检验与当前状态，供后续开发与审阅参考。

## 1. 项目概述

- **目标**：对问题 1 中给定的监测数据 `A`（10000×99）寻找线性变换，使其尽可能接近目标量 `B`（10000×1），并对误差来源进行系统诊断。
- **当前实现**：仅完成 `problem1` 的完整建模与诊断流程。`main.py` 可运行 `--problem 1`，问题 2–5 尚未实现。
- **核心结论**：线性假设充分（加入二次项 `R²` 提升约 `10⁻⁴`），噪声近似为 `U(-3.02, 3.02)` 的对称有界轻尾噪声，残差无自相关、基本同方差。

## 2. 目录与文件结构

```
.
├── main.py                     # 项目入口，目前仅支持 --problem 1
├── pyproject.toml              # uv 依赖：numpy, pandas, scipy, statsmodels, matplotlib, scikit-learn
├── uv.lock                     # uv 锁定文件
├── problem1/                   # 问题 1 数据与说明文档
│   ├── 附件1/                  # A.xlsx, B.xlsx
│   ├── 说明.md                 # 完整建模推导、假设、结果分析（保留详细论述）
│   ├── report.bk.md            # 旧版报告备份，含摘要、误差分解、图片引用
│   └── weights.npy             # 已保存的权重
├── outputs/problem1/           # 代码生成输出
│   ├── report.md               # 当前精简版报告（仅指标表与诊断表）
│   ├── weights_bar.png
│   ├── residual_hist.png
│   └── residual_fitted.png
└── src/                        # 源代码
    ├── data_loader.py          # Excel 数据读取
    ├── linear_model.py         # LinearRegression（最小二乘/SVD）
    ├── metrics.py              # R²、Adjusted R²、RMSE、MAE、MSE
    ├── diagnostics.py          # 残差诊断：DW、偏度、峰度、JB、BP、Koenker BP、KS、χ²
    ├── visualization.py        # 权重柱状图、残差直方图、残差-拟合值散点图
    ├── report.py               # 极简 Markdown 报告生成器（仅表格）
    └── problems/problem1.py    # 问题 1 完整流程编排
```

## 3. 运行方式

使用 **uv**（不要使用系统 Python）：

```bash
uv run python main.py --problem 1
```

可选参数：

```bash
uv run python main.py --problem 1 --data-dir problem1/附件1 --output-dir outputs/problem1
```

## 4. 核心方法

### 4.1 模型

线性模型：

```
B = A w + b·1 + ε
X = [A, 1]   (10000 × 100)
β = (wᵀ, b)ᵀ
```

最小二乘闭式解：

```
β̂ = (XᵀX)⁻¹XᵀB
```

实际实现使用 `numpy.linalg.lstsq`（基于 SVD），避免显式求逆，提高数值稳定性。

### 4.2 指标

| 指标 | 公式/说明 |
|---|---|
| R² | 1 − SS_res / SS_tot |
| Adjusted R² | 1 − (1 − R²)(n − 1)/(n − p − 1) |
| RMSE | √MSE |
| MAE | mean\|残差\| |
| MSE | 噪声方差估计 |

### 4.3 可视化

- `weights_bar.png`：99 维权重的柱状图，高亮 top/bottom 10。
- `residual_hist.png`：残差直方图，叠加正态与均匀分布密度。
- `residual_fitted.png`：残差 vs 拟合值散点图，用于直观判断异方差与模型偏差。

## 5. 诊断检验

所有诊断在 `src/diagnostics.py` 中实现，并在 `run_all_diagnostics` 中统一返回。

| 检验 | 用途 | 当前结果 | 判定 |
|---|---|---|---|
| Durbin-Watson | 残差序列独立性 | 1.9908 | ≈2，无自相关 |
| 残差-拟合值相关性 | OLS 正交性验证 | ≈0 | 符合理论 |
| 偏度 | 分布对称性 | 0.0010 | 对称 |
| 超额峰度 | 分布尾部特征 | −1.1803 | 接近均匀分布理论值 −1.2 |
| Jarque-Bera | 正态性 | 580.46 | 拒绝正态 |
| Breusch-Pagan LM | 原始同方差检验 | 130.51 | 临界拒绝 |
| **Koenker BP LM** | **稳健同方差检验** | **52.45** | **不拒绝同方差** |
| KS 对均匀分布 | 是否服从 U(−a, a) | 0.0100 | 不拒绝 |
| χ² 拟合优度 | 均匀分布严格检验 | 21.75 | 轻微拒绝严格均匀 |

### 5.1 Koenker 修正 BP 说明

- 实现位于 `src/diagnostics.py:koenker_breusch_pagan_test`。
- 使用无偏噪声方差估计 `σ² = Σε² / (n − p)`，将残差平方标准化后做辅助回归，取 `LM = ESS / 2`。
- 代码输出为 **52.45**，与 `problem1/说明.md` 中的 **52.2** 略有差异（约 0.25），推测为文档中的近似值或略有不同的实现。结论一致：均小于临界值 `χ²₀.₀₅(99) = 123.23`，不拒绝同方差。

## 6. 不精简但有意义的内容

以下部分对项目理解、可复现性与报告完整性有价值，建议保留：

### 6.1 `problem1/说明.md` 中的详细推导

- 五条基本假设的明确陈述与依据。
- 最小二乘闭式解的推导与高斯-马尔可夫定理引用。
- 为何不直接求逆、为何不用神经网络的方法论论证。
- 二次项对照实验：线性模型 `R² = 0.989127`，加入二次项后 `R² = 0.989235`，仅提升 `10⁻⁴`，支撑“线性假设充分”的结论。
- 误差分解公式：`Var(B) = Var(Aŵ + b̂) + σ̂²`。
- 噪声物理意义解读：矿山监测中的量化误差/传感器截断误差常近似均匀分布。

### 6.2 `problem1/report.bk.md` 中的结构化总结

- **摘要**：一段式概括模型、拟合优度与噪声类型。
- **误差分解**：文字说明模型解释方差比例与噪声方差比例。
- **图片引用**：直接链接 `weights_bar.png`、`residual_hist.png`、`residual_fitted.png`，便于阅读者对照图表。

> 注意：`outputs/problem1/report.md` 为当前精简版，已移除摘要、误差分解和图片引用；如需恢复完整报告，可调用 `ReportBuilder` 的已删除方法，或从 `report.bk.md` 恢复内容。

### 6.3 代码中的“保留项”

- `main.py` 的 `else` 分支：`print(f"Problem {args.problem} is not implemented yet.")` 虽未实现其他问题，但作为 CLI 占位信息有意义，保留。
- `src/problems/__init__.py` 与 `src/__init__.py`：空文件，保留包结构，无需删除。
- `problem1/weights.npy`：已保存的权重，可供后续问题复用或验证，保留。

## 7. 当前项目缺口与可扩展点

1. **二次项对照实验未在代码中实现**：`problem1/说明.md` 中提到的“线性 + 二次项模型”目前只在文档中，未在 `src/problems/problem1.py` 或 `src/diagnostics.py` 中编码。如需自动化报告，应补充二次特征构造与对照拟合。
2. **问题 2–5 未实现**：`main.py` 仅路由到 `problem1`，`src/problems/` 只有 `problem1.py`。
3. **报告生成器仅支持表格**：`ReportBuilder` 已精简为只渲染标题 + 表格。如果未来需要摘要、误差分解、图片，需要重新扩展或手动维护 Markdown。
4. **未保存模型/结果**：`weights.npy` 已存在，但 `run_problem1` 不自动保存权重为 `.npy`；如需要持久化，应补充保存逻辑。
5. **无单元测试**：项目缺少 `tests/` 目录，核心指标与诊断函数未做自动化断言验证。

## 8. 重要约定

- **使用 uv 运行**：所有 Python 命令应通过 `uv run python ...` 执行，避免使用系统 Python。
- **数据路径**：默认 `problem1/附件1`，可在 `main.py` 通过 `--data-dir` 覆盖。
- **输出路径**：默认 `outputs/problem1`，可在 `main.py` 通过 `--output-dir` 覆盖。
- **Markdown 报告**：`outputs/problem1/report.md` 由代码自动生成，不建议手动修改；如需手动润色，请保留 `report.bk.md` 作为参考。
- **说明.md 数值校对**：说明.md 中 Koenker BP 统计量写为 52.2，代码实际输出 52.45。建议后续统一说明为 52.45，或注明为近似值。

## 9. 关键结果速查

| 项目 | 数值 |
|---|---|
| R² | 0.9891 |
| Adjusted R² | 0.9890 |
| RMSE | 1.7434 |
| MAE | 1.5087 |
| MSE（噪声方差估计） | 3.0395 |
| 偏置 b | 151.6440 |
| 权重均值 / 标准差 | 5.1024 / 2.8253 |
| 噪声半宽 a | 3.0197 |
| 越界比例 | 0.0188（1.88%） |
| 原始 BP LM | 130.51 |
| Koenker BP LM | 52.45 |
