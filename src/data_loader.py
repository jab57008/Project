import pandas as pd
import numpy as np
from pathlib import Path


def load_excel(path: Path, sheet_name=0) -> np.ndarray:
    """读取 Excel 文件为 numpy 数组，默认无表头。"""
    df = pd.read_excel(path, sheet_name=sheet_name, header=None)
    return df.to_numpy(dtype=np.float64)


def load_problem1_data(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    读取问题 1 的 A.xlsx 和 B.xlsx。

    Returns
    -------
    A : np.ndarray, shape (10000, 99)
    B : np.ndarray, shape (10000, 1)
    """
    data_dir = Path(data_dir)
    A = load_excel(data_dir / "A.xlsx")
    B = load_excel(data_dir / "B.xlsx")
    return A, B
