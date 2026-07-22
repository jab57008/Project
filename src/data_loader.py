import pandas as pd
import numpy as np
from pathlib import Path

def load_excel(path: Path, sheet_name=0) -> np.ndarray:
    df=pd.read_excel(path, sheet_name=sheet_name,header=None)
    return df.to_numpy (dtype=np.float64)

#测试
B= load_excel(Path("C:/Users/16915/Desktop/Problem B/Project/problem1/附件1/B.xlsx"))
print(B.shape)