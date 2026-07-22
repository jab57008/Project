import pandas as pd
import numpy as np
from pathlib import Path

def load_excel(path: Path, sheet_name=0) -> np.ndarray:
    df=pd.read_excel(path, sheet_name=sheet_name)
    return df.to_numpy (dtype=np.float64)