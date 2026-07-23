import numpy as np


def build_design_matrix(X: np.ndarray, fit_intercept: bool = True) -> np.ndarray:
    if fit_intercept:
        ones = np.ones((X.shape[0], 1), dtype=X.dtype)
        return np.hstack([X, ones])
    return X

class LinearRegression:
    def __init__(self,fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.coef_: np.ndarray | None = None    # w
        self.intercept_: float | None = None    # b

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
        X_aug = build_design_matrix(X, self.fit_intercept)

        '''
        使用lstsq 求解 min ||X_aug @ beta - y||^2
        '''
        beta, residuals, rank, s = np.linalg.lstsq(X_aug, y, rcond=None)
        
        if self.fit_intercept:
            self.coef_ = beta[:-1]      # w，长度 99
            self.intercept_ = beta[-1]    # b，标量
        else:
            self.coef_ = beta             # w，长度 99
            self.intercept_ = 0.0
        
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("模型未拟合，先调用 fit()")
        
        return X @ self.coef_ + self.intercept_

    @property
    def params_(self) -> np.ndarray:
        """
        合并 coef_ 和 intercept_ 为完整 β 向量（长度 100）。
        """
        if self.coef_ is None:
            raise RuntimeError("模型未拟合")
        
        if self.fit_intercept:
            return np.concatenate([self.coef_, np.array([self.intercept_])])
        return self.coef_