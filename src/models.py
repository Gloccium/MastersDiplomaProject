import time
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.optimize import minimize

# ---- compat: causalimpact 0.2.6 vs pandas>=3 ----
import pandas.core.dtypes.common as _pandas_common
from pandas.api.types import is_datetime64_any_dtype, is_timedelta64_dtype

if not hasattr(_pandas_common, "is_datetime_or_timedelta_dtype"):
    def is_datetime_or_timedelta_dtype(x):
        try:
            return bool(is_datetime64_any_dtype(x) or is_timedelta64_dtype(x))
        except Exception:
            pass
        try:
            s = pd.Series(x)
            return bool(is_datetime64_any_dtype(s) or is_timedelta64_dtype(s))
        except Exception:
            return False

    _pandas_common.is_datetime_or_timedelta_dtype = is_datetime_or_timedelta_dtype
# -----------------------------------------------

from causalimpact import CausalImpact


class BaselineDiD:
    """Классический Difference-in-Differences (OLS)"""

    def __init__(self):
        self.effect = None
        self.time_taken = 0.0

    def fit_predict(self, df: pd.DataFrame) -> float:
        start_time = time.time()
        formula = "metric ~ is_test * post_treatment"
        model = smf.ols(formula, data=df).fit()
        self.effect = float(model.params["is_test:post_treatment"])
        self.time_taken = time.time() - start_time
        return self.effect


class CausalImpactBSTS:
    """BSTS через causalimpact (совместимо с pandas 3.x + causalimpact 0.2.6)"""

    def __init__(self, t_pre: int):
        self.t_pre = int(t_pre)
        self.effect = None
        self.ci_lower = None
        self.ci_upper = None
        self.time_taken = 0.0
        self.result = None

    @staticmethod
    def _find_inference_col(inf: pd.DataFrame, target: str) -> str:
        target_norm = str(target).lower().strip().replace(" ", "_").replace(".", "")
        cols = list(inf.columns)

        def norm(c):
            return str(c).lower().strip().replace(" ", "_").replace(".", "")

        for c in cols:
            if norm(c) == target_norm:
                return c
        for c in cols:
            if target_norm in norm(c):
                return c
        raise KeyError(f"Cannot find '{target}' in inferences columns: {cols}")

    def fit_predict(self, df: pd.DataFrame):
        start_time = time.time()

        # Panel -> wide
        wide_df = df.pivot(index="time", columns="unit", values="metric").sort_index()

        control_cols = df.loc[df["is_test"] == 0, "unit"].unique()
        test_cols = df.loc[df["is_test"] == 1, "unit"].unique()

        y = wide_df[test_cols].mean(axis=1)
        X = wide_df[control_cols].mean(axis=1)

        ci_data = pd.concat([y, X], axis=1).astype(float)

        # pandas 3.x + causalimpact 0.2.6: series[0] interpreted as label
        ci_data.columns = list(range(ci_data.shape[1]))  # [0=response, 1=covariate]

        # causalimpact стабильнее работает с datetime индексом
        try:
            ci_data.index = pd.to_datetime(ci_data.index)
        except Exception:
            ci_data.index = pd.date_range(start="2020-01-01", periods=len(ci_data), freq="D")

        if self.t_pre <= 1 or self.t_pre >= len(ci_data):
            raise ValueError(f"Invalid t_pre={self.t_pre} for series length={len(ci_data)}")

        pre_period = [ci_data.index[0], ci_data.index[self.t_pre - 1]]
        post_period = [ci_data.index[self.t_pre], ci_data.index[-1]]

        ci = CausalImpact(
            ci_data,
            pre_period,
            post_period,
            model_args={"nseasons": 7, "season_duration": 1},
        )

        # Некоторые сборки требуют явного запуска
        if getattr(ci, "inferences", None) is None and hasattr(ci, "run"):
            ci.run()

        if getattr(ci, "inferences", None) is None:
            raise RuntimeError("CausalImpact did not produce `inferences` (it is None).")

        inf = ci.inferences
        c_eff = self._find_inference_col(inf, "point_effect")
        c_lo = self._find_inference_col(inf, "point_effect_lower")
        c_hi = self._find_inference_col(inf, "point_effect_upper")

        post_inf = inf.loc[post_period[0]:post_period[1], :]

        self.effect = float(post_inf[c_eff].mean())

        # FIX: гарантируем lower <= upper (иногда causalimpact/SM может вернуть наоборот)
        lo = post_inf[c_lo]
        hi = post_inf[c_hi]
        lo2 = np.minimum(lo, hi)
        hi2 = np.maximum(lo, hi)

        self.ci_lower = float(lo2.mean())
        self.ci_upper = float(hi2.mean())

        self.time_taken = time.time() - start_time
        self.result = ci
        return self.effect, self.ci_lower, self.ci_upper


class SyntheticDiD:
    """
    SDID / Synthetic DiD (практический вариант):
    - на pre-периоде подбираем веса доноров (unit weights) с ограничениями w>=0, sum(w)=1
    - затем считаем двойную разность (Test - Synth) post vs pre

    Это не "канонический" SDID из papers (там ещё time weights), но по смыслу
    это разумный SDID-подобный baseline (часто его и называют SDID в прикладных проектах).
    """

    def __init__(self, t_pre: int, ridge_lambda: float = 0.01):
        self.t_pre = int(t_pre)
        self.ridge_lambda = float(ridge_lambda)
        self.effect = None
        self.time_taken = 0.0
        self.weights = None
        self.control_units = None

    def fit_predict(self, df: pd.DataFrame) -> float:
        start_time = time.time()

        wide_df = df.pivot(index="time", columns="unit", values="metric").sort_index()

        control_cols = df.loc[df["is_test"] == 0, "unit"].unique()
        test_cols = df.loc[df["is_test"] == 1, "unit"].unique()

        if len(control_cols) < 2:
            raise ValueError("SDID requires at least 2 control units for stable weights.")
        if len(test_cols) < 1:
            raise ValueError("No treated unit found (is_test==1).")

        self.control_units = list(control_cols)

        y = wide_df[test_cols].mean(axis=1).values
        X = wide_df[control_cols].values

        if self.t_pre <= 1 or self.t_pre >= len(y):
            raise ValueError(f"Invalid t_pre={self.t_pre} for series length={len(y)}")

        y_pre = y[: self.t_pre]
        X_pre = X[: self.t_pre, :]

        y_post = y[self.t_pre :]
        X_post = X[self.t_pre :, :]

        n_controls = X_pre.shape[1]
        w0 = np.ones(n_controls) / n_controls

        def loss(w):
            diff = y_pre - X_pre.dot(w)
            return np.mean(diff ** 2) + self.ridge_lambda * np.sum(w ** 2)

        cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
        bounds = [(0.0, 1.0) for _ in range(n_controls)]

        res = minimize(loss, w0, method="SLSQP", bounds=bounds, constraints=cons)

        if not res.success:
            # fallback на равномерные веса (чтобы приложение не падало)
            w = w0
        else:
            w = res.x

        self.weights = w

        synth_pre = X_pre.dot(w)
        synth_post = X_post.dot(w)

        diff_post = float(np.mean(y_post) - np.mean(synth_post))
        diff_pre = float(np.mean(y_pre) - np.mean(synth_pre))

        self.effect = float(diff_post - diff_pre)
        self.time_taken = time.time() - start_time
        return self.effect