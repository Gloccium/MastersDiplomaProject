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
    """Классический Difference-in-Differences"""

    def __init__(self):
        self.effect = None
        self.time_taken = 0.0

    def fit_predict(self, df):
        start_time = time.time()
        formula = "metric ~ is_test * post_treatment"
        model = smf.ols(formula, data=df).fit()
        self.effect = float(model.params["is_test:post_treatment"])
        self.time_taken = time.time() - start_time
        return self.effect


class CausalImpactBSTS:
    """Обертка для алгоритма BSTS"""

    def __init__(self, t_pre):
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

    def fit_predict(self, df):
        start_time = time.time()

        wide_df = df.pivot(index="time", columns="unit", values="metric").sort_index()

        control_cols = df.loc[df["is_test"] == 0, "unit"].unique()
        test_cols = df.loc[df["is_test"] == 1, "unit"].unique()

        y = wide_df[test_cols].mean(axis=1)
        X = wide_df[control_cols].mean(axis=1)

        ci_data = pd.concat([y, X], axis=1).astype(float)
        ci_data.columns = list(range(ci_data.shape[1]))  # [0, 1] для pandas 3 + causalimpact 0.2.6

        # datetime индекс для стабильности
        ci_data.index = pd.date_range(start="2020-01-01", periods=len(ci_data), freq="D")

        pre_period = [ci_data.index[0], ci_data.index[self.t_pre - 1]]
        post_period = [ci_data.index[self.t_pre], ci_data.index[-1]]

        ci = CausalImpact(ci_data, pre_period, post_period, model_args={"nseasons": 7, "season_duration": 1})

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

        # FIX: гарантируем правильный порядок границ
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
    Практическая реализация Synthetic Difference-in-Differences.
    Использует L2-регуляризованную оптимизацию для поиска весов доноров (unit weights).
    """
    def __init__(self, t_pre):
        self.t_pre = int(t_pre)
        self.effect = None
        self.time_taken = 0.0
        self.weights = None # Веса объектов-доноров

    def fit_predict(self, df):
        start_time = time.time()

        # 1. Подготовка данных (переход к матричному виду)
        wide_df = df.pivot(index="time", columns="unit", values="metric").sort_index()
        control_cols = df.loc[df["is_test"] == 0, "unit"].unique()
        test_cols = df.loc[df["is_test"] == 1, "unit"].unique()

        # Усредняем тестовую группу (это наша целевая переменная)
        y = wide_df[test_cols].mean(axis=1).values
        X = wide_df[control_cols].values # Матрица контрольных доноров

        # Разделение на pre и post периоды
        y_pre = y[:self.t_pre]
        X_pre = X[:self.t_pre, :]

        y_post = y[self.t_pre:]
        X_post = X[self.t_pre:, :]

        # 2. Поиск весов юнитов (Unit Weights) через квадратичную оптимизацию
        n_controls = X_pre.shape[1]
        initial_weights = np.ones(n_controls) / n_controls

        # Функция потерь: MSE между тестовой группой и взвешенным контролем + L2 регуляризация
        def loss(w):
            diff = y_pre - X_pre.dot(w)
            return np.mean(diff**2) + 0.01 * np.sum(w**2) # 0.01 - параметр ridge-регуляризации

        # Ограничения: сумма весов = 1, все веса >= 0
        cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = [(0, 1) for _ in range(n_controls)]

        # Запуск решателя SLSQP
        res = minimize(loss, initial_weights, method='SLSQP', bounds=bounds, constraints=cons)
        self.weights = res.x

        # 3. Построение синтетического контроля
        synth_pre = X_pre.dot(self.weights)
        synth_post = X_post.dot(self.weights)

        # 4. Расчет эффекта (двойная разность: Test vs SynthControl)
        diff_post = np.mean(y_post) - np.mean(synth_post)
        diff_pre = np.mean(y_pre) - np.mean(synth_pre)

        self.effect = float(diff_post - diff_pre)
        self.time_taken = time.time() - start_time

        return self.effect