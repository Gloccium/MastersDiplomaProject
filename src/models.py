import time
import pandas as pd
import statsmodels.formula.api as smf

# -------------------- pandas compat for causalimpact (pandas 3.x) --------------------
# causalimpact==0.2.6 uses a removed private pandas function:
# pd.core.dtypes.common.is_datetime_or_timedelta_dtype
# Restore it using public pandas.api.types.
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
# ------------------------------------------------------------------------------------

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
        self.result = None  # сохраняем объект CausalImpact

    @staticmethod
    def _find_inference_col(inf: pd.DataFrame, target: str) -> str:
        """
        Ищет колонку в ci.inferences по нормализованному имени.
        target: например 'point_effect', 'point_effect_lower', 'point_effect_upper'
        """
        target_norm = str(target).lower().strip().replace(" ", "_").replace(".", "")
        cols = list(inf.columns)

        def norm(c):
            return str(c).lower().strip().replace(" ", "_").replace(".", "")

        # exact match
        for c in cols:
            if norm(c) == target_norm:
                return c

        # contains match (fallback)
        for c in cols:
            if target_norm in norm(c):
                return c

        raise KeyError(f"Cannot find '{target}' in inferences columns: {cols}")

    def fit_predict(self, df):
        start_time = time.time()

        # Панель -> wide (time x unit)
        wide_df = df.pivot(index="time", columns="unit", values="metric").sort_index()

        control_cols = df.loc[df["is_test"] == 0, "unit"].unique()
        test_cols = df.loc[df["is_test"] == 1, "unit"].unique()

        y = wide_df[test_cols].mean(axis=1)
        X = wide_df[control_cols].mean(axis=1)

        ci_data = pd.concat([y, X], axis=1).astype(float)

        # КРИТИЧНО для pandas 3.x + causalimpact 0.2.6:
        # causalimpact местами делает series[0] ожидая позиционный доступ.
        # В pandas 3 series[0] = label 0, поэтому даем колонкам label 0,1,...
        ci_data.columns = list(range(ci_data.shape[1]))  # response=0, covariate=1

        # causalimpact устойчивее работает на datetime индексе
        ci_data.index = pd.date_range(start="2020-01-01", periods=len(ci_data), freq="D")

        pre_period = [ci_data.index[0], ci_data.index[self.t_pre - 1]]
        post_period = [ci_data.index[self.t_pre], ci_data.index[-1]]

        ci = CausalImpact(
            ci_data,
            pre_period,
            post_period,
            model_args={"nseasons": 7, "season_duration": 1},
        )

        # Некоторые версии требуют явного запуска
        if getattr(ci, "inferences", None) is None and hasattr(ci, "run"):
            ci.run()

        if getattr(ci, "inferences", None) is None:
            raise RuntimeError("CausalImpact did not produce `inferences` (it is None).")

        inf = ci.inferences

        # Берем эффект по пост периоду:
        # 'Average Abs Effect' ~= mean(point_effect) over post period
        c_eff = self._find_inference_col(inf, "point_effect")
        c_lo = self._find_inference_col(inf, "point_effect_lower")
        c_hi = self._find_inference_col(inf, "point_effect_upper")

        post_inf = inf.loc[post_period[0] : post_period[1], :]

        self.effect = float(post_inf[c_eff].mean())
        self.ci_lower = float(post_inf[c_lo].mean())
        self.ci_upper = float(post_inf[c_hi].mean())

        self.time_taken = time.time() - start_time
        self.result = ci

        return self.effect, self.ci_lower, self.ci_upper