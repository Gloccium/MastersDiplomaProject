import numpy as np
import pandas as pd


class DataGenerator:
    """Генератор синтетических панельных данных"""

    def __init__(self, n_control=30, n_test=5, t_pre=60, t_post=30, base_value=100):
        self.n_control = n_control
        self.n_test = n_test
        self.t_pre = t_pre
        self.t_post = t_post
        self.total_time = t_pre + t_post
        self.total_units = n_control + n_test
        self.base_value = base_value

        units = np.repeat(np.arange(self.total_units), self.total_time)
        times = np.tile(np.arange(self.total_time), self.total_units)

        self.df = pd.DataFrame(
            {
                "unit": units,
                "time": times,
                "is_test": np.where(units >= self.n_control, 1, 0),
                "metric": float(self.base_value),
            }
        )
        self.df["post_treatment"] = np.where(self.df["time"] >= self.t_pre, 1, 0)
        self.df["treatment_effect_true"] = 0.0

    def add_ar_noise(self, rho=0.5, sigma=5.0):
        noise = np.zeros(len(self.df))
        for i in range(self.total_units):
            mask = self.df["unit"] == i
            n_obs = mask.sum()
            unit_noise = np.zeros(n_obs)
            unit_noise[0] = np.random.normal(0, sigma)
            for t in range(1, n_obs):
                unit_noise[t] = rho * unit_noise[t - 1] + np.random.normal(0, sigma)
            noise[mask] = unit_noise
        self.df["metric"] += noise
        return self

    def add_trend(self, slope_control=0.2, slope_test=0.2):
        control_trend = self.df["time"] * slope_control
        test_trend = self.df["time"] * slope_test
        trend = np.where(self.df["is_test"] == 0, control_trend, test_trend)
        self.df["metric"] += trend
        return self

    def add_seasonality(self, period=7, amplitude_control=10, amplitude_test=10):
        time_vals = self.df["time"].values
        control_season = amplitude_control * np.sin(2 * np.pi * time_vals / period)
        test_season = amplitude_test * np.sin(2 * np.pi * time_vals / period)
        seasonality = np.where(self.df["is_test"] == 0, control_season, test_season)
        self.df["metric"] += seasonality
        return self

    def inject_treatment(self, effect_size=20.0):
        self.df["metric_counterfactual"] = self.df["metric"].copy()
        treatment_mask = (self.df["is_test"] == 1) & (self.df["post_treatment"] == 1)
        self.df.loc[treatment_mask, "metric"] += effect_size
        self.df.loc[treatment_mask, "treatment_effect_true"] = effect_size
        return self

    # --- scenarios ---
    def get_scenario_ideal(self, effect_size=15.0):
        """Параллельные тренды + шум + лечение"""
        self.add_ar_noise(rho=0.4, sigma=2.0)
        self.add_trend(slope_control=0.2, slope_test=0.2)
        self.inject_treatment(effect_size)
        return self.df

    def get_scenario_non_parallel_trends(self, effect_size=15.0):
        self.add_ar_noise(rho=0.4, sigma=2.0)
        self.add_trend(slope_control=0.1, slope_test=0.8)
        self.inject_treatment(effect_size)
        return self.df

    def get_scenario_seasonality(self, effect_size=15.0):
        self.add_ar_noise(rho=0.5, sigma=2.0)
        self.add_trend(slope_control=0.1, slope_test=0.1)
        self.add_seasonality(period=7, amplitude_control=5, amplitude_test=20)
        self.inject_treatment(effect_size)
        return self.df