from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class OlistPanelResult:
    panel: pd.DataFrame
    t_pre: int
    dates: pd.DatetimeIndex
    states: list[str]
    treated_state: str
    intervention_date: pd.Timestamp


def build_olist_daily_gmv(
    orders: pd.DataFrame,
    payments: pd.DataFrame,
    customers: pd.DataFrame,
) -> pd.DataFrame:
    """
    Builds daily GMV by customer_state from raw Olist tables.
    Output columns: ['date', 'customer_state', 'gmv']
    """
    # Parse timestamps
    if "order_purchase_timestamp" not in orders.columns:
        raise ValueError("orders CSV must contain column: order_purchase_timestamp")
    if "order_id" not in orders.columns or "order_id" not in payments.columns:
        raise ValueError("orders & payments must contain column: order_id")
    if "customer_id" not in orders.columns or "customer_id" not in customers.columns:
        raise ValueError("orders & customers must contain column: customer_id")
    if "payment_value" not in payments.columns:
        raise ValueError("payments CSV must contain column: payment_value")
    if "customer_state" not in customers.columns:
        raise ValueError("customers CSV must contain column: customer_state")

    tmp = orders.merge(payments, on="order_id", how="inner").merge(customers, on="customer_id", how="inner")

    tmp["date"] = pd.to_datetime(tmp["order_purchase_timestamp"], errors="coerce").dt.floor("D")
    tmp = tmp.dropna(subset=["date", "customer_state", "payment_value"])
    tmp["payment_value"] = pd.to_numeric(tmp["payment_value"], errors="coerce")
    tmp = tmp.dropna(subset=["payment_value"])

    daily = (
        tmp.groupby(["date", "customer_state"], as_index=False)["payment_value"]
        .sum()
        .rename(columns={"payment_value": "gmv"})
        .sort_values(["date", "customer_state"])
    )
    return daily


def select_top_states(daily_gmv: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, top_n: int) -> list[str]:
    d = daily_gmv[(daily_gmv["date"] >= start) & (daily_gmv["date"] <= end)].copy()
    if d.empty:
        return []
    totals = d.groupby("customer_state")["gmv"].sum().sort_values(ascending=False)
    return totals.head(top_n).index.tolist()


def build_panel_from_daily_gmv(
    daily_gmv: pd.DataFrame,
    states: Sequence[str],
    treated_state: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    intervention_date: pd.Timestamp,
) -> OlistPanelResult:
    """
    Converts daily_gmv into required panel format used by your models:
    columns: time, unit, metric, is_test, post_treatment
    Ensures balanced panel (full grid date x state) and fills missing with 0.
    """
    if treated_state not in states:
        raise ValueError("treated_state must be included in selected states")
    if len(states) < 2:
        raise ValueError("Select at least 2 states (treated + at least 1 control).")

    dates = pd.date_range(start=start, end=end, freq="D")
    if intervention_date not in dates:
        raise ValueError("Intervention date must be within selected date range.")

    t_pre = int((dates < intervention_date).sum())
    if t_pre < 5:
        # не ошибка, но для устойчивости модели лучше предупреждать на UI
        pass

    # filter + full grid
    d = daily_gmv[daily_gmv["customer_state"].isin(states)].copy()
    d = d[(d["date"] >= start) & (d["date"] <= end)].copy()

    full_index = pd.MultiIndex.from_product([dates, list(states)], names=["time", "unit"])
    panel = (
        d.rename(columns={"date": "time", "customer_state": "unit", "gmv": "metric"})
        .set_index(["time", "unit"])
        .reindex(full_index)
        .reset_index()
    )
    panel["metric"] = panel["metric"].fillna(0.0).astype(float)
    panel["is_test"] = (panel["unit"] == treated_state).astype(int)
    panel["post_treatment"] = (panel["time"] >= intervention_date).astype(int)

    return OlistPanelResult(
        panel=panel,
        t_pre=t_pre,
        dates=dates,
        states=list(states),
        treated_state=treated_state,
        intervention_date=intervention_date,
    )