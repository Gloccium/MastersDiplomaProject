import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# streamlit config
st.set_page_config(page_title="Causal Inference MVP", layout="wide")

# убираем шумный warning от statsmodels/causalimpact (не влияет на результат)
warnings.filterwarnings(
    "ignore",
    message=r"Unknown keyword arguments: dict_keys\(\['alpha'\]\)",
    category=FutureWarning,
)

# imports from project
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.dgp import DataGenerator
from src.models import BaselineDiD, CausalImpactBSTS, SyntheticDiD
from app.olist_utils import build_olist_daily_gmv, select_top_states, build_panel_from_daily_gmv

# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------
def plot_causalimpact_result(ci_obj):
    ci_obj.plot()
    fig = plt.gcf()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

def did_requirements_ok(df: pd.DataFrame) -> bool:
    need = {"metric", "is_test", "post_treatment"}
    return need.issubset(df.columns)

def bsts_requirements_ok(df: pd.DataFrame) -> bool:
    need = {"time", "unit", "metric", "is_test", "post_treatment"}
    return need.issubset(df.columns)

# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
st.title("Инструмент оценки причинного эффекта (Causal Inference)")
st.markdown("Оценка продуктовых изменений в условиях отсутствия A/B тестов.")

st.sidebar.header("Настройки")

data_source = st.sidebar.radio(
    "Источник данных:",
    ["Синтетические данные (Демо)", "Olist (3 CSV файла)"],
)

# session defaults
if "df" not in st.session_state:
    st.session_state["df"] = None
if "t_pre" not in st.session_state:
    st.session_state["t_pre"] = 60

# -----------------------------------------------------------------------------
# DATA SOURCE: Synthetic
# -----------------------------------------------------------------------------
if data_source == "Синтетические данные (Демо)":
    st.sidebar.subheader("Синтетика")

    scenario = st.sidebar.selectbox(
        "Сценарий (DGP):",
        [
            "Базовый (параллельные тренды)",
            "Сложный (нарушение трендов)",
            "Гетерогенная сезонность",
        ],
    )
    t_pre = st.sidebar.number_input("t_pre (точка вмешательства)", min_value=5, value=int(st.session_state["t_pre"]))
    true_effect = st.sidebar.number_input("Истинный эффект (для демо)", value=15.0)

    if st.sidebar.button("Сгенерировать данные", type="primary"):
        dgp = DataGenerator(t_pre=int(t_pre), t_post=30)

        if scenario == "Базовый (параллельные тренды)":
            # если у тебя нет get_scenario_ideal — делаем fallback без изменения смысла
            if hasattr(dgp, "get_scenario_ideal"):
                df = dgp.get_scenario_ideal(effect_size=float(true_effect))
            else:
                dgp.add_ar_noise(rho=0.4, sigma=2.0)
                dgp.add_trend(slope_control=0.2, slope_test=0.2)
                dgp.inject_treatment(effect_size=float(true_effect))
                df = dgp.df

        elif scenario == "Сложный (нарушение трендов)":
            df = dgp.get_scenario_non_parallel_trends(effect_size=float(true_effect))
        else:
            df = dgp.get_scenario_seasonality(effect_size=float(true_effect))

        st.session_state["df"] = df
        st.session_state["t_pre"] = int(t_pre)
        st.success("Синтетические данные сгенерированы.")

# -----------------------------------------------------------------------------
# DATA SOURCE: Olist
# -----------------------------------------------------------------------------
else:
    st.sidebar.subheader("Olist: загрузите 3 файла")

    # Удобно: если файлы лежат в ../data, можно не загружать руками
    use_local = st.sidebar.checkbox("Использовать локальные файлы из ../data", value=True)

    orders_file = payments_file = customers_file = None

    data_dir = PROJECT_ROOT / "data"
    orders_path = data_dir / "olist_orders_dataset.csv"
    payments_path = data_dir / "olist_order_payments_dataset.csv"
    customers_path = data_dir / "olist_customers_dataset.csv"

    if use_local and orders_path.exists() and payments_path.exists() and customers_path.exists():
        orders = pd.read_csv(orders_path)
        payments = pd.read_csv(payments_path)
        customers = pd.read_csv(customers_path)
        st.sidebar.success("Локальные Olist файлы найдены и загружены.")
    else:
        orders_file = st.sidebar.file_uploader("1) olist_orders_dataset.csv", type=["csv"])
        payments_file = st.sidebar.file_uploader("2) olist_order_payments_dataset.csv", type=["csv"])
        customers_file = st.sidebar.file_uploader("3) olist_customers_dataset.csv", type=["csv"])

        if orders_file is not None and payments_file is not None and customers_file is not None:
            orders = pd.read_csv(orders_file)
            payments = pd.read_csv(payments_file)
            customers = pd.read_csv(customers_file)
        else:
            orders = payments = customers = None

    if orders is not None:
        daily_gmv = build_olist_daily_gmv(orders, payments, customers)

        st.sidebar.markdown("---")
        st.sidebar.subheader("Параметры панели")

        min_date = daily_gmv["date"].min()
        max_date = daily_gmv["date"].max()

        # диапазон анализа
        start_date = st.sidebar.date_input("Начало периода", value=min_date.date())
        end_date = st.sidebar.date_input("Конец периода", value=max_date.date())

        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)

        if start >= end:
            st.sidebar.error("Начало периода должно быть раньше конца.")
        else:
            top_n = st.sidebar.slider("Сколько штатов оставить (Top-N по GMV)", min_value=2, max_value=15, value=5)
            top_states = select_top_states(daily_gmv, start, end, top_n=top_n)

            if not top_states:
                st.sidebar.error("Не удалось выбрать штаты. Проверь диапазон дат.")
            else:
                states = st.sidebar.multiselect("Штаты (unit)", options=sorted(daily_gmv["customer_state"].unique()),
                                                default=top_states)

                if len(states) < 2:
                    st.sidebar.warning("Нужно выбрать минимум 2 штата (тест + контроль).")
                else:
                    default_treated = "RJ" if "RJ" in states else states[0]
                    treated_state = st.sidebar.selectbox("Тестовый штат (treated)", options=states, index=states.index(default_treated))

                    # дата вмешательства
                    # дефолт: середина периода
                    default_int = (start + (end - start) / 2).floor("D")
                    intervention_date = st.sidebar.date_input(
                        "Дата вмешательства",
                        value=default_int.date(),
                        min_value=start.date(),
                        max_value=end.date(),
                    )
                    intervention_date = pd.to_datetime(intervention_date)

                    if st.sidebar.button("Собрать панель Olist", type="primary"):
                        try:
                            res = build_panel_from_daily_gmv(
                                daily_gmv=daily_gmv,
                                states=states,
                                treated_state=treated_state,
                                start=start,
                                end=end,
                                intervention_date=intervention_date,
                            )
                            st.session_state["df"] = res.panel
                            st.session_state["t_pre"] = res.t_pre
                            st.success(f"Панель собрана: states={len(res.states)}, t_pre={res.t_pre}")
                        except Exception as e:
                            st.error(f"Ошибка подготовки панели: {e}")

# -----------------------------------------------------------------------------
# MAIN: run models if data exists
# -----------------------------------------------------------------------------
df = st.session_state.get("df", None)
t_pre = int(st.session_state.get("t_pre", 60))

if df is None:
    st.info("Слева выберите источник и подготовьте данные.")
    st.stop()

st.subheader("Превью данных")
st.dataframe(df.head(30), use_container_width=True)

st.markdown("---")
st.subheader("Оценка эффекта")

col1, col2 = st.columns([1, 3])
with col1:
    model_choice = st.radio(
        "Алгоритм:",
        ["BSTS (CausalImpact)", "Synthetic DiD (SDID)", "Difference-in-Differences (DiD)"]
    )
    run_button = st.button("Рассчитать эффект", type="primary")

with col2:
    if run_button:
        if model_choice == "BSTS (CausalImpact)" and not bsts_requirements_ok(df):
            st.error("Для BSTS нужны колонки: time, unit, metric, is_test, post_treatment")
            st.stop()

        elif model_choice == "Synthetic DiD (SDID)":
            if not bsts_requirements_ok(df):  # SDID требует таких же колонок, как BSTS
                st.error("Для SDID нужны колонки: time, unit, metric, is_test, post_treatment")
                st.stop()

            model = SyntheticDiD(t_pre=t_pre)
            effect = model.fit_predict(df)

            m1, m2, m3 = st.columns(3)
            m1.metric("Оценка эффекта (ATE)", f"{effect:.2f}")
            m2.metric("Метод", "SDID (L2 Opt)")
            m3.metric("Время расчёта", f"{model.time_taken:.3f} сек")

            st.success("Веса доноров успешно рассчитаны. Синтетический контроль построен.")
            # Вывод топ-3 весов доноров для интерпретируемости
            control_cols = df.loc[df["is_test"] == 0, "unit"].unique()
            weights_df = pd.DataFrame({"Штат/Юнит": control_cols, "Вес": model.weights})
            weights_df = weights_df[weights_df["Вес"] > 0.01].sort_values("Вес", ascending=False)
            st.markdown("#### Наиболее значимые доноры")
            st.dataframe(weights_df.head(), use_container_width=True)

        elif model_choice == "Difference-in-Differences (DiD)" and not did_requirements_ok(df):
            st.error("Для DiD нужны колонки: metric, is_test, post_treatment")
            st.stop()

        with st.spinner("Обучение модели и расчёт..."):
            if model_choice == "BSTS (CausalImpact)":
                model = CausalImpactBSTS(t_pre=t_pre)
                effect, lower, upper = model.fit_predict(df)

                # страховка от перевёрнутого интервала
                lower, upper = (min(lower, upper), max(lower, upper))
                is_significant = not (lower <= 0 <= upper)

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Оценка эффекта (ATE)", f"{effect:.2f}")
                m2.metric("Доверительный интервал", f"[{lower:.2f}, {upper:.2f}]")
                m3.metric("Стат. значимость", "Да" if is_significant else "Нет")
                m4.metric("Время расчёта", f"{model.time_taken:.3f} сек")

                st.markdown("#### Графики")
                plot_causalimpact_result(model.result)

            else:
                st.caption("Примечание: сейчас это классический DiD (OLS), не SDID.")
                model = BaselineDiD()
                effect = model.fit_predict(df)

                m1, m2, m3 = st.columns(3)
                m1.metric("Оценка эффекта (ATE)", f"{effect:.2f}")
                m2.metric("Метод", "OLS DiD")
                m3.metric("Время расчёта", f"{model.time_taken:.3f} сек")

                st.info("DiD здесь без графиков контрфакта. SDID требует отдельной реализации.")