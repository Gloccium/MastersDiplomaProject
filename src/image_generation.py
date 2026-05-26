import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import timedelta

# Настройки стиля для красивой бизнес-презентации
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 14, 'axes.titlesize': 18, 'axes.labelsize': 14})


def create_business_plot(dates, cumulative_effect, ci_lower, ci_upper, intervention_date,
                         title, annotation_text, is_injection=False, filename='plot.png'):
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)  # Высокое разрешение для презентации

    # 1. Базовая линия (Ноль)
    ax.axhline(0, color='black', linewidth=1.5, zorder=1)

    # 2. Доверительный интервал (Серая зона)
    ax.fill_between(dates, ci_lower, ci_upper, color='lightgray', alpha=0.6, label='95% Доверительный интервал',
                    zorder=2)

    # 3. Линия эффекта (Красный пунктир)
    ax.plot(dates, cumulative_effect, color='red', linestyle='--', linewidth=3, label='Накопленный каузальный эффект',
            zorder=3)

    # 4. Линия вмешательства
    ax.axvline(intervention_date, color='black', linestyle=':', linewidth=2, zorder=4)
    ax.text(intervention_date - timedelta(days=2), ax.get_ylim()[1] * 0.8, 'Дата релиза\n(Вмешательство)',
            rotation=90, va='top', ha='right', fontsize=12, color='black', fontweight='bold')

    # 5. Бизнес-аннотации (САМОЕ ВАЖНОЕ)
    if is_injection:
        # Стрелка и текст для инъекции
        ax.annotate(annotation_text,
                    xy=(intervention_date + timedelta(days=25), cumulative_effect[-15]),
                    xytext=(intervention_date + timedelta(days=5), cumulative_effect[-15] + 150000),
                    arrowprops=dict(facecolor='green', shrink=0.05, width=4, headwidth=15),
                    fontsize=14, fontweight='bold', color='darkgreen',
                    bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="green", lw=2))
    else:
        # Текстовая плашка для А/А теста
        ax.text(intervention_date + timedelta(days=10), 100000, annotation_text,
                fontsize=14, fontweight='bold', color='darkgreen',
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="green", lw=2))

    # Оформление осей
    ax.set_title(title, pad=20, fontweight='bold')
    ax.set_ylabel('Накопленный эффект (GMV)')
    ax.set_xlabel('Дата')

    # Форматирование дат
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))

    ax.legend(loc='upper left', frameon=True, shadow=True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Сохранен график: {filename}")


# --- ГЕНЕРАЦИЯ ДАННЫХ (Имитация твоих графиков Olist) ---
dates = pd.date_range(start='2020-01-01', end='2020-05-30')
intervention_date = pd.to_datetime('2020-04-01')
idx_int = list(dates).index(intervention_date)

# 1. Данные для A/A теста (Плацебо)
# Эффект колеблется около нуля
cum_eff_aa = np.zeros(len(dates))
cum_eff_aa[idx_int:] = np.cumsum(np.random.normal(0, 1000, len(dates) - idx_int))
ci_lower_aa = np.zeros(len(dates))
ci_upper_aa = np.zeros(len(dates))
ci_lower_aa[idx_int:] = -np.arange(len(dates) - idx_int) * 4000
ci_upper_aa[idx_int:] = np.arange(len(dates) - idx_int) * 4000

create_business_plot(
    dates, cum_eff_aa, ci_lower_aa, ci_upper_aa, intervention_date,
    title='Валидация 1: А/А тест (Плацебо) на исторических данных Olist',
    annotation_text='✅ Эффект = 0\nМодель не выявила\nложных срабатываний\n(Ошибки I рода нет)',
    is_injection=False,
    filename='aa_test_business.png'
)

# 2. Данные для Теста с инъекцией (+20%)
# Эффект уверенно растет вверх
cum_eff_inj = np.zeros(len(dates))
cum_eff_inj[idx_int:] = np.cumsum(np.random.normal(1000, 500, len(dates) - idx_int))
ci_lower_inj = np.zeros(len(dates))
ci_upper_inj = np.zeros(len(dates))
ci_lower_inj[idx_int:] = cum_eff_inj[idx_int:] - np.arange(len(dates) - idx_int) * 4000
ci_upper_inj[idx_int:] = cum_eff_inj[idx_int:] + np.arange(len(dates) - idx_int) * 4000

create_business_plot(
    dates, cum_eff_inj, ci_lower_inj, ci_upper_inj, intervention_date,
    title='Валидация 2: Искусственная инъекция эффекта (+20% к GMV)',
    annotation_text='🚀 Модель успешно\nзафиксировала рост\nнакопленной выручки\nна фоне рыночного шума',
    is_injection=True,
    filename='injection_test_business.png'
)