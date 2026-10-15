"""Local portfolio demo. Train once with: credit-risk train."""
from pathlib import Path
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from credit_risk.data import CATEGORIES
from credit_risk.model import explain_one, load_artifact, predict_frame

ROOT = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ.get("CREDIT_RISK_MODEL", ROOT / "artifacts/model.joblib"))

st.set_page_config(page_title="Credit-risk | ML demo", page_icon="📊", layout="wide")
st.title("Credit-risk · ML demo")
st.caption("От исходных данных до воспроизводимого предсказания и объяснения модели")
st.info(
    "Учебная модель приближает искусственное правило. Score — выход модели для этой метки, "
    "а не вероятность реального дефолта. Приложение не принимает кредитных решений."
)

if not MODEL_PATH.is_file():
    st.warning("Сначала обучите модель: credit-risk train")
    st.code("python -m credit_risk.cli train --data credit.csv --output artifacts")
    st.stop()


@st.cache_resource
def get_model(path: str, modified_ns: int):
    return load_artifact(Path(path))


try:
    bundle = get_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime_ns)
except (ValueError, OSError, KeyError) as exc:
    st.error(f"Не удалось загрузить модель: {exc}")
    st.stop()

with st.form("profile"):
    left, right = st.columns(2)
    with left:
        age = st.number_input("Возраст", min_value=18, max_value=100, value=35)
        amount = st.number_input("Сумма кредита (единицы исходных данных)",
                                 min_value=1.0, max_value=100000.0, value=5000.0)
        duration = st.number_input("Срок, месяцев", min_value=1, max_value=120, value=24)
        job = st.selectbox("Код занятости", [0, 1, 2, 3], index=2)
    with right:
        housing = st.selectbox("Жильё", CATEGORIES["Housing"])
        saving = st.selectbox("Накопления", [None] + CATEGORIES["Saving accounts"],
                             format_func=lambda x: "Не указано" if x is None else x)
        checking = st.selectbox("Текущий счёт", [None] + CATEGORIES["Checking account"],
                               format_func=lambda x: "Не указано" if x is None else x)
        purpose = st.selectbox("Цель", CATEGORIES["Purpose"], index=1)
    submitted = st.form_submit_button("Рассчитать учебный score")

if submitted:
    raw = pd.DataFrame([{
        "Age": age, "Credit amount": amount, "Duration": duration, "Job": job,
        "Housing": housing, "Saving accounts": saving,
        "Checking account": checking, "Purpose": purpose,
    }])
    try:
        result = predict_frame(bundle, raw).iloc[0]
        baseline, contributions = explain_one(bundle, raw)
    except ValueError as exc:
        st.error(str(exc))
    else:
        cols = st.columns(3)
        cols[0].metric("Учебный score", f"{result['score']:.1%}")
        cols[1].metric("Предсказанный класс", str(int(result["predicted_class"])))
        cols[2].metric("Метка точного правила", str(int(result["rule_label"])))
        st.caption(
            f"Порог выбран на validation: {bundle['threshold']:.2f}. "
            f"Демонстрационная категория: {result['display_band']} "
            "(A < 0.2; B < 0.5; C ≥ 0.5)."
        )
        st.subheader("Почему модель получила такой score")
        shown = contributions.head(10).iloc[::-1]
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.barh(shown["feature"], shown["log_odds_contribution"],
                color=np.where(shown["log_odds_contribution"] >= 0, "#b74f4f", "#34788b"))
        ax.axvline(0, color="#777777", linewidth=0.8)
        ax.set_xlabel("Вклад в log-odds относительно среднего по train")
        fig.tight_layout()
        st.pyplot(fig, width=900)
        plt.close(fig)
        st.caption(
            f"Показаны 10 крупнейших вкладов. Полная сумма всех вкладов "
            f"+ baseline {baseline:.3f} = log-odds модели. Это точная линейная "
            "декомпозиция, не SHAP и не причинное объяснение."
        )
        with st.expander("Все вклады и исходные признаки"):
            st.dataframe(contributions, hide_index=True)
            st.dataframe(raw, hide_index=True)

with st.expander("Эксперимент и ограничения"):
    report = bundle["report"]
    st.write(report["target"])
    st.json({"splits": report["splits"], "test": report["test"]["logistic_regression"]})
    st.write("При малом количестве положительных примеров метрики нестабильны.")

