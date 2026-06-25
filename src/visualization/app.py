import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.config import (
    CLASSIFICATION_FEATURE_COLS,
    CLASSIFICATION_TARGET_MENTAL_HEALTH,
    DATA_PATH,
    MODELS_DIR,
    REGRESSION_FEATURE_COLS,
    REPORTS_DIR,
)
from src.data.load import load_raw
from src.features.engineer import add_all_engineered_features
from src.simulation.scenarios import predict_scenario
from src.visualization.plots import (
    plot_correlation_matrix,
    plot_missing_heatmap,
    plot_target_distribution,
)

st.set_page_config(page_title="Factores de Riesgo - El Salvador GSHS 2013", layout="wide")


@st.cache_data
def load_data() -> pd.DataFrame:
    df = load_raw(DATA_PATH)
    return add_all_engineered_features(df)


def render_data_overview(df: pd.DataFrame):
    st.header("Resumen del Dataset")
    col1, col2, col3 = st.columns(3)
    col1.metric("Filas", df.shape[0])
    col2.metric("Columnas", df.shape[1])
    col3.metric("Valores nulos (%)", f"{df.isnull().mean().mean() * 100:.1f}%")

    st.subheader("Patrón de valores faltantes")
    fig = plot_missing_heatmap(df)
    st.pyplot(fig)

    st.subheader("Estadísticas por columna")
    summary = pd.DataFrame({
        "null_pct": df.isnull().mean().round(3),
        "nunique": df.nunique(),
        "min": df.min(numeric_only=True),
        "max": df.max(numeric_only=True),
    })
    st.dataframe(summary)


def render_feature_distributions(df: pd.DataFrame):
    st.header("Distribución de Variables")
    col = st.selectbox("Selecciona una columna", sorted(df.columns.tolist()))
    st.write(df[col].describe())
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    df[col].dropna().hist(ax=ax, bins=20, edgecolor="black")
    ax.set_title(f"Distribución - {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Frecuencia")
    st.pyplot(fig)


def render_correlation_explorer(df: pd.DataFrame):
    st.header("Explorador de Correlaciones")
    group = st.selectbox("Grupo de features", ["Regresión (IMC)", "Clasificación (Salud Mental)"])
    cols = REGRESSION_FEATURE_COLS if group.startswith("Regresión") else CLASSIFICATION_FEATURE_COLS
    valid_cols = [c for c in cols if c in df.columns]
    fig = plot_correlation_matrix(df, valid_cols)
    st.pyplot(fig)


def render_model_results():
    st.header("Resultados de Modelos")
    report_path = REPORTS_DIR / "evaluation.json"
    if not report_path.exists():
        st.warning("No se encontraron reportes. Ejecuta `make evaluate` primero.")
        return

    with open(report_path) as f:
        results = json.load(f)

    st.json(results)

    for model_key in results:
        cm_path = REPORTS_DIR / f"confusion_matrix_{model_key}.png"
        fi_path = REPORTS_DIR / f"feature_importance_{model_key}.png"
        if cm_path.exists():
            st.subheader(f"Matriz de Confusión - {model_key}")
            st.image(str(cm_path))
        if fi_path.exists():
            st.subheader(f"Importancia de Características - {model_key}")
            st.image(str(fi_path))


def render_scenario_simulator(df: pd.DataFrame):
    st.header("Simulador de Escenarios de Riesgo Mental")
    target_col = CLASSIFICATION_TARGET_MENTAL_HEALTH

    model_path = MODELS_DIR / f"classification_logistic_{target_col}.joblib"
    if not model_path.exists():
        st.warning("Modelo no encontrado. Ejecuta `make train` primero.")
        return

    pipeline = joblib.load(model_path)
    available_features = [c for c in CLASSIFICATION_FEATURE_COLS if c in df.columns]

    st.markdown(
        "Ajusta los valores para simular distintos perfiles de riesgo de salud mental. "
        "Las columnas QN usan la escala OMS: **1 = Sí**, **2 = No**."
    )

    col1, col2 = st.columns(2)
    with col1:
        q1 = st.slider("Q1 - Edad (1=≤11, 6=≥16)", 1, 6, 3)
        q2 = st.selectbox("Q2 - Sexo (1=Masculino, 2=Femenino)", [1, 2])
        qn22 = st.selectbox("QN22 - Soledad frecuente (1=Sí, 2=No)", [1, 2], index=1)
        qn23 = st.selectbox("QN23 - Preocupación que impide dormir (1=Sí, 2=No)", [1, 2], index=1)
    with col2:
        qn35 = st.selectbox("QN35 - Consumió alcohol en últimos 30 días (1=Sí, 2=No)", [1, 2], index=1)
        qn16 = st.selectbox("QN16 - Estuvo en peleas físicas (1=Sí, 2=No)", [1, 2], index=1)
        qn54 = st.selectbox("QN54 - Compañeros amables en escuela (1=Sí, 2=No)", [1, 2], index=0)
        qn56 = st.selectbox("QN56 - Padres comprenden sus problemas (1=Sí, 2=No)", [1, 2], index=0)

    base_row = df[available_features].dropna().iloc[0].copy()
    scenario_changes = {
        "Q1": q1, "Q2": q2,
        "QN22": qn22, "QN23": qn23,
        "QN35": qn35, "QN16": qn16,
        "QN54": qn54, "QN56": qn56,
    }

    if st.button("Predecir riesgo"):
        result = predict_scenario(
            pipeline, base_row, available_features,
            [scenario_changes], predict_proba=True,
        )
        pred = result.loc[1, "prediction"]
        proba_col = "probability_class1"
        proba = result.loc[1, proba_col] if proba_col in result.columns else None
        label = "En riesgo de salud mental" if pred == 1 else "Sin riesgo detectado"
        st.metric("Predicción", label)
        if proba is not None:
            st.metric("Probabilidad de riesgo", f"{proba:.1%}")


def main():
    page = st.sidebar.selectbox("Página", [
        "Resumen del Dataset",
        "Distribución de Variables",
        "Explorador de Correlaciones",
        "Resultados de Modelos",
        "Simulador de Escenarios",
    ])

    df = load_data()

    if page == "Resumen del Dataset":
        render_data_overview(df)
    elif page == "Distribución de Variables":
        render_feature_distributions(df)
    elif page == "Explorador de Correlaciones":
        render_correlation_explorer(df)
    elif page == "Resultados de Modelos":
        render_model_results()
    elif page == "Simulador de Escenarios":
        render_scenario_simulator(df)


main()
