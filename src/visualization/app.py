import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.config import (
    CLASSIFICATION_FEATURE_COLS,
    CLASSIFICATION_TARGET_DEFAULT,
    DATA_PATH,
    MODELS_DIR,
    REPORTS_DIR,
)
from src.data.clean import encode_binary_targets
from src.data.load import load_raw
from src.simulation.scenarios import predict_scenario
from src.visualization.plots import (
    plot_correlation_matrix,
    plot_likert_profiles,
    plot_missing_heatmap,
    plot_target_distribution,
)

st.set_page_config(page_title="Factores de Riesgo — El Salvador GSHS 2013", layout="wide")


@st.cache_data
def load_data() -> pd.DataFrame:
    return load_raw(DATA_PATH)


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
    fig, ax = __import__("matplotlib.pyplot", fromlist=["subplots"]).subplots()
    df[col].dropna().hist(ax=ax, bins=20, edgecolor="black")
    ax.set_title(f"Distribución — {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Frecuencia")
    st.pyplot(fig)


def render_correlation_explorer(df: pd.DataFrame):
    st.header("Explorador de Correlaciones")
    from src.config import CLASSIFICATION_FEATURE_COLS, REGRESSION_FEATURE_COLS
    group = st.selectbox("Grupo de features", ["Regresión", "Clasificación"])
    cols = REGRESSION_FEATURE_COLS if group == "Regresión" else CLASSIFICATION_FEATURE_COLS
    fig = plot_correlation_matrix(df, cols)
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
            st.subheader(f"Matriz de Confusión — {model_key}")
            st.image(str(cm_path))
        if fi_path.exists():
            st.subheader(f"Importancia de Características — {model_key}")
            st.image(str(fi_path))


def render_scenario_simulator(df: pd.DataFrame):
    st.header("Simulador de Escenarios de Riesgo")
    target_col = CLASSIFICATION_TARGET_DEFAULT

    model_path = MODELS_DIR / f"classification_logistic_{target_col}.joblib"
    if not model_path.exists():
        st.warning("Modelo no encontrado. Ejecuta `make train` primero.")
        return

    pipeline = joblib.load(model_path)
    df_cls = encode_binary_targets(df, [target_col])
    available_features = [c for c in CLASSIFICATION_FEATURE_COLS if c in df_cls.columns]

    st.markdown("Ajusta los valores para simular distintos perfiles de riesgo:")

    q1 = st.slider("Q1 — Grupo de edad (1=13 años, 6=≥18 años)", 1, 6, 3)
    q2 = st.selectbox("Q2 — Sexo (1=Masculino, 2=Femenino)", [1, 2])
    q7 = st.slider("Q7 — Días de actividad física por semana", 1, 7, 3)
    q10 = st.slider("Q10 — Porciones de fruta diarias", 1, 8, 3)
    q12 = st.slider("Q12 — Días con bebidas azucaradas", 1, 5, 2)

    base_row = df_cls[available_features].dropna().iloc[0].copy()
    scenario_changes = {"Q1": q1, "Q2": q2, "Q7": q7, "Q10": q10, "Q12": q12}

    if st.button("Predecir"):
        result = predict_scenario(
            pipeline, base_row, available_features,
            [scenario_changes], predict_proba=True,
        )
        pred = result.loc[1, "prediction"]
        proba = result.loc[1, "probability_class1"] if "probability_class1" in result.columns else "N/A"
        label = "Sobrepeso (1)" if pred == 1 else "No sobrepeso (0)"
        st.metric("Predicción", label)
        st.metric("Probabilidad clase 1", f"{proba:.2%}" if isinstance(proba, float) else proba)


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
