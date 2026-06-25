import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.config import REPORTS_DIR


def regression_report(pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Devuelve MAE, RMSE y R² para un pipeline de regresión"""
    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(np.mean((y_test.values - y_pred) ** 2)))
    r2 = r2_score(y_test, y_pred)
    return {"mae": round(mae, 4), "rmse": round(rmse, 4), "r2": round(r2, 4)}


def classification_report_extended(
    pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    target_name: str,
) -> dict:
    """Devuelve accuracy, F1, ROC-AUC, precision y recall para un pipeline de clasificación"""
    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    try:
        if hasattr(pipeline, "predict_proba"):
            y_proba = pipeline.predict_proba(X_test)[:, 1]
        else:
            y_proba = pipeline.decision_function(X_test)
        roc_auc = round(roc_auc_score(y_test, y_proba), 4)
    except Exception:
        roc_auc = None

    # F1 de la clase minoritaria (clase 1 = en riesgo)
    f1_minority = round(report.get("1", {}).get("f1-score", float("nan")), 4)

    return {
        "target": target_name,
        "accuracy": round(report["accuracy"], 4),
        "f1_minority_class": f1_minority,
        "f1_weighted": round(report["weighted avg"]["f1-score"], 4),
        "precision_weighted": round(report["weighted avg"]["precision"], 4),
        "recall_weighted": round(report["weighted avg"]["recall"], 4),
        "roc_auc": roc_auc,
    }


def plot_confusion_matrix(
    pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    target_name: str,
    ax=None,
    save: bool = True,
) -> plt.Figure:
    """Matriz de confusión normalizada (heatmap seaborn)"""
    y_pred = pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, normalize="true")
    labels = sorted(y_test.unique())

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    else:
        fig = ax.get_figure()

    sns.heatmap(
        cm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax,
    )
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title(f"Matriz de Confusión - {target_name}")

    if save:
        path = REPORTS_DIR / f"confusion_matrix_{target_name}.png"
        fig.savefig(path, bbox_inches="tight", dpi=150)
        print(f"Guardado: {path}")

    return fig


def _extract_feature_importances(pipeline, feature_names: list[str]):
    """Extrae importancias del estimador final, compatible con sklearn e imblearn Pipeline.

    Devuelve (importances array, names list) alineados.
    El preprocesador usa índices posicionales, así que get_feature_names_out()
    devuelve 'cat__x0_N' / 'num__x0_N'. Los reemplazamos con los nombres originales
    expandidos (OHE puede generar múltiples columnas por feature categórica).
    """
    estimator = pipeline[-1]

    try:
        preprocessor = pipeline.named_steps.get("preprocessor") or pipeline[0]
        raw_names = list(preprocessor.get_feature_names_out())

        # Reconstruir nombres legibles desde los índices posicionales
        readable = []
        for raw in raw_names:
            # formato: "cat__x0_2" - transformer=cat, feature_idx=0, category=2
            # formato: "num__x0"   - transformer=num, feature_idx=0
            parts = raw.split("__", 1)
            if len(parts) != 2:
                readable.append(raw)
                continue
            transformer_name, rest = parts
            # rest es como "x0_2" o "x0"
            idx_part = rest.split("_")[1] if "_" in rest else rest.lstrip("x")
            try:
                feat_idx = int(rest.lstrip("x").split("_")[0])
            except (ValueError, IndexError):
                readable.append(raw)
                continue

            # Recupera el índice original desde el ColumnTransformer
            ct = preprocessor
            for tname, _, cols in ct.transformers_:
                if tname == transformer_name:
                    if feat_idx < len(cols):
                        orig_col_idx = cols[feat_idx]
                        orig_name = feature_names[orig_col_idx] if orig_col_idx < len(feature_names) else raw
                        # Para OHE agrega el valor de categoría al nombre
                        if transformer_name == "cat" and "_" in rest[rest.index("_") + 1:]:
                            cat_val = rest.split("_", 2)[-1] if rest.count("_") >= 2 else ""
                            readable.append(f"{orig_name}={cat_val}")
                        else:
                            readable.append(orig_name)
                    else:
                        readable.append(raw)
                    break
            else:
                readable.append(raw)
        names = readable
    except Exception:
        names = feature_names

    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        importances = np.abs(estimator.coef_).flatten()
    else:
        raise AttributeError("El modelo no tiene feature_importances_ ni coef_")

    n = min(len(importances), len(names))
    return importances[:n], names[:n]


def plot_feature_importance(
    pipeline,
    feature_names: list[str],
    top_n: int = 20,
    model_name: str = "model",
    ax=None,
    save: bool = True,
) -> plt.Figure:
    """Gráfico de barras horizontal con las top_n features más importantes"""
    importances, names = _extract_feature_importances(pipeline, feature_names)

    indices = np.argsort(importances)[-top_n:]
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, max(4, len(indices) * 0.35)))
    else:
        fig = ax.get_figure()

    ax.barh([names[i] for i in indices], [importances[i] for i in indices], color="steelblue")
    ax.set_xlabel("Importancia")
    ax.set_title(f"Importancia de Características - {model_name}")
    plt.tight_layout()

    if save:
        path = REPORTS_DIR / f"feature_importance_{model_name}.png"
        fig.savefig(path, bbox_inches="tight", dpi=150)
        print(f"Guardado: {path}")

    return fig


def save_evaluation_report(results: dict, filename: str = "evaluation.json") -> None:
    """Serializa los resultados de evaluación a JSON en el directorio de reports"""
    path = REPORTS_DIR / filename
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Guardado: {path}")


def main():
    import joblib
    from sklearn.model_selection import train_test_split

    from src.config import (
        CLASSIFICATION_FEATURE_COLS,
        CLASSIFICATION_TARGET_MENTAL_HEALTH,
        MODELS_DIR,
        RANDOM_STATE,
        REGRESSION_FEATURE_COLS,
        REGRESSION_TARGET_BMI,
        TEST_SIZE,
    )
    from src.data.load import load_raw
    from src.features.engineer import add_all_engineered_features

    df = load_raw()
    df = add_all_engineered_features(df)
    results = {}

    # --- Regresión: IMC ---
    available_reg = [c for c in REGRESSION_FEATURE_COLS if c in df.columns]
    mask = df[REGRESSION_TARGET_BMI].notna()
    X_reg = df.loc[mask, available_reg]
    y_reg = df.loc[mask, REGRESSION_TARGET_BMI]
    _, X_test_reg, _, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    for model_name in ["linear", "random_forest"]:
        key = f"regression_{model_name}_bmi"
        model_path = MODELS_DIR / f"{key}.joblib"
        if not model_path.exists():
            print(f"Modelo no encontrado: {model_path}")
            continue
        pipeline = joblib.load(model_path)
        results[key] = regression_report(pipeline, X_test_reg, y_test_reg)
        print(f"{key}: {results[key]}")
        plot_feature_importance(pipeline, available_reg, model_name=key)

    # --- Clasificación: salud mental ---
    target_col = CLASSIFICATION_TARGET_MENTAL_HEALTH
    available_cls = [c for c in CLASSIFICATION_FEATURE_COLS if c in df.columns]
    mask = df[target_col].notna()
    X_cls = df.loc[mask, available_cls]
    y_cls = df.loc[mask, target_col]
    _, X_test_cls, _, y_test_cls = train_test_split(
        X_cls, y_cls, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    for model_name in ["logistic", "random_forest"]:
        key = f"classification_{model_name}_{target_col}"
        model_path = MODELS_DIR / f"{key}.joblib"
        if not model_path.exists():
            print(f"Modelo no encontrado: {model_path}")
            continue
        pipeline = joblib.load(model_path)
        results[key] = classification_report_extended(pipeline, X_test_cls, y_test_cls, key)
        print(f"{key}: {results[key]}")
        plot_confusion_matrix(pipeline, X_test_cls, y_test_cls, key)
        plot_feature_importance(pipeline, available_cls, model_name=key)

    save_evaluation_report(results)


if __name__ == "__main__":
    main()
