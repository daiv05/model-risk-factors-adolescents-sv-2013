import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.config import REPORTS_DIR


def regression_report(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Return MAE, RMSE, and R² for a regression pipeline."""
    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(np.mean((y_test.values - y_pred) ** 2)))
    r2 = r2_score(y_test, y_pred)
    return {"mae": round(mae, 4), "rmse": round(rmse, 4), "r2": round(r2, 4)}


def classification_report_extended(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    target_name: str,
) -> dict:
    """Return accuracy, F1, ROC-AUC, precision, recall for a classification pipeline."""
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

    return {
        "target": target_name,
        "accuracy": round(report["accuracy"], 4),
        "f1_weighted": round(report["weighted avg"]["f1-score"], 4),
        "precision_weighted": round(report["weighted avg"]["precision"], 4),
        "recall_weighted": round(report["weighted avg"]["recall"], 4),
        "roc_auc": roc_auc,
    }


def plot_confusion_matrix(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    target_name: str,
    ax=None,
    save: bool = True,
) -> plt.Figure:
    """Plot a normalized confusion matrix."""
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
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {target_name}")

    if save:
        path = REPORTS_DIR / f"confusion_matrix_{target_name}.png"
        fig.savefig(path, bbox_inches="tight", dpi=150)
        print(f"Saved: {path}")

    return fig


def plot_feature_importance(
    pipeline: Pipeline,
    feature_names: list[str],
    top_n: int = 20,
    model_name: str = "model",
    ax=None,
    save: bool = True,
) -> plt.Figure:
    """Plot a horizontal bar chart of the top_n most important features."""
    estimator = pipeline.named_steps.get("model")
    if estimator is None:
        # imblearn pipeline: last step may have a different name
        estimator = pipeline[-1]

    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        importances = np.abs(estimator.coef_).flatten()
    else:
        raise AttributeError("Model has neither feature_importances_ nor coef_.")

    # ColumnTransformer / SMOTE may shift feature count; align best-effort
    n = min(len(importances), len(feature_names))
    importances = importances[:n]
    names = feature_names[:n]

    indices = np.argsort(importances)[-top_n:]
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.35)))
    else:
        fig = ax.get_figure()

    ax.barh([names[i] for i in indices], [importances[i] for i in indices])
    ax.set_xlabel("Importance")
    ax.set_title(f"Feature Importance — {model_name}")
    plt.tight_layout()

    if save:
        path = REPORTS_DIR / f"feature_importance_{model_name}.png"
        fig.savefig(path, bbox_inches="tight", dpi=150)
        print(f"Saved: {path}")

    return fig


def save_evaluation_report(results: dict, filename: str = "evaluation.json") -> None:
    """Serialize evaluation results to JSON in the reports directory."""
    path = REPORTS_DIR / filename
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {path}")


def main():
    import joblib
    from src.config import (
        CLASSIFICATION_FEATURE_COLS,
        CLASSIFICATION_TARGET_DEFAULT,
        MODELS_DIR,
        REGRESSION_FEATURE_COLS,
        REGRESSION_TARGET_HEIGHT,
        REGRESSION_TARGET_WEIGHT,
        TEST_SIZE,
        RANDOM_STATE,
    )
    from src.data.clean import encode_binary_targets
    from src.data.load import load_raw
    from sklearn.model_selection import train_test_split

    df = load_raw()
    results = {}

    for target_name, target_col in [
        ("height", REGRESSION_TARGET_HEIGHT),
        ("weight", REGRESSION_TARGET_WEIGHT),
    ]:
        for model_name in ["linear", "random_forest"]:
            key = f"regression_{model_name}_{target_name}"
            model_path = MODELS_DIR / f"{key}.joblib"
            if not model_path.exists():
                print(f"Model not found: {model_path}")
                continue
            pipeline = joblib.load(model_path)
            available_features = [c for c in REGRESSION_FEATURE_COLS if c in df.columns]
            subset = df[available_features + [target_col]].dropna()
            _, X_test, _, y_test = train_test_split(
                subset[available_features], subset[target_col],
                test_size=TEST_SIZE, random_state=RANDOM_STATE,
            )
            results[key] = regression_report(pipeline, X_test, y_test)
            print(f"{key}: {results[key]}")

    target_col = CLASSIFICATION_TARGET_DEFAULT
    df_cls = encode_binary_targets(df, [target_col])
    available_features = [c for c in CLASSIFICATION_FEATURE_COLS if c in df_cls.columns]
    subset = df_cls[available_features + [target_col]].dropna()
    _, X_test, _, y_test = train_test_split(
        subset[available_features], subset[target_col],
        test_size=TEST_SIZE, random_state=RANDOM_STATE,
    )

    for model_name in ["logistic", "random_forest"]:
        key = f"classification_{model_name}_{target_col}"
        model_path = MODELS_DIR / f"{key}.joblib"
        if not model_path.exists():
            print(f"Model not found: {model_path}")
            continue
        pipeline = joblib.load(model_path)
        results[key] = classification_report_extended(pipeline, X_test, y_test, key)
        print(f"{key}: {results[key]}")
        plot_confusion_matrix(pipeline, X_test, y_test, key)
        plot_feature_importance(pipeline, available_features, model_name=key)

    save_evaluation_report(results)


if __name__ == "__main__":
    main()
