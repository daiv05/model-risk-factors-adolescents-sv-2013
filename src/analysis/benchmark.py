import pandas as pd

from src.config import (
    CLASSIFICATION_FEATURE_COLS,
    CLASSIFICATION_TARGET_MENTAL_HEALTH,
    CV_FOLDS,
    REGRESSION_FEATURE_COLS,
    REGRESSION_TARGET_BMI,
    REPORTS_DIR,
)
from src.data.load import load_raw
from src.features.engineer import add_all_engineered_features
from src.models.train import (
    build_classification_pipeline,
    build_regression_pipeline,
    cross_validate_model,
)


def benchmark_all_models(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    # --- Regresión: IMC ---
    available_features = [c for c in REGRESSION_FEATURE_COLS if c in df.columns]
    if REGRESSION_TARGET_BMI in df.columns:
        mask = df[REGRESSION_TARGET_BMI].notna()
        X = df.loc[mask, available_features]
        y = df.loc[mask, REGRESSION_TARGET_BMI]

        for model_name in ["linear", "random_forest"]:
            pipeline = build_regression_pipeline(model_name, available_features)
            scores = cross_validate_model(
                pipeline, X, y, scoring=["r2", "neg_mean_absolute_error"]
            )
            rows.append({
                "task": "regression",
                "model": model_name,
                "target": "bmi",
                "metric": "r2",
                "mean_score": round(scores.get("test_r2_mean", float("nan")), 4),
                "std_score": round(scores.get("test_r2_std", float("nan")), 4),
            })
            rows.append({
                "task": "regression",
                "model": model_name,
                "target": "bmi",
                "metric": "mae",
                "mean_score": round(
                    -scores.get("test_neg_mean_absolute_error_mean", float("nan")), 4
                ),
                "std_score": round(
                    scores.get("test_neg_mean_absolute_error_std", float("nan")), 4
                ),
            })

    # --- Clasificación: riesgo de salud mental ---
    target_col = CLASSIFICATION_TARGET_MENTAL_HEALTH
    if target_col in df.columns:
        available_features = [c for c in CLASSIFICATION_FEATURE_COLS if c in df.columns]
        mask = df[target_col].notna()
        X = df.loc[mask, available_features]
        y = df.loc[mask, target_col]

        for model_name in ["logistic", "random_forest"]:
            pipeline = build_classification_pipeline(model_name, available_features, use_smote=True)
            scores = cross_validate_model(
                pipeline, X, y, scoring=["f1", "roc_auc", "balanced_accuracy"]
            )
            for metric in ["f1", "roc_auc", "balanced_accuracy"]:
                rows.append({
                    "task": "classification",
                    "model": model_name,
                    "target": target_col,
                    "metric": metric,
                    "mean_score": round(
                        scores.get(f"test_{metric}_mean", float("nan")), 4
                    ),
                    "std_score": round(
                        scores.get(f"test_{metric}_std", float("nan")), 4
                    ),
                })

    return pd.DataFrame(rows)


def run_benchmark(df: pd.DataFrame = None) -> pd.DataFrame:
    if df is None:
        df = load_raw()
        df = add_all_engineered_features(df)

    print("Corriendo benchmark...")
    results = benchmark_all_models(df)
    print(results.to_string(index=False))

    out_path = REPORTS_DIR / "benchmark_results.csv"
    results.to_csv(out_path, index=False)
    print(f"\nBenchmark guardado en {out_path}")
    return results


def main():
    run_benchmark()


if __name__ == "__main__":
    main()
