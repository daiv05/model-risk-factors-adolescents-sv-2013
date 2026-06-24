import pandas as pd

from src.config import (
    CLASSIFICATION_FEATURE_COLS,
    CLASSIFICATION_TARGET_DEFAULT,
    CV_FOLDS,
    REGRESSION_FEATURE_COLS,
    REGRESSION_TARGET_HEIGHT,
    REGRESSION_TARGET_WEIGHT,
    REPORTS_DIR,
)
from src.data.clean import encode_binary_targets
from src.data.load import load_raw
from src.models.train import (
    build_classification_pipeline,
    build_regression_pipeline,
    cross_validate_model,
)


def benchmark_all_models(df: pd.DataFrame) -> pd.DataFrame:
    """Evaluate all 4 models with CV. Returns tidy DataFrame of scores."""
    rows = []

    # --- Regression ---
    for target_name, target_col in [
        ("height", REGRESSION_TARGET_HEIGHT),
        ("weight", REGRESSION_TARGET_WEIGHT),
    ]:
        available_features = [c for c in REGRESSION_FEATURE_COLS if c in df.columns]
        if target_col not in df.columns:
            continue
        subset = df[available_features + [target_col]].dropna()
        X = subset[available_features]
        y = subset[target_col]

        for model_name in ["linear", "random_forest"]:
            pipeline = build_regression_pipeline(model_name)
            scores = cross_validate_model(
                pipeline, X, y, scoring=["r2", "neg_mean_absolute_error"]
            )
            rows.append({
                "task": "regression",
                "model": model_name,
                "target": target_name,
                "metric": "r2",
                "mean_score": round(scores.get("test_r2_mean", float("nan")), 4),
                "std_score": round(scores.get("test_r2_std", float("nan")), 4),
            })
            rows.append({
                "task": "regression",
                "model": model_name,
                "target": target_name,
                "metric": "mae",
                "mean_score": round(-scores.get("test_neg_mean_absolute_error_mean", float("nan")), 4),
                "std_score": round(scores.get("test_neg_mean_absolute_error_std", float("nan")), 4),
            })

    # --- Classification ---
    target_col = CLASSIFICATION_TARGET_DEFAULT
    df_cls = encode_binary_targets(df, [target_col])
    available_features = [c for c in CLASSIFICATION_FEATURE_COLS if c in df_cls.columns]
    subset = df_cls[available_features + [target_col]].dropna()
    X = subset[available_features]
    y = subset[target_col]

    for model_name in ["logistic", "random_forest"]:
        pipeline = build_classification_pipeline(model_name, use_smote=True)
        scores = cross_validate_model(pipeline, X, y, scoring=["f1", "roc_auc"])
        for metric in ["f1", "roc_auc"]:
            rows.append({
                "task": "classification",
                "model": model_name,
                "target": target_col,
                "metric": metric,
                "mean_score": round(scores.get(f"test_{metric}_mean", float("nan")), 4),
                "std_score": round(scores.get(f"test_{metric}_std", float("nan")), 4),
            })

    return pd.DataFrame(rows)


def run_benchmark(df: pd.DataFrame = None):
    if df is None:
        df = load_raw()

    print("Running benchmark across all models...")
    results = benchmark_all_models(df)
    print(results.to_string(index=False))

    out_path = REPORTS_DIR / "benchmark_results.csv"
    results.to_csv(out_path, index=False)
    print(f"\nBenchmark saved to {out_path}")
    return results


def main():
    run_benchmark()


if __name__ == "__main__":
    main()
