import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    CLASSIFICATION_ALL_TARGETS,
    CLASSIFICATION_FEATURE_COLS,
    CLASSIFICATION_TARGET_DEFAULT,
    CV_FOLDS,
    MODELS_DIR,
    RANDOM_STATE,
    REGRESSION_FEATURE_COLS,
    REGRESSION_TARGET_HEIGHT,
    REGRESSION_TARGET_WEIGHT,
    TEST_SIZE,
)
from src.data.clean import encode_binary_targets
from src.data.load import load_raw


def build_regression_pipeline(model_name: str) -> Pipeline:
    """Build a regression pipeline.

    model_name: 'linear' | 'random_forest'
    """
    if model_name == "linear":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ])
    elif model_name == "random_forest":
        return Pipeline([
            ("model", RandomForestRegressor(
                n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
            )),
        ])
    raise ValueError(f"Unknown regression model: {model_name}")


def build_classification_pipeline(
    model_name: str,
    use_smote: bool = True,
) -> ImbPipeline | Pipeline:
    """Build a classification pipeline.

    model_name: 'logistic' | 'random_forest'
    use_smote: wrap pipeline with SMOTE (applied only inside CV training folds)
    """
    if model_name == "logistic":
        steps = [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=RANDOM_STATE,
            )),
        ]
    elif model_name == "random_forest":
        steps = [
            ("model", RandomForestClassifier(
                class_weight="balanced",
                n_estimators=100,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]
    else:
        raise ValueError(f"Unknown classification model: {model_name}")

    if use_smote:
        # imblearn Pipeline ensures SMOTE runs only on training folds
        return ImbPipeline([("smote", SMOTE(random_state=RANDOM_STATE))] + steps)
    return Pipeline(steps)


def cross_validate_model(
    pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = CV_FOLDS,
    scoring=None,
) -> dict:
    """Run cross_validate and return mean/std of train and test scores."""
    results = cross_validate(
        pipeline, X, y, cv=cv, scoring=scoring,
        return_train_score=True, n_jobs=-1,
    )
    summary = {}
    for key, values in results.items():
        if key.startswith("test_") or key.startswith("train_"):
            summary[f"{key}_mean"] = float(np.mean(values))
            summary[f"{key}_std"] = float(np.std(values))
    return summary


def train_all_regression_models(df: pd.DataFrame) -> dict[str, Pipeline]:
    """Train LinearRegression and RandomForestRegressor. Returns fitted pipelines."""
    fitted = {}
    for target_name, target_col in [
        ("height", REGRESSION_TARGET_HEIGHT),
        ("weight", REGRESSION_TARGET_WEIGHT),
    ]:
        available_features = [c for c in REGRESSION_FEATURE_COLS if c in df.columns]
        available_target = target_col if target_col in df.columns else None
        if available_target is None:
            print(f"Target {target_col} not found, skipping.")
            continue

        subset = df[available_features + [available_target]].dropna()
        X = subset[available_features]
        y = subset[available_target]

        for model_name in ["linear", "random_forest"]:
            key = f"regression_{model_name}_{target_name}"
            print(f"\nTraining {key} ...")
            pipeline = build_regression_pipeline(model_name)
            cv_scores = cross_validate_model(
                pipeline, X, y, scoring=["r2", "neg_mean_absolute_error"]
            )
            print(f"  CV R2: {cv_scores.get('test_r2_mean', 'n/a'):.4f}")
            pipeline.fit(X, y)
            fitted[key] = pipeline
            joblib.dump(pipeline, MODELS_DIR / f"{key}.joblib")
            print(f"  Saved to models/{key}.joblib")

    return fitted


def train_all_classification_models(
    df: pd.DataFrame,
    target_col: str = CLASSIFICATION_TARGET_DEFAULT,
) -> dict[str, Pipeline]:
    """Train LogisticRegression and RandomForestClassifier for a binary target."""
    available_features = [c for c in CLASSIFICATION_FEATURE_COLS if c in df.columns]
    df = encode_binary_targets(df, [target_col])

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not in DataFrame.")

    subset = df[available_features + [target_col]].dropna()
    X = subset[available_features]
    y = subset[target_col]

    fitted = {}
    for model_name in ["logistic", "random_forest"]:
        key = f"classification_{model_name}_{target_col}"
        print(f"\nTraining {key} ...")
        pipeline = build_classification_pipeline(model_name, use_smote=True)
        cv_scores = cross_validate_model(
            pipeline, X, y, scoring=["f1", "roc_auc"]
        )
        print(f"  CV F1:      {cv_scores.get('test_f1_mean', 'n/a'):.4f}")
        print(f"  CV ROC-AUC: {cv_scores.get('test_roc_auc_mean', 'n/a'):.4f}")
        pipeline.fit(X, y)
        fitted[key] = pipeline
        joblib.dump(pipeline, MODELS_DIR / f"{key}.joblib")
        print(f"  Saved to models/{key}.joblib")

    return fitted


def main():
    print("Loading data...")
    df = load_raw()
    print(f"  Shape: {df.shape}")
    train_all_regression_models(df)
    train_all_classification_models(df)
    print("\nAll models trained and saved.")


if __name__ == "__main__":
    main()
