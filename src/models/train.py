import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    CLASSIFICATION_CATEGORICAL_COLS,
    CLASSIFICATION_FEATURE_COLS,
    CLASSIFICATION_TARGET_MENTAL_HEALTH,
    CV_FOLDS,
    MODELS_DIR,
    RANDOM_STATE,
    REGRESSION_CATEGORICAL_COLS,
    REGRESSION_FEATURE_COLS,
    REGRESSION_TARGET_BMI,
    TEST_SIZE,
)
from src.data.load import load_raw
from src.features.engineer import add_all_engineered_features
from src.models.tune import PARAM_GRIDS, tune_pipeline


def _build_preprocessor(feature_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    """ColumnTransformer con imputer + OHE para categóricas, imputer + scaler para numéricas.

    Usa índices posicionales en lugar de nombres de columna para ser compatible
    tanto con DataFrame como con arrays numpy (cross_validate clona el pipeline
    y puede pasar numpy en algunos casos).

    Categóricas (Q1, Q2, Q3): moda - OneHotEncoder (drop='first').
    Numéricas (Q y QN ordinales): mediana - StandardScaler.
    """
    cat_present = [c for c in categorical_cols if c in feature_cols]
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]

    # Índices posicionales - funcionan con DataFrame y con numpy array
    cat_idx = [feature_cols.index(c) for c in cat_present]
    num_idx = [feature_cols.index(c) for c in numeric_cols]

    transformers = []
    if cat_idx:
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("cat", cat_pipe, cat_idx))
    if num_idx:
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", num_pipe, num_idx))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_regression_pipeline(model_name: str, feature_cols: list[str] = None) -> Pipeline:
    """Pipeline completo: preprocesamiento + modelo de regresión.

    model_name: 'linear' | 'random_forest'
    """
    if feature_cols is None:
        feature_cols = REGRESSION_FEATURE_COLS
    preprocessor = _build_preprocessor(feature_cols, REGRESSION_CATEGORICAL_COLS)

    if model_name == "linear":
        model = LinearRegression()
    elif model_name == "random_forest":
        model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    else:
        raise ValueError(f"Unknown regression model: {model_name}")

    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def build_classification_pipeline(
    model_name: str,
    feature_cols: list[str] = None,
    use_smote: bool = True,
) -> ImbPipeline | Pipeline:
    """Pipeline completo: preprocesamiento + modelo de clasificación.

    model_name: 'logistic' | 'random_forest'
    use_smote: inserta SMOTE antes del modelo usando imblearn.Pipeline para que
               el sobremuestreo ocurra solo en los folds de entrenamiento (evita data leakage).
    """
    if feature_cols is None:
        feature_cols = CLASSIFICATION_FEATURE_COLS
    preprocessor = _build_preprocessor(feature_cols, CLASSIFICATION_CATEGORICAL_COLS)

    if model_name == "logistic":
        model = LogisticRegression(
            class_weight="balanced", max_iter=1000, solver="saga",
            random_state=RANDOM_STATE,
        )
    elif model_name == "random_forest":
        model = RandomForestClassifier(
            class_weight="balanced", n_estimators=100,
            random_state=RANDOM_STATE, n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown classification model: {model_name}")

    if use_smote:
        # imblearn Pipeline garantiza que SMOTE solo aplica en folds de entrenamiento
        return ImbPipeline([
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("model", model),
        ])
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def cross_validate_model(
    pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = CV_FOLDS,
    scoring=None,
) -> dict:
    """Ejecuta cross_validate y devuelve media y desviación estándar de cada métrica."""
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


def train_bmi_models(df: pd.DataFrame) -> dict[str, Pipeline]:
    """Entrena modelos de regresión para predecir IMC.

    Features: hábitos de alimentación y actividad física (sin Q4/Q5).
    Target: 'bmi' (debe estar en df, generado por add_all_engineered_features).
    El preprocesamiento (imputer, OHE, scaler) queda encapsulado dentro del pipeline.
    """
    if REGRESSION_TARGET_BMI not in df.columns:
        raise ValueError(
            f"Target '{REGRESSION_TARGET_BMI}' no encontrado. "
            "Ejecuta add_all_engineered_features(df) antes."
        )

    available_features = [c for c in REGRESSION_FEATURE_COLS if c in df.columns]
    X = df[available_features]
    y = df[REGRESSION_TARGET_BMI]

    mask = y.notna()
    X, y = X[mask], y[mask]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    print(f"\nRegresión - target: {REGRESSION_TARGET_BMI}")
    print(f"  Muestras: {len(y)} (train={len(y_train)}, test={len(y_test)})")
    print(f"  IMC: min={y.min():.1f}, media={y.mean():.1f}, max={y.max():.1f}")

    fitted = {}
    for model_name in ["linear", "random_forest"]:
        key = f"regression_{model_name}_bmi"
        grid_key = f"regression_{model_name}"
        print(f"\n  Entrenando {key} ...")
        pipeline = build_regression_pipeline(model_name, available_features)

        param_grid = PARAM_GRIDS.get(grid_key, {})
        if param_grid:
            print(f"    Tuning hiperparámetros ({len(param_grid)} params)...")
            search = tune_pipeline(
                pipeline, param_grid, X_train, y_train,
                search_type="grid", scoring="neg_mean_absolute_error",
            )
            pipeline = search.best_estimator_
        else:
            cv_scores = cross_validate_model(
                pipeline, X_train, y_train,
                scoring=["r2", "neg_mean_absolute_error", "neg_root_mean_squared_error"]
            )
            print(f"    CV R²:   {cv_scores.get('test_r2_mean', float('nan')):.4f} "
                  f"± {cv_scores.get('test_r2_std', float('nan')):.4f}")
            print(f"    CV MAE:  {-cv_scores.get('test_neg_mean_absolute_error_mean', float('nan')):.4f}")
            print(f"    CV RMSE: {-cv_scores.get('test_neg_root_mean_squared_error_mean', float('nan')):.4f}")
            pipeline.fit(X_train, y_train)

        fitted[key] = pipeline
        joblib.dump(pipeline, MODELS_DIR / f"{key}.joblib")
        print(f"    Guardado - models/{key}.joblib")

    return fitted


def train_mental_health_models(df: pd.DataFrame) -> dict[str, Pipeline]:
    """Entrena modelos de clasificación para predecir riesgo de salud mental.

    Features: factores de riesgo y protección psicosocial (columnas QN + demográficas Q1-Q3).
    Target: 'mental_health_risk' (debe estar en df, generado por add_all_engineered_features).
    Las columnas QN están en escala OMS (1=Sí, 2=No); el modelo aprende directamente
    de estos valores - no se recodifican porque la información cardinal (1 vs 2) es útil.
    El desbalance se maneja con class_weight='balanced' + SMOTE.
    """
    target_col = CLASSIFICATION_TARGET_MENTAL_HEALTH
    if target_col not in df.columns:
        raise ValueError(
            f"Target '{target_col}' no encontrado. "
            "Ejecuta add_all_engineered_features(df) antes."
        )

    available_features = [c for c in CLASSIFICATION_FEATURE_COLS if c in df.columns]
    X = df[available_features]
    y = df[target_col]

    mask = y.notna()
    X, y = X[mask], y[mask].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    class_counts = y.value_counts().sort_index()
    ratio = class_counts.max() / class_counts.min()
    print(f"\nClasificación - target: {target_col}")
    print(f"  Muestras: {len(y)} (train={len(y_train)}, test={len(y_test)})")
    print(f"  Clases: {class_counts.to_dict()}  |  Ratio: {ratio:.1f}:1")

    fitted = {}
    for model_name in ["logistic", "random_forest"]:
        key = f"classification_{model_name}_{target_col}"
        grid_key = f"classification_{model_name}"
        print(f"\n  Entrenando {key} ...")
        pipeline = build_classification_pipeline(model_name, available_features, use_smote=True)

        param_grid = PARAM_GRIDS.get(grid_key, {})
        if param_grid:
            print(f"    Tuning hiperparámetros ({len(param_grid)} params)...")
            search = tune_pipeline(
                pipeline, param_grid, X_train, y_train,
                search_type="grid", scoring="f1",
            )
            pipeline = search.best_estimator_
        else:
            cv_scores = cross_validate_model(
                pipeline, X_train, y_train, scoring=["f1", "roc_auc", "balanced_accuracy"]
            )
            print(f"    CV F1:                {cv_scores.get('test_f1_mean', float('nan')):.4f} "
                  f"± {cv_scores.get('test_f1_std', float('nan')):.4f}")
            print(f"    CV ROC-AUC:           {cv_scores.get('test_roc_auc_mean', float('nan')):.4f} "
                  f"± {cv_scores.get('test_roc_auc_std', float('nan')):.4f}")
            print(f"    CV Balanced Accuracy: {cv_scores.get('test_balanced_accuracy_mean', float('nan')):.4f}")
            pipeline.fit(X_train, y_train)

        fitted[key] = pipeline
        joblib.dump(pipeline, MODELS_DIR / f"{key}.joblib")
        print(f"    Guardado - models/{key}.joblib")

    return fitted


def main():
    print("Cargando datos...")
    df = load_raw()
    print(f"  Shape: {df.shape}")

    print("\nCalculando features derivadas (IMC, mental_health_risk, scores)...")
    df = add_all_engineered_features(df)

    train_bmi_models(df)
    train_mental_health_models(df)
    print("\nTodos los modelos entrenados y guardados.")


if __name__ == "__main__":
    main()
