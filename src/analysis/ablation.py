import pandas as pd

from src.config import (
    CLASSIFICATION_FEATURE_COLS,
    CLASSIFICATION_TARGET_MENTAL_HEALTH,
    CV_FOLDS,
    RANDOM_STATE,
    REPORTS_DIR,
)
from src.data.load import load_raw
from src.features.engineer import add_all_engineered_features
from src.models.train import build_classification_pipeline, cross_validate_model

# Grupos de features para el estudio de ablación del modelo de salud mental.
# Cada grupo agrupa columnas por dominio conceptual según la encuesta GSHS.
FEATURE_GROUPS: dict[str, list[str]] = {
    "demographics": ["Q1", "Q2", "Q3"],
    "diet_nutrition": ["QN6", "QN7", "QN8", "QN9", "QN10"],
    "violence_bullying": ["QN15", "QN16", "QN20"],
    "affective": ["QN22", "QN23"],
    "substance_use": ["QN35", "QN38", "QN39", "QN44", "QN46"],
    "physical_activity": ["QN49", "QN52", "QN53"],
    "social_support": ["QN54", "QN55", "QN56", "QN57"],
}


def leave_one_group_out(
    df: pd.DataFrame,
    target_col: str = CLASSIFICATION_TARGET_MENTAL_HEALTH,
    feature_cols: list[str] = None,
    model_name: str = "logistic",
    scoring: str = "f1",
) -> pd.DataFrame:
    """Quitar cada grupo de features, reentrenar y medir el delta."""
    if feature_cols is None:
        feature_cols = CLASSIFICATION_FEATURE_COLS

    available_features = [c for c in feature_cols if c in df.columns]
    subset = df[available_features + [target_col]].dropna()
    X_full = subset[available_features]
    y = subset[target_col]

    baseline_pipeline = build_classification_pipeline(model_name, available_features, use_smote=True)
    baseline_scores = cross_validate_model(baseline_pipeline, X_full, y, scoring=scoring)
    baseline = baseline_scores.get(f"test_{scoring}_mean", float("nan"))
    print(f"Baseline {scoring}: {baseline:.4f}")

    rows = [{"group_removed": "none (baseline)", "score": baseline, "delta": 0.0}]

    for group_name, group_cols in FEATURE_GROUPS.items():
        reduced_features = [c for c in available_features if c not in group_cols]
        if not reduced_features:
            continue
        X_reduced = subset[reduced_features]
        pipeline = build_classification_pipeline(model_name, reduced_features, use_smote=True)
        scores = cross_validate_model(pipeline, X_reduced, y, scoring=scoring)
        score = scores.get(f"test_{scoring}_mean", float("nan"))
        delta = score - baseline
        rows.append({
            "group_removed": group_name,
            "score": round(score, 4),
            "delta": round(delta, 4),
        })
        print(f"  -{group_name}: {score:.4f} (delta {delta:+.4f})")

    return pd.DataFrame(rows)


def run_ablation_study(df: pd.DataFrame = None) -> pd.DataFrame:
    if df is None:
        df = load_raw()
        df = add_all_engineered_features(df)

    target_col = CLASSIFICATION_TARGET_MENTAL_HEALTH
    results = leave_one_group_out(df, target_col, CLASSIFICATION_FEATURE_COLS)
    out_path = REPORTS_DIR / f"ablation_{target_col}.csv"
    results.to_csv(out_path, index=False)
    print(f"\nResultados de ablation guardados en {out_path}")
    return results


def main():
    run_ablation_study()


if __name__ == "__main__":
    main()
