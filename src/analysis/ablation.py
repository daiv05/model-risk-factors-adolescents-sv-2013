import pandas as pd

from src.config import (
    CLASSIFICATION_FEATURE_COLS,
    CLASSIFICATION_TARGET_DEFAULT,
    CV_FOLDS,
    RANDOM_STATE,
    REPORTS_DIR,
)
from src.data.clean import encode_binary_targets
from src.data.load import load_raw
from src.models.train import build_classification_pipeline, cross_validate_model

FEATURE_GROUPS: dict[str, list[str]] = {
    "demographics": ["Q1", "Q2", "Q3"],
    "diet": ["QN10", "QN11", "QN12", "QN13", "QN14"],
    "physical_activity": ["QN7"],
    "mental_health": ["QN26", "QN54", "QN55", "QN57"],
    "substance_use": ["QN35", "QN38", "QN39", "QN46", "QN49", "QN50"],
    "violence": ["QN22", "QN23"],
    "hygiene_sleep": ["QN6", "QN9", "QN16"],
}


def leave_one_group_out(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    model_name: str = "logistic",
    scoring: str = "f1",
) -> pd.DataFrame:
    """Leave-one-group-out ablation: remove each feature group, retrain, measure delta."""
    available_features = [c for c in feature_cols if c in df.columns]
    subset = df[available_features + [target_col]].dropna()
    X_full = subset[available_features]
    y = subset[target_col]

    baseline_pipeline = build_classification_pipeline(model_name, use_smote=True)
    baseline_scores = cross_validate_model(baseline_pipeline, X_full, y, scoring=scoring)
    baseline = baseline_scores.get(f"test_{scoring}_mean", float("nan"))
    print(f"Baseline ({scoring}): {baseline:.4f}")

    rows = [{"group_removed": "none (baseline)", "score": baseline, "delta": 0.0}]

    for group_name, group_cols in FEATURE_GROUPS.items():
        reduced_features = [c for c in available_features if c not in group_cols]
        if not reduced_features:
            continue
        X_reduced = subset[reduced_features]
        pipeline = build_classification_pipeline(model_name, use_smote=True)
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


def run_ablation_study(df: pd.DataFrame = None, target_col: str = CLASSIFICATION_TARGET_DEFAULT):
    if df is None:
        df = load_raw()
    df = encode_binary_targets(df, [target_col])

    results = leave_one_group_out(df, target_col, CLASSIFICATION_FEATURE_COLS)
    out_path = REPORTS_DIR / f"ablation_{target_col}.csv"
    results.to_csv(out_path, index=False)
    print(f"\nAblation results saved to {out_path}")
    return results


def main():
    run_ablation_study()


if __name__ == "__main__":
    main()
