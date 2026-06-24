import pandas as pd
from sklearn.pipeline import Pipeline


def build_scenario(base_row: pd.Series, changes: dict) -> pd.Series:
    """Return a copy of base_row with the specified column values replaced."""
    row = base_row.copy()
    for col, value in changes.items():
        row[col] = value
    return row


def predict_scenario(
    pipeline: Pipeline,
    base_row: pd.Series,
    feature_cols: list[str],
    scenarios: list[dict],
    predict_proba: bool = False,
) -> pd.DataFrame:
    """For each scenario dict, build a modified row and predict.

    Returns a DataFrame with columns: scenario, prediction (and probability if applicable).
    """
    rows_to_predict = [base_row[feature_cols]]
    scenario_labels = ["baseline"]

    for scenario in scenarios:
        modified = build_scenario(base_row, scenario)
        rows_to_predict.append(modified[feature_cols])
        label = ", ".join(f"{k}={v}" for k, v in scenario.items())
        scenario_labels.append(label)

    X = pd.DataFrame(rows_to_predict, columns=feature_cols)
    predictions = pipeline.predict(X)

    result = pd.DataFrame({"scenario": scenario_labels, "prediction": predictions})

    if predict_proba and hasattr(pipeline, "predict_proba"):
        probas = pipeline.predict_proba(X)
        result["probability_class1"] = probas[:, 1].round(4)

    return result


def run_sample_scenarios(df: pd.DataFrame, pipelines: dict, feature_cols: list[str]):
    """Demo: effect of changing Q7 (physical activity days) on overweight prediction."""
    target_key = next((k for k in pipelines if "classification" in k), None)
    if target_key is None:
        print("No classification pipeline found.")
        return

    pipeline = pipelines[target_key]
    base_row = df[feature_cols].dropna().iloc[0]

    scenarios = [
        {"Q7": 1},   # sedentary (1 day active per week)
        {"Q7": 4},   # moderately active
        {"Q7": 7},   # fully active (7 days per week)
    ]

    results = predict_scenario(
        pipeline, base_row, feature_cols, scenarios, predict_proba=True
    )
    print(f"\nScenario simulation for pipeline: {target_key}")
    print(results.to_string(index=False))
    return results
