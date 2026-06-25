import pandas as pd
from sklearn.pipeline import Pipeline


def build_scenario(base_row: pd.Series, changes: dict) -> pd.Series:
    """Devuelve una copia de base_row con los valores de columna especificados reemplazados."""
    row = base_row.copy()
    for col, value in changes.items():
        row[col] = value
    return row


def predict_scenario(
    pipeline,
    base_row: pd.Series,
    feature_cols: list[str],
    scenarios: list[dict],
    predict_proba: bool = False,
) -> pd.DataFrame:
    """Para cada escenario, construye una fila modificada y predice.

    Devuelve un DataFrame con columnas: scenario, prediction (y probability si aplica).
    El índice 0 es siempre el baseline (fila original sin cambios).
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
    """Demo: efecto de cambiar factores de protección/riesgo sobre la predicción de salud mental.

    Simula tres perfiles contrastantes:
      1. Perfil de riesgo alto: soledad frecuente, sin apoyo parental, consumo de alcohol
      2. Perfil neutro: valores medios
      3. Perfil protegido: apoyo parental alto, sin consumo de sustancias, sin soledad

    Escala OMS: 1=Sí, 2=No para columnas QN.
    """
    target_key = next((k for k in pipelines if "classification" in k), None)
    if target_key is None:
        print("No se encontró pipeline de clasificación.")
        return

    pipeline = pipelines[target_key]
    base_row = df[feature_cols].dropna().iloc[0]

    scenarios = [
        # Riesgo alto: solo, sin dormir bien, con alcohol, sin apoyo familiar
        {"QN22": 1, "QN23": 1, "QN35": 1, "QN54": 2, "QN55": 2, "QN56": 2, "QN57": 2},
        # Neutro: no solo, algo de apoyo
        {"QN22": 2, "QN23": 2, "QN35": 2, "QN54": 1, "QN55": 2, "QN56": 2, "QN57": 2},
        # Protegido: apoyo familiar completo, sin factores de riesgo
        {"QN22": 2, "QN23": 2, "QN35": 2, "QN54": 1, "QN55": 1, "QN56": 1, "QN57": 1},
    ]

    results = predict_scenario(
        pipeline, base_row, feature_cols, scenarios, predict_proba=True
    )
    print(f"\nSimulación de escenarios para: {target_key}")
    print(results.to_string(index=False))
    return results
