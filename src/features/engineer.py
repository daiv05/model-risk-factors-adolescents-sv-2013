import numpy as np
import pandas as pd


def compute_bmi(df: pd.DataFrame) -> pd.DataFrame:
    """Derive bmi = Q5 (kg) / Q4 (m)^2.

    Q4 = altura en metros, Q5 = peso en kilogramos.
    BMI se usa como TARGET de regresión. Q4/Q5 no deben usarse como features.
    """
    df = df.copy()
    if "Q4" in df.columns and "Q5" in df.columns:
        df["bmi"] = df["Q5"] / (df["Q4"] ** 2)
    return df


def compute_mental_health_risk(df: pd.DataFrame) -> pd.DataFrame:
    """Crear variable binaria de riesgo grave de salud mental (suicidalidad).

    Un estudiante se clasifica en riesgo (1) si presenta CUALQUIERA de los tres
    indicadores del continuo de suicidalidad (valor 1 en la escala OMS 1=Sí, 2=No):
      QN24 - consideró seriamente el suicidio en los últimos 12 meses (ideación)
      QN25 - hizo un plan sobre cómo intentar suicidarse (plan)
      QN26 - intentó suicidarse en los últimos 12 meses (intento)

    Estos tres ítems forman un continuo clínico (ideación - plan - intento) con
    correlaciones internas 0.6-0.7, por lo que se combinan con OR lógico: la
    presencia de cualquiera señala riesgo grave. Es una definición clínicamente
    interpretable y no mezcla dominios (ej. soledad o insomnio, que quedan como
    posibles predictores en lugar de formar parte del target).

    Como estas 3 columnas definen el target, se EXCLUYEN de las features
    (ver CLASSIFICATION_FEATURE_COLS en config.py) para no filtrar la respuesta.
    """
    df = df.copy()
    suicidality_cols = ["QN24", "QN25", "QN26"]
    available = [c for c in suicidality_cols if c in df.columns]
    if not available:
        return df

    sub = df[available]
    any_yes = (sub == 1).any(axis=1)
    all_na = sub.isna().all(axis=1)
    has_any_na = sub.isna().any(axis=1)
    risk = any_yes.astype("float")
    risk[all_na] = np.nan
    risk[(~any_yes) & has_any_na] = np.nan
    df["mental_health_risk"] = risk
    return df


def compute_dietary_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    """Score de riesgo dietético (0-4): mayor puntaje = mayor riesgo.

    Componentes (escala original Q):
      Q9  >= 4 : bebidas azucaradas >=2 veces/día
      Q7  <= 2 : fruta escasa (< 1 vez/día)
      Q8  <= 2 : verdura escasa (< 1 vez/día)
      Q10 >= 3 : comida rápida >=2 días en la semana
    """
    df = df.copy()
    score = pd.Series(0, index=df.index)
    if "Q9" in df.columns:
        score += (df["Q9"] >= 4).astype(int)
    if "Q7" in df.columns:
        score += (df["Q7"] <= 2).astype(int)
    if "Q8" in df.columns:
        score += (df["Q8"] <= 2).astype(int)
    if "Q10" in df.columns:
        score += (df["Q10"] >= 3).astype(int)
    df["dietary_risk_score"] = score
    return df


def compute_physical_activity_score(df: pd.DataFrame) -> pd.DataFrame:
    """Score de actividad física (0-3): mayor puntaje = más activo.

    Componentes (escala original Q):
      Q49 >= 5 : activo >=60 min en 5+ días de la semana
      Q50 >= 5 : camina/bicicleta al colegio 5+ días
      Q51 >= 3 : educación física 3+ días/semana
    """
    df = df.copy()
    score = pd.Series(0, index=df.index)
    if "Q49" in df.columns:
        score += (df["Q49"] >= 5).astype(int)
    if "Q50" in df.columns:
        score += (df["Q50"] >= 5).astype(int)
    if "Q51" in df.columns:
        score += (df["Q51"] >= 3).astype(int)
    df["physical_activity_score"] = score
    return df


def compute_substance_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    """Score de riesgo por consumo de sustancias (0-3).

    Componentes (QN ya dicotomizados. 1=Sí, 2=No en escala OMS):
      QN35 - consumió alcohol en últimos 30 días
      QN38 - se emborrachó alguna vez
      QN39 - tuvo problemas por el alcohol
    """
    df = df.copy()
    score = pd.Series(0, index=df.index)
    for col in ["QN35", "QN38", "QN39"]:
        if col in df.columns:
            score += (df[col] == 1).astype(int)
    df["substance_risk_score"] = score
    return df


def compute_social_support_score(df: pd.DataFrame) -> pd.DataFrame:
    """Score de apoyo social/familiar (0-4): mayor puntaje = más apoyo.

    Componentes (QN. 1=Sí, 2=No en escala OMS):
      QN54 - compañeros amables y serviciales
      QN55 - padres revisan tareas
      QN56 - padres comprenden problemas
      QN57 - padres saben qué hace en tiempo libre
    """
    df = df.copy()
    score = pd.Series(0, index=df.index)
    for col in ["QN54", "QN55", "QN56", "QN57"]:
        if col in df.columns:
            score += (df[col] == 1).astype(int)
    df["social_support_score"] = score
    return df


def get_engineered_feature_names() -> list[str]:
    return [
        "bmi",
        "mental_health_risk",
        "dietary_risk_score",
        "physical_activity_score",
        "substance_risk_score",
        "social_support_score",
    ]


def add_all_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = compute_bmi(df)
    df = compute_mental_health_risk(df)
    df = compute_dietary_risk_score(df)
    df = compute_physical_activity_score(df)
    df = compute_substance_risk_score(df)
    df = compute_social_support_score(df)
    return df
