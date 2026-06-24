import pandas as pd


def compute_bmi(df: pd.DataFrame) -> pd.DataFrame:
    """Derive bmi = Q5 (kg) / Q4 (m)^2. Requires both columns to be present."""
    df = df.copy()
    if "Q4" in df.columns and "Q5" in df.columns:
        df["bmi"] = df["Q5"] / (df["Q4"] ** 2)
    return df


def compute_dietary_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    """Sum of 4 binary dietary risk indicators (0-4 scale).

    High score = higher dietary risk:
      Q12 >= 4: bebidas azucaradas frecuentes
      Q10 <= 2: bajo consumo de fruta
      Q11 <= 2: bajo consumo de verdura
      Q14 <= 2: pocos días con desayuno
    """
    df = df.copy()
    score = pd.Series(0, index=df.index)
    if "Q12" in df.columns:
        score += (df["Q12"] >= 4).astype(int)
    if "Q10" in df.columns:
        score += (df["Q10"] <= 2).astype(int)
    if "Q11" in df.columns:
        score += (df["Q11"] <= 2).astype(int)
    if "Q14" in df.columns:
        score += (df["Q14"] <= 2).astype(int)
    df["dietary_risk_score"] = score
    return df


def compute_substance_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    """Sum of 3 binary substance use indicators (0-3 scale)."""
    df = df.copy()
    score = pd.Series(0, index=df.index)
    for col in ["QN35", "QN38", "QN46"]:
        if col in df.columns:
            score += df[col].fillna(0).astype(int)
    df["substance_risk_score"] = score
    return df


def compute_mental_health_score(df: pd.DataFrame) -> pd.DataFrame:
    """Sum of 4 binary mental health risk indicators (0-4 scale)."""
    df = df.copy()
    score = pd.Series(0, index=df.index)
    for col in ["QN26", "QN54", "QN55", "QN57"]:
        if col in df.columns:
            score += df[col].fillna(0).astype(int)
    df["mental_health_score"] = score
    return df


def compute_violence_exposure_score(df: pd.DataFrame) -> pd.DataFrame:
    """Sum of 2 binary violence exposure indicators (0-2 scale)."""
    df = df.copy()
    score = pd.Series(0, index=df.index)
    for col in ["QN22", "QN23"]:
        if col in df.columns:
            score += df[col].fillna(0).astype(int)
    df["violence_exposure_score"] = score
    return df


def get_engineered_feature_names() -> list[str]:
    return ["bmi", "dietary_risk_score", "substance_risk_score",
            "mental_health_score", "violence_exposure_score"]


def add_all_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = compute_bmi(df)
    df = compute_dietary_risk_score(df)
    df = compute_substance_risk_score(df)
    df = compute_mental_health_score(df)
    df = compute_violence_exposure_score(df)
    return df
