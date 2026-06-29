import numpy as np
import pandas as pd
from pathlib import Path

from src.config import DATA_PATH


def load_raw(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df[df >= np.finfo(np.float64).max] = np.nan
    return df


def get_data_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = pd.DataFrame({
        "dtype": df.dtypes,
        "null_count": df.isnull().sum(),
        "null_pct": df.isnull().mean().round(4),
        "nunique": df.nunique(),
        "min": df.min(numeric_only=True),
        "max": df.max(numeric_only=True),
    })
    return summary.sort_values("null_pct", ascending=False)


def validate_columns(df: pd.DataFrame, expected: list[str]) -> None:
    missing = set(expected) - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")
