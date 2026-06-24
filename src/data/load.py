import numpy as np
import pandas as pd
from pathlib import Path

from src.config import DATA_PATH, SENTINEL_VALUE


def load_raw(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load CSV and replace WHO GSHS sentinel value with NaN.

    The sentinel 1.79769313486232e+308 (IEEE 754 float max) marks
    skipped/not-applicable responses in the WHO GSHS software export.
    It must be replaced before any arithmetic or imputation.
    """
    df = pd.read_csv(path)
    df.replace(SENTINEL_VALUE, np.nan, inplace=True)
    return df


def get_data_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-column statistics useful for initial EDA."""
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
    """Assert all expected columns are present in the DataFrame."""
    missing = set(expected) - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")
