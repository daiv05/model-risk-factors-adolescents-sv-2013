import pandas as pd

from src.config import HIGH_MISSINGNESS_THRESHOLD


def drop_high_missingness_cols(
    df: pd.DataFrame,
    threshold: float = HIGH_MISSINGNESS_THRESHOLD,
) -> pd.DataFrame:
    """Drop columns where the fraction of nulls exceeds threshold."""
    null_frac = df.isnull().mean()
    cols_to_drop = null_frac[null_frac > threshold].index.tolist()
    if cols_to_drop:
        print(f"Dropping {len(cols_to_drop)} high-missingness columns: {cols_to_drop}")
    return df.drop(columns=cols_to_drop)


def flag_extreme_outliers(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Add boolean <col>_outlier columns using IQR * 3 for the given numeric columns."""
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 3 * iqr
        upper = q3 + 3 * iqr
        df[f"{col}_outlier"] = (df[col] < lower) | (df[col] > upper)
    return df


def encode_binary_targets(df: pd.DataFrame, target_cols: list[str]) -> pd.DataFrame:
    """Recode WHO binary convention (1=yes, 2=no) to sklearn convention (1=yes, 0=no).

    WHO GSHS encodes binary responses as 1/2; scikit-learn classifiers expect 0/1.
    Only recodes columns that actually contain {1, 2} (or subsets thereof).
    """
    df = df.copy()
    for col in target_cols:
        if col not in df.columns:
            continue
        unique_vals = set(df[col].dropna().unique())
        if unique_vals <= {1.0, 2.0}:
            df[col] = df[col].map({1.0: 1, 2.0: 0})
    return df
