import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def build_imputer_pipeline(
    categorical_cols: list[str],
    numeric_cols: list[str],
) -> ColumnTransformer:
    """Return a ColumnTransformer that imputes:
    - categorical columns with mode (most_frequent)
    - numeric columns with median
    """
    transformers = []
    if categorical_cols:
        transformers.append((
            "cat_imputer",
            SimpleImputer(strategy="most_frequent"),
            categorical_cols,
        ))
    if numeric_cols:
        transformers.append((
            "num_imputer",
            SimpleImputer(strategy="median"),
            numeric_cols,
        ))
    return ColumnTransformer(transformers=transformers, remainder="passthrough")


def impute_dataframe(
    df: pd.DataFrame,
    categorical_cols: list[str],
    numeric_cols: list[str],
) -> pd.DataFrame:
    """Fit-transform the imputer and return a clean DataFrame with original column order."""
    ct = build_imputer_pipeline(categorical_cols, numeric_cols)
    all_cols = categorical_cols + numeric_cols
    remaining = [c for c in df.columns if c not in all_cols]
    imputed = ct.fit_transform(df[all_cols + remaining])
    output_cols = all_cols + remaining
    return pd.DataFrame(imputed, columns=output_cols, index=df.index)
