import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_missing_heatmap(df: pd.DataFrame, figsize: tuple = (14, 6)) -> plt.Figure:
    """Heatmap of null values across all columns (yellow = missing)."""
    fig, ax = plt.subplots(figsize=figsize)
    missing = df.isnull()
    # Only plot columns with at least some missing data for readability
    cols_with_missing = missing.columns[missing.any()].tolist()
    if not cols_with_missing:
        ax.text(0.5, 0.5, "No missing values", ha="center", va="center")
        return fig
    sns.heatmap(missing[cols_with_missing].T, cbar=False, ax=ax, cmap="viridis")
    ax.set_title("Missing Value Pattern")
    ax.set_xlabel("Row index")
    ax.set_ylabel("Column")
    plt.tight_layout()
    return fig


def plot_target_distribution(df: pd.DataFrame, target_col: str) -> plt.Figure:
    """Bar chart of value counts for a binary (0/1) target column."""
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df[target_col].value_counts().sort_index()
    ax.bar(counts.index.astype(str), counts.values, color=["#4C72B0", "#DD8452"])
    ax.set_title(f"Class Distribution — {target_col}")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    for i, (label, val) in enumerate(zip(counts.index, counts.values)):
        ax.text(i, val + 5, str(val), ha="center")
    plt.tight_layout()
    return fig


def plot_correlation_matrix(df: pd.DataFrame, cols: list[str]) -> plt.Figure:
    """Spearman correlation heatmap for the given columns."""
    available = [c for c in cols if c in df.columns]
    corr = df[available].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(max(6, len(available) * 0.5), max(5, len(available) * 0.45)))
    sns.heatmap(
        corr, annot=len(available) <= 20, fmt=".2f",
        cmap="RdBu_r", center=0, ax=ax, linewidths=0.3,
    )
    ax.set_title("Spearman Correlation Matrix")
    plt.tight_layout()
    return fig


def plot_likert_profiles(
    df: pd.DataFrame,
    col_group: list[str],
    title: str = "Response Distributions",
) -> plt.Figure:
    """Stacked bar chart showing response frequency for a group of Likert columns."""
    available = [c for c in col_group if c in df.columns]
    melted = df[available].apply(lambda s: s.value_counts(normalize=True)).T.fillna(0)
    melted = melted.sort_index(axis=1)

    fig, ax = plt.subplots(figsize=(10, max(4, len(available) * 0.5)))
    bottom = pd.Series(0.0, index=melted.index)
    cmap = plt.get_cmap("tab10")
    for i, col in enumerate(melted.columns):
        ax.barh(melted.index, melted[col], left=bottom, label=str(col), color=cmap(i))
        bottom += melted[col]

    ax.set_xlabel("Proportion")
    ax.set_title(title)
    ax.legend(title="Response", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    return fig
