from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline

from src.config import CV_FOLDS, RANDOM_STATE

# ---------------------------------------------------------------------------
# Grids de hiperparámetros
# ---------------------------------------------------------------------------
# Nota sobre prefijos:
#   "preprocessor__"  - paso ColumnTransformer dentro del pipeline
#   "model__"         - estimador final (LinearRegression, RF, LR)
#   "smote__"         - SMOTE step en imblearn Pipeline (solo clasificación)
#
# El preprocesador raramente se tunea; los grids se enfocan en el modelo.

PARAM_GRIDS: dict[str, dict] = {
    "regression_linear": {},

    "regression_random_forest": {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_split": [2, 5],
        "model__min_samples_leaf": [1, 2],
        "model__max_features": ["sqrt", "log2"],
    },

    "classification_logistic": {
        "model__C": [0.01, 0.1, 1.0, 10.0],
        "model__l1_ratio": [0.0, 0.5, 1.0],
    },

    "classification_random_forest": {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2"],
    },
}


def tune_pipeline(
    pipeline: Pipeline,
    param_grid: dict,
    X_train,
    y_train,
    search_type: str = "grid",
    cv: int = CV_FOLDS,
    n_iter: int = 20,
    scoring: str | None = None,
) -> GridSearchCV | RandomizedSearchCV:
    if not param_grid:
        raise ValueError("param_grid está vacío - no hay nada que tunear")

    if search_type == "random":
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=1,
        )
    else:
        search = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=1,
        )

    search.fit(X_train, y_train)
    print(f"Mejores parámetros: {search.best_params_}")
    print(f"Mejor CV score:     {search.best_score_:.4f}")
    return search
