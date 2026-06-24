from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline

from src.config import CV_FOLDS, RANDOM_STATE

PARAM_GRIDS: dict[str, dict] = {
    "regression_linear": {},  # no hyperparameters to tune
    "regression_random_forest": {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_split": [2, 5],
        "model__min_samples_leaf": [1, 2],
    },
    "classification_logistic": {
        "model__C": [0.01, 0.1, 1.0, 10.0],
        "model__solver": ["lbfgs", "saga"],
    },
    "classification_random_forest": {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_leaf": [1, 2, 4],
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
    """Fit a hyperparameter search and return the fitted search object.

    search_type: 'grid' uses GridSearchCV; 'random' uses RandomizedSearchCV.
    """
    if not param_grid:
        raise ValueError("param_grid is empty — nothing to tune.")

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
    print(f"Best params: {search.best_params_}")
    print(f"Best CV score: {search.best_score_:.4f}")
    return search
