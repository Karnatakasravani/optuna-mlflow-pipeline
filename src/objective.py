import optuna
import mlflow
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from math import sqrt


def objective(trial, X_train, y_train, X_val, y_val):
    """
    Optuna objective function defining:
    - hyperparameter search space
    - model training
    - pruning
    - RMSE evaluation
    """

    # ----- Hyperparameter search space -----
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 200),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "random_state": 42,
        "tree_method": "hist",
        "n_jobs": -1,

    }

    # ----- Train model -----
    model = xgb.XGBRegressor(**params)

    model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)


    # ----- Predictions -----
    preds = model.predict(X_val)

    # ----- RMSE metric -----
    rmse = sqrt(mean_squared_error(y_val, preds))

    # ----- Log to MLflow -----
    mlflow.log_params(params)
    mlflow.log_metric("rmse", rmse)

    return rmse
