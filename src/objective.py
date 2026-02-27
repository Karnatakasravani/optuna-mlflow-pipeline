import optuna
import numpy as np
import mlflow

from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


def objective(trial, X_train, y_train):

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
    }

    model = XGBRegressor(
        **params,
        random_state=42,
        n_jobs=-1,
        tree_method="hist"
    )

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    rmse_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):

        X_tr = X_train.iloc[train_idx]
        X_val = X_train.iloc[val_idx]
        y_tr = y_train.iloc[train_idx]
        y_val = y_train.iloc[val_idx]

        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)

        rmse = np.sqrt(mean_squared_error(y_val, preds))
        rmse_scores.append(rmse)

        trial.report(rmse, step=fold)

        if trial.should_prune():
            mlflow.set_tag("trial_state", "pruned")
            raise optuna.TrialPruned()

    mean_rmse = np.mean(rmse_scores)

    mlflow.log_params(params)
    mlflow.log_metric("cv_rmse", mean_rmse)
    mlflow.log_param("trial_number", trial.number)
    mlflow.set_tag("trial_state", "completed")

    return mean_rmse