import os
import json
import joblib
import optuna
import mlflow
import mlflow.sklearn
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from pathlib import Path

from src.data_loader import load_and_split_data
from src.objective import objective


# ======================
# GLOBAL CONFIG
# ======================
N_TRIALS = 3
RANDOM_STATE = 42
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def main():

    # ------------------
    # Setup MLflow
    # ------------------
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("optuna_xgboost_experiment")

    # ------------------
    # Load data
    # ------------------
    X_train, X_test, y_train, y_test = load_and_split_data()

    # Create validation split from training
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=RANDOM_STATE
    )

    # ------------------
    # Create Optuna study
    # ------------------
    study = optuna.create_study(
    study_name="xgb_opt_study",
    direction="minimize",
    storage=f"sqlite:///outputs/optuna_study.db",
    load_if_exists=True
)


    # ------------------
       # Optimization loop
    # ------------------
    def wrapped_objective(trial):
        with mlflow.start_run(nested=True):
            return objective(trial, X_tr, y_tr, X_val, y_val)

    study.optimize(wrapped_objective, n_trials=N_TRIALS)

    # Save study DB
    study_path = OUTPUT_DIR / "optuna_study.db"
    study.trials_dataframe().to_sql(
    "trials_export",
    f"sqlite:///{study_path}",
    if_exists="replace",
    index=False
)

    # ------------------
    # Train best model on FULL TRAINING DATA
    # ------------------
    best_params = study.best_params
    best_rmse = study.best_value

    import xgboost as xgb
    model = xgb.XGBRegressor(
        **best_params,
        random_state=RANDOM_STATE,
        tree_method="hist"
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    test_rmse = sqrt(mean_squared_error(y_test, preds))

    # ------------------
    # Log final model
    # ------------------
    with mlflow.start_run(run_name="best_model_final") as run:
        mlflow.log_params(best_params)
        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.sklearn.log_model(model, artifact_path="model")

    # ------------------
    # Save results.json
    # ------------------
    results = {
        "best_params": best_params,
        "validation_rmse": best_rmse,
        "test_rmse": test_rmse
    }

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=4)

    print("Optimization complete!")
    print("Best RMSE:", best_rmse)
    print("Test RMSE:", test_rmse)


if __name__ == "__main__":
    main()
