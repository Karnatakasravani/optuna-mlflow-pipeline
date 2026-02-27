import json
import optuna
import mlflow
import mlflow.xgboost
import mlflow.sklearn
import random
import numpy as np
import time

from math import sqrt
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score

from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from src.data_loader import load_and_split_data
from src.objective import objective


# ======================
# GLOBAL SEEDS
# ======================
random.seed(42)
np.random.seed(42)

# ======================
# GLOBAL CONFIG
# ======================
N_TRIALS = 100
RANDOM_STATE = 42
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def main():

    # ------------------
    # Setup MLflow
    # ------------------
    mlflow.set_tracking_uri("file:./outputs/mlruns")
    mlflow.set_experiment("optuna-xgboost-optimization")

    # ------------------
    # Load Data
    # ------------------
    X_train, X_test, y_train, y_test = load_and_split_data()

    # ------------------
    # Create Optuna Study
    # ------------------
    sampler = TPESampler(seed=42)

    pruner = MedianPruner(
        n_startup_trials=10,
        n_warmup_steps=5
    )

    study = optuna.create_study(
        study_name="xgboost-housing-optimization",
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        storage="sqlite:///optuna_study.db",
        load_if_exists=True
    )

    # ------------------
    # Optimization
    # ------------------
    start_time = time.time()

    def wrapped_objective(trial):
        with mlflow.start_run(nested=True):
            return objective(trial, X_train, y_train)

    study.optimize(
        wrapped_objective,
        n_trials=N_TRIALS,
        n_jobs=2
    )

    optimization_time = time.time() - start_time

    # ------------------
    # Train Best Model
    # ------------------
    best_params = study.best_params
    best_cv_rmse = study.best_value

    import xgboost as xgb

    model = xgb.XGBRegressor(
        **best_params,
        random_state=RANDOM_STATE,
        tree_method="hist",
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    test_mse = mean_squared_error(y_test, preds)
    test_rmse = sqrt(test_mse)
    test_r2 = r2_score(y_test, preds)

    # ------------------
    # Log Final Model to MLflow
    # ------------------
    with mlflow.start_run(run_name="best_model_final"):

        mlflow.set_tag("best_model", "true")

        mlflow.log_params(best_params)

        mlflow.log_metric("test_mse", test_mse)
        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.log_metric("test_r2", test_r2)

        mlflow.sklearn.log_model(model, artifact_path="model")

    # ------------------
    # Save results.json
    # ------------------
    results = {
        "n_trials_completed": len(
            [t for t in study.trials if t.state.name == "COMPLETE"]
        ),
        "n_trials_pruned": len(
            [t for t in study.trials if t.state.name == "PRUNED"]
        ),
        "best_params": best_params,
        "best_cv_rmse": best_cv_rmse,
        "test_rmse": test_rmse,
        "test_r2": test_r2,
        "optimization_time_seconds": optimization_time
    }

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=4)

    print("Optimization complete!")
    print("Best CV RMSE:", best_cv_rmse)
    print("Test RMSE:", test_rmse)
    print("Test R2:", test_r2)


if __name__ == "__main__":
    main()