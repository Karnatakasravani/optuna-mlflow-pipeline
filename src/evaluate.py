import json
from pathlib import Path
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances

import mlflow

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

DB_PATH = OUTPUT_DIR / "optuna_study.db"
RESULTS_JSON = OUTPUT_DIR / "results.json"


def main():

    # Load study from DB
    study = optuna.load_study(
        study_name="xgb_opt_study",
        storage=f"sqlite:///{DB_PATH}"
    )

    # ------------ Save plots --------------

    # Optimization history plot
    fig1 = plot_optimization_history(study)
    fig1.write_image(OUTPUT_DIR / "optimization_history.png")

    # Parameter importance plot
    fig2 = plot_param_importances(study)
    fig2.write_image(OUTPUT_DIR / "param_importance.png")

    # ------------ Save summary JSON ---------
    results = {
        "n_trials": len(study.trials),
        "best_value_rmse": study.best_value,
        "best_params": study.best_params,
    }

    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=4)

    print("Evaluation complete!")
    print("Plots and results.json saved in outputs/")


if __name__ == "__main__":
    main()
