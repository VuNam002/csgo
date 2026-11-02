from pathlib import Path

import pandas as pd
from models.RandomForestClassifier import train_random_forest
from models.svm import train_svm
from utils.artifacts import (
    build_run_info,
    create_run_dir,
    save_json,
    save_model,
)
from utils.evaluation import evaluate_model
from utils.preprocessing import split_data

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "csgo.csv"

TEST_SIZE = 0.2
RANDOM_STATE = 42


MODELS = {
    "svm": train_svm,
    "random_forest": train_random_forest,
}


def main():
    data = pd.read_csv(DATA_PATH)
    X_train, X_test, y_train, y_test = split_data(
        data, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    run_dir = create_run_dir(BASE_DIR)
    save_json(
        build_run_info(
            DATA_PATH,
            data,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            models=list(MODELS),
        ),
        run_dir / "run_info.json",
    )

    summary = []
    for name, train_fn in MODELS.items():
        print(f"Training {name}...")
        model = train_fn(X_train, X_test, y_train, y_test)
        metrics = evaluate_model(model, X_test, y_test)

        save_model(model, run_dir / "model" / f"{name}.joblib")
        save_json(metrics, run_dir / f"{name}_metrics.json")

        summary.append({
            "model": name,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "roc_auc": metrics["roc_auc"],
        })

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(run_dir / "summary.csv", index=False)
    print(summary_df.to_string(index=False))
    print(f"\nSaved to: {run_dir}")


if __name__ == "__main__":
    main()
