from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from simulator import simulate_ball_mill_point


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
DATASET_PATH = DATA_DIR / "ball_mill_demo_data.csv"
METADATA_PATH = MODELS_DIR / "metadata.json"

FEATURES: List[str] = [
    "mill_speed_pct",
    "ball_filling_pct",
    "feed_rate_tph",
    "solids_pct",
    "bond_work_index",
    "cyclone_pressure_kpa",
    "feed_f80_um",
    "mill_diameter_m",
    "mill_length_m",
    "liner_factor",
]

TARGETS = ["power_kw", "p80_um", "throughput_tph"]


def generate_demo_dataset(n_samples: int = 2500, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    rows = []
    for _ in range(n_samples):
        inputs = {
            "mill_speed_pct": float(rng.uniform(65, 82)),
            "ball_filling_pct": float(rng.uniform(24, 40)),
            "feed_rate_tph": float(rng.uniform(12, 40)),
            "solids_pct": float(rng.uniform(60, 78)),
            "bond_work_index": float(rng.uniform(10, 19)),
            "cyclone_pressure_kpa": float(rng.uniform(80, 160)),
            "feed_f80_um": float(rng.uniform(800, 2600)),
            "mill_diameter_m": float(rng.uniform(2.7, 4.5)),
            "mill_length_m": float(rng.uniform(3.2, 6.0)),
            "liner_factor": float(rng.uniform(0.85, 1.10)),
        }
        rows.append(simulate_ball_mill_point(inputs, rng=rng))
    return pd.DataFrame(rows)


def _train_single_model(X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=14,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_and_save_models(df: pd.DataFrame, random_state: int = 42) -> Dict[str, Dict[str, float]]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATASET_PATH, index=False)

    X = df[FEATURES]
    metrics: Dict[str, Dict[str, float]] = {}

    split = train_test_split(X, df[TARGETS], test_size=0.2, random_state=random_state)
    X_train, X_test, y_train_all, y_test_all = split

    for target in TARGETS:
        model = _train_single_model(X_train, y_train_all[target], random_state=random_state)
        preds = model.predict(X_test)
        metrics[target] = {
            "r2": float(r2_score(y_test_all[target], preds)),
            "mae": float(mean_absolute_error(y_test_all[target], preds)),
        }
        joblib.dump(model, MODELS_DIR / f"{target}_model.pkl")

    metadata = {
        "features": FEATURES,
        "targets": TARGETS,
        "metrics": metrics,
        "dataset_path": str(DATASET_PATH),
        "n_rows": int(len(df)),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    return metrics


if __name__ == "__main__":
    dataset = generate_demo_dataset(n_samples=2500, random_state=42)
    results = train_and_save_models(dataset, random_state=42)
    print("Dataset saved to:", DATASET_PATH)
    print("Training complete.")
    print(json.dumps(results, indent=2))
