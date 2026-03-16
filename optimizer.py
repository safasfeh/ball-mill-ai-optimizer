from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
METADATA_PATH = MODELS_DIR / "metadata.json"


def _load_metadata() -> Dict:
    return json.loads(METADATA_PATH.read_text())


def load_models() -> Tuple[Dict[str, object], List[str]]:
    metadata = _load_metadata()
    models = {
        target: joblib.load(MODELS_DIR / f"{target}_model.pkl")
        for target in metadata["targets"]
    }
    return models, metadata["features"]


def predict_point(row: Dict[str, float], models: Dict[str, object], features: List[str]) -> Dict[str, float]:
    X = pd.DataFrame([{k: row[k] for k in features}])
    power_kw = float(models["power_kw"].predict(X)[0])
    p80_um = float(models["p80_um"].predict(X)[0])
    throughput_tph = float(models["throughput_tph"].predict(X)[0])
    sec = power_kw / max(throughput_tph, 1e-6)
    return {
        **row,
        "power_kw": power_kw,
        "p80_um": p80_um,
        "throughput_tph": throughput_tph,
        "specific_energy_kwhpt": sec,
    }


def optimize_ball_mill(
    fixed_inputs: Dict[str, float],
    bounds: Dict[str, Tuple[float, float, float]],
    target_p80_um: float,
    min_throughput_tph: float,
    motor_limit_kw: float,
) -> Tuple[Dict[str, float] | None, pd.DataFrame]:
    models, features = load_models()

    axes = []
    ordered_keys = list(bounds.keys())
    for key in ordered_keys:
        low, high, step = bounds[key]
        axes.append(np.arange(low, high + 1e-9, step))

    candidate_rows = []
    best = None
    best_obj = float("inf")

    for combo in itertools.product(*axes):
        row = dict(fixed_inputs)
        row.update({k: float(v) for k, v in zip(ordered_keys, combo)})
        pred = predict_point(row, models, features)
        pred["feasible"] = bool(
            pred["p80_um"] <= target_p80_um
            and pred["throughput_tph"] >= min_throughput_tph
            and pred["power_kw"] <= motor_limit_kw
        )
        candidate_rows.append(pred)
        if pred["feasible"] and pred["specific_energy_kwhpt"] < best_obj:
            best = pred
            best_obj = pred["specific_energy_kwhpt"]

    results_df = pd.DataFrame(candidate_rows).sort_values("specific_energy_kwhpt", ascending=True)
    return best, results_df
