from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass(frozen=True)
class CircuitConstants:
    motor_limit_kw: float = 1500.0
    target_p80_default_um: float = 150.0
    min_throughput_default_tph: float = 20.0


def _clip(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def compute_critical_speed_rpm(mill_diameter_m: float) -> float:
    """Approximate critical speed for a ball mill."""
    return 42.3 / math.sqrt(max(mill_diameter_m, 0.1))


def simulate_ball_mill_point(inputs: Dict[str, float], rng: np.random.Generator | None = None) -> Dict[str, float]:
    """Synthetic but physically guided process response for a ball-mill circuit.

    Parameters
    ----------
    inputs : dict
        Expected keys:
        - mill_speed_pct
        - ball_filling_pct
        - feed_rate_tph
        - solids_pct
        - bond_work_index
        - cyclone_pressure_kpa
        - feed_f80_um
        - mill_diameter_m
        - mill_length_m
        - liner_factor
    rng : np.random.Generator, optional
        Adds mild process noise when provided.
    """
    speed = inputs["mill_speed_pct"]
    filling = inputs["ball_filling_pct"]
    feed_rate = inputs["feed_rate_tph"]
    solids = inputs["solids_pct"]
    bwi = inputs["bond_work_index"]
    cyclone = inputs["cyclone_pressure_kpa"]
    f80 = inputs["feed_f80_um"]
    diameter = inputs["mill_diameter_m"]
    length = inputs["mill_length_m"]
    liner = inputs["liner_factor"]

    # Dimension factor: larger mills draw more power and support more throughput.
    dim_factor = (diameter ** 2.3) * (length ** 0.7)

    # Speed effect: power rises toward an optimum zone then weakens.
    speed_eff = 0.9 + 0.018 * (speed - 68.0) - 0.00035 * (speed - 75.0) ** 2
    speed_eff = _clip(speed_eff, 0.65, 1.25)

    # Filling effect: moderate-to-high filling raises power and breakage.
    filling_eff = 0.84 + 0.022 * (filling - 26.0) - 0.00022 * (filling - 34.0) ** 2
    filling_eff = _clip(filling_eff, 0.65, 1.28)

    # Slurry effect: too low or too high solids reduces performance.
    solids_eff = 1.00 - 0.00095 * (solids - 70.0) ** 2 + 0.002 * (solids - 70.0)
    solids_eff = _clip(solids_eff, 0.78, 1.10)

    # Cyclone pressure influences classification sharpness and circulating load proxy.
    cyclone_eff = 0.96 + 0.0018 * (cyclone - 110.0) - 0.000012 * (cyclone - 120.0) ** 2
    cyclone_eff = _clip(cyclone_eff, 0.82, 1.10)

    # Ore hardness and feed size penalties.
    hardness_factor = 0.72 + 0.038 * bwi
    feed_factor = 0.74 + 0.00018 * f80

    # Base power draw (kW): simplified, monotonic in the expected directions.
    power_kw = 55.0 * dim_factor * speed_eff * filling_eff * (0.86 + 0.14 * solids_eff)
    power_kw *= (0.78 + 0.024 * bwi) * (0.87 + 0.0012 * feed_rate)
    power_kw *= (0.96 + 0.05 * liner)
    power_kw = _clip(power_kw, 180.0, 1500.0)

    # Grinding intensity: proxy for how much size reduction is achieved.
    grinding_intensity = (
        power_kw
        * solids_eff
        * cyclone_eff
        * filling_eff
        / max(feed_rate * hardness_factor * feed_factor, 1e-6)
    )

    # Product size P80 (um): reduced by better grinding intensity and finer classification.
    p80_um = f80 * (0.23 / max(grinding_intensity, 1e-6) ** 0.42)
    p80_um *= 1.04 - 0.0018 * (cyclone - 110.0)
    p80_um *= 1.02 + 0.0045 * max(0.0, solids - 74.0)
    p80_um = _clip(p80_um, 60.0, 420.0)

    # Throughput response: depends on feed, available power, and ore hardness.
    throughput_tph = feed_rate * (0.93 + 0.0035 * (speed - 70.0)) * (0.95 + 0.0022 * (solids - 68.0))
    throughput_tph *= (0.98 + 0.0017 * (cyclone - 110.0))
    throughput_tph *= (1.02 - 0.018 * max(0.0, bwi - 14.0))
    throughput_tph = _clip(throughput_tph, 8.0, 55.0)

    specific_energy = power_kw / max(throughput_tph, 1e-6)

    if rng is not None:
        power_kw *= float(rng.normal(1.0, 0.025))
        p80_um *= float(rng.normal(1.0, 0.035))
        throughput_tph *= float(rng.normal(1.0, 0.02))
        specific_energy = power_kw / max(throughput_tph, 1e-6)

    return {
        **inputs,
        "power_kw": round(float(power_kw), 3),
        "p80_um": round(float(p80_um), 3),
        "throughput_tph": round(float(throughput_tph), 3),
        "specific_energy_kwhpt": round(float(specific_energy), 4),
    }
