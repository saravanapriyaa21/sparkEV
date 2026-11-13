import json
import os

import numpy as np


# ============================================================
# MODEL 4 - ENERGY CONSUMPTION AND ARRIVAL SOC
#
# Two Elastic Net models (standard scaler + linear weights):
#
#   energy          : energy consumption of a trip [kWh/km]
#                     from the 23 Stable-23 trip features
#   arrival_soc_50  : battery SOC at arrival (0-1) from what is
#                     observed in the first 50 % of the trip
#
# The ESP32 runs both. This module gives the same result on the
# server, for the /predict test endpoint.
#
# Source: notebooks/module4_energy_and_soc/
# Export: esp32/tools/export_model4_models.py
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT, "models", "model4_energy_arrival_soc.json"
)

DEMO_TRIPS_PATH = os.path.join(
    PROJECT_ROOT, "esp32", "tools", "vectors", "model4_demo_trips.json"
)

models = json.load(open(MODEL_PATH))
demo_trips = json.load(open(DEMO_TRIPS_PATH))

print("Model 4 loaded: energy consumption and arrival SOC (Elastic Net)")


def linear_predict(model, features):
    x = np.array([features[f] for f in model["features"]], dtype=float)
    z = (x - np.array(model["mean"])) / np.array(model["scale"])
    return float(z @ np.array(model["coef"]) + model["intercept"])


def run_model4(driver_id):
    trip = demo_trips[str(driver_id)]
    energy = linear_predict(models["energy"], trip["energy_features"])
    arrival = linear_predict(models["arrival_soc_50"], trip["arrival_features"])
    return dict(
        trip=trip["trip"],
        energy_kwh_per_km=round(energy, 4),
        arrival_soc_pct=round(float(np.clip(arrival, 0, 1)) * 100, 1),
    )
