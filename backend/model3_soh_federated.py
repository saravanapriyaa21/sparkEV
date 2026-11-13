import json
import os
import time

import numpy as np


# ============================================================
# MODULE 3 - SOH FORECASTING, FEDERATED MLP (FedAvg)
#
# 357 -> 64 -> 32 -> 1 network trained with FedAvg (each battery
# is a client). It predicts the SOH change over the next 10 cycles;
# the forecast is SOH now + predicted change. Same model and same
# input as the ESP32 (fl_esp32_generated.h).
#
# Source: notebooks/module3_battery_health/02_soh_centralized_vs_federated.ipynb
# Export: esp32/tools/export_fl_setup.py -> models/fl/m3_fl.{json,npz}
#
# The demo cars' batteries are independent test batteries of the
# notebook ("new cars joining the fleet"); the input is the
# processed features of each battery's newest cycle.
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FL_DIR = os.path.join(PROJECT_ROOT, "models", "fl")

config = json.load(open(os.path.join(FL_DIR, "m3_fl.json")))
_z = np.load(os.path.join(FL_DIR, "m3_fl.npz"))
WEIGHTS = [_z[f"G{i}"].astype(np.float32) for i in range(6)]
DELTA_MEAN = config["delta_mean"]
DELTA_STD = config["delta_std"]

DRIVER_BATTERIES = {
    int(driver): dict(
        battery=info["battery"],
        cycle=info["seg_num"],
        soh_now=info["soh_now"],
        x=_z[f"car{driver}_now_X"].astype(np.float32),
    )
    for driver, info in config["demo"].items()
}

print(f"Model 3 federated SOH model loaded (FedAvg, test R2 "
      f"{config['notebook_result']['test_r2']:.4f})")


def soh_risk(soh_fraction):
    """Same thresholds as the ESP32 (85 % / 80 %)."""
    pct = 100 * soh_fraction
    if pct >= 85:
        return "Low"
    if pct >= 80:
        return "Medium"
    return "High"


def forecast(x, soh_now):
    h = np.maximum(x @ WEIGHTS[0] + WEIGHTS[1], 0)
    h = np.maximum(h @ WEIGHTS[2] + WEIGHTS[3], 0)
    change = float((h @ WEIGHTS[4] + WEIGHTS[5])[0]) * DELTA_STD + DELTA_MEAN
    return soh_now + change


def run_model3(driver_id):
    car = DRIVER_BATTERIES[driver_id]
    start = time.perf_counter()
    soh_future = forecast(car["x"], car["soh_now"])
    latency_ms = (time.perf_counter() - start) * 1000
    return {
        "battery": car["battery"],
        "cycle": car["cycle"],
        "soh_now_percent": round(100 * car["soh_now"], 2),
        "soh_percent": round(100 * soh_future, 2),
        "soh_change_points": round(100 * (soh_future - car["soh_now"]), 2),
        "battery_risk": soh_risk(soh_future),
        "model_3_latency_ms": round(latency_ms, 3),
    }
