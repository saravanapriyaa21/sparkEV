import json
import os

import numpy as np


# ============================================================
# BATTERY AGEING BY CHARGE LEVEL, CHARGING SPEED, TEMPERATURE
#
# Fitted on the open KIT ageing dataset (228 NMC/C-SiO cells;
# Luh & Blank 2024, Scientific Data, doi:10.1038/s41597-024-03831-x,
# data doi:10.35097/1969, CC BY 4.0).
#
#   cycle wear : capacity points lost per equivalent full cycle,
#                for charging to 90 % or 100 % (interpolated
#                between), by charging rate and temperature
#   storage    : ageing rate while parked at a charge level,
#                relative to 50 %
#
# Source: notebooks/module1_battery_stress/04_charge_level_ageing_model.ipynb
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

ageing = json.load(open(os.path.join(
    PROJECT_ROOT, "models", "battery_ageing_kit.json"
)))

CYCLE = ageing["cycle_model"]
STORAGE_SOC = np.array(ageing["storage_k"]["soc"])

print("Battery ageing model loaded (KIT dataset)")


def _log_rate(to_100, c_rate, temp_c, dis_c):
    c = CYCLE["coef"]
    log_chg = np.log(max(c_rate, CYCLE["min_charge_rate_c"]) / (1 / 3))
    log_dis = np.log(dis_c / (1 / 3))
    return (CYCLE["intercept"] + c["to_100"] * to_100 + c["log_chg"] * log_chg
            + c["log_dis"] * log_dis + c["t0"] * (temp_c == 0) + c["t10"] * (temp_c == 10)
            + c["t40"] * (temp_c == 40) + c["log_chg_cold"] * log_chg * (temp_c <= 10))


def cycle_wear_rate(target_soc, c_rate, temp_c=25, dis_c=1 / 3):
    """Capacity points lost per equivalent full cycle. Targets below 90 % use the
    90 % value (lower levels were not tested), rates below 1/3 C use 1/3 C."""
    frac = float(np.clip((target_soc - 90.0) / 10.0, 0.0, 1.0))
    return float(np.exp((1 - frac) * _log_rate(0.0, c_rate, temp_c, dis_c)
                        + frac * _log_rate(1.0, c_rate, temp_c, dis_c)))


def session_wear(energy_kwh, battery_kwh, target_soc, c_rate, temp_c=25):
    """Capacity points lost by one charge (energy charged as a share of a full cycle)."""
    if energy_kwh <= 0:
        return 0.0
    return energy_kwh / battery_kwh * cycle_wear_rate(target_soc, c_rate, temp_c)


def storage_factor(soc, temp_c=25):
    """Ageing rate while parked at this charge level, relative to 50 %."""
    k = np.array(ageing["storage_k"]["by_temp"][str(int(temp_c))])
    return float(np.interp(soc, STORAGE_SOC, k) / np.interp(50.0, STORAGE_SOC, k))
