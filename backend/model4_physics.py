import json
import os

import numpy as np
import pandas as pd


# ============================================================
# MODEL 4 - PHYSICS-BASED VEHICLE ENERGY MODEL (WHAT-IF)
#
# Grey-box driving-resistance model fitted to the measured
# battery power of the Munich trips:
#
#   P_wheel   = (m a + m g Crr + m g grade + 1/2 rho CdA v^2) v
#   P_battery = P_wheel / eta  (driving)  or  P_wheel * eta_regen
#               (braking) + k_heat P_heat + k_ac P_ac + P_aux
#
# Used by the What-If Simulator to estimate how driving style,
# speed and heating / AC change the energy consumption of the
# driver's demo trip. The Elastic Net (model4_energy.py) stays
# the model that predicts a trip's consumption.
#
# Source: notebooks/module4_energy_and_soc/04_physics_energy_model.ipynb
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

physics = json.load(open(os.path.join(
    PROJECT_ROOT, "models", "model4_physics_energy.json"
)))

PARAMS = physics["params"]
G = physics["constants"]["g"]
RHO = physics["constants"]["rho"]
TREND_WINDOW_S = physics["constants"]["trend_window_s"]

print("Model 4 physics energy model loaded (What-If Simulator)")


def battery_power(v, a, grade, heat, ac, is_b):
    p = PARAMS
    force = (p["m_kg"] * a + p["m_kg"] * G * p["Crr"] * (v > 0.1)
             + p["m_kg"] * G * grade + 0.5 * RHO * p["CdA_m2"] * v ** 2)
    p_wheel = force * v
    p_drive = np.where(p_wheel >= 0, p_wheel / p["eta"], p_wheel * p["eta_regen"])
    k_heat = is_b * p["k_heat_B"] + (1 - is_b) * p["k_heat_A"]
    k_ac = is_b * p["k_ac_B"] + (1 - is_b) * p["k_ac_A"]
    return p_drive + k_heat * heat + k_ac * ac + p["P_aux_W"]


def simulate_kwh_per_km(driver_id, style=1.0, speed=1.0, climate=1.0):
    """Energy consumption of the driver's demo trip with a changed driving style
    (scales accelerating and braking around the traffic flow), speed and
    heating / AC use. 1.0 = as driven."""
    trip = physics["demo_profiles"][str(driver_id)]
    v = np.array(trip["v"])
    trend = pd.Series(v).rolling(TREND_WINDOW_S, center=True, min_periods=1).mean().values
    v_new = np.clip(trend + style * (v - trend), 0, None) * speed
    dt = 1.0 / speed
    a_new = np.gradient(v_new, dt)
    power = battery_power(v_new, a_new, np.array(trip["grade"]),
                          climate * np.array(trip["heat"]), climate * np.array(trip["ac"]),
                          trip["is_b"])
    km = v_new.sum() * dt / 1000
    return float(power.sum() * dt / 3.6e6 / km)
