import numpy as np
import pandas as pd

from model2_charging_recommendation import evaluate_charging_option
from model4_physics import simulate_kwh_per_km
from battery_ageing import session_wear, storage_factor


# ============================================================
# WHAT-IF SIMULATOR
#
# Lets the driver try other decisions than the recommended
# one and shows the predicted consequences side by side:
#
#   charging : station, charge level, charger power
#              -> time, energy, battery wear      (Module 2 +
#                 battery ageing model, KIT data)
#   driving  : driving style, speed  -> driving stress (Module 1)
#              driving style, speed,
#              heating / AC          -> energy use and range
#                                       (Module 4 physics model)
#
# Every number comes from the project's own models. Only the
# inputs the driver controls are changed; everything else stays
# at the driver's current values, so the effects are model
# estimates, not measured outcomes.
# ============================================================


# ------------------------------------------------------------
# Battery wear of a charging session
#
# Measured on the open KIT ageing dataset (battery_ageing.py,
# notebooks/module1_battery_stress/04_charge_level_ageing_model.ipynb):
# wear per full cycle depends on the charge level reached (90 % vs 100 %)
# and, in the cold, on the charging rate. Parking at a high level ages the
# battery faster; that is reported separately, relative to 50 %.
# ------------------------------------------------------------

CHARGE_TEMP_C = 25      # battery temperature while charging (thermally managed pack)


# ------------------------------------------------------------
# Driving inputs the driver controls
# ------------------------------------------------------------

# Module 1 (driving stress) inputs scaled by the driving-style factor
MODEL1_STYLE_FEATURES = [
    "accel_std", "accel_max", "jerk_std", "hard_accel_count",
    "hard_brake_count", "driver_aggressiveness_score",
]
MODEL1_SPEED_FEATURES = ["avg_speed_kmh", "speed_std_kmh", "max_speed_kmh"]

# Energy use comes from the physics-based vehicle model of Module 4
# (notebooks/module4_energy_and_soc/04_physics_energy_model.ipynb): it
# changes the speed profile of the driver's demo trip and recomputes the
# battery power with the driving-resistance equation, so smoother or slower
# driving and less heating / AC lower the energy use by a physical amount.
# The Elastic Net is not used here: its driving coefficients are not causal.


def driving_outcome(model1, feature_names, driver_profile, driver_id, style, speed, climate):
    """Driving stress (Module 1) and energy use (Module 4 physics model)."""
    f1 = dict(driver_profile)
    for name in MODEL1_STYLE_FEATURES:
        f1[name] = f1[name] * style
    f1["accel_min"] = f1["accel_min"] * style          # braking: negative, scaled the same way
    f1["driver_aggressiveness_score"] = float(np.clip(f1["driver_aggressiveness_score"], 0, 100))
    for name in MODEL1_SPEED_FEATURES:
        f1[name] = f1[name] * speed
    stress = float(model1.predict(pd.DataFrame([[f1[n] for n in feature_names]], columns=feature_names))[0])

    energy = simulate_kwh_per_km(driver_id, style, speed, climate)

    return float(np.clip(stress, 0, 100)), max(energy, 0.01)


def scenario(context, model1, feature_names, driver_profile, station, target_soc,
             power_kw=None, style=1.0, speed=1.0, climate=1.0):
    option = evaluate_charging_option(
        context["current_soc"], context["battery_kwh"], target_soc, station, power_kw)
    stress, energy = driving_outcome(
        model1, feature_names, driver_profile, context["driver_id"], style, speed, climate)
    final_soc = max(context["current_soc"], target_soc)
    wear = session_wear(option["energy_stored_kwh"], context["battery_kwh"], final_soc,
                        option["c_rate"], CHARGE_TEMP_C)
    return dict(
        **option,
        wear=wear,
        capacity_loss_per_100=round(100 * wear, 2),
        parked_ageing=round(storage_factor(final_soc, CHARGE_TEMP_C), 2),
        driving_stress=round(stress, 1),
        energy_kwh_per_km=round(energy, 4),
        range_km=round(final_soc / 100 * context["battery_kwh"] / energy),
        range_added_km=round(option["energy_stored_kwh"] / energy),
    )


def run_whatif(context, model1, feature_names, driver_profile, request):
    """Recommended option vs the driver's scenario, with differences."""
    base = scenario(context, model1, feature_names, driver_profile,
                    context["recommended_station"], context["target_soc"])
    alt = scenario(context, model1, feature_names, driver_profile,
                   request["station"], request["target_soc"], request.get("power_kw"),
                   request.get("driving_style", 1.0), request.get("speed", 1.0),
                   request.get("climate", 1.0))

    # battery wear relative to the recommended charging session
    ref = base["wear"] if base["wear"] > 0 else None
    for row in (base, alt):
        row["relative_wear"] = round(row["wear"] / ref, 2) if ref else None
        del row["wear"]

    compare = [
        ("total_min", "Total time (travel + wait + charge)", "min"),
        ("charge_min", "Charging time", "min"),
        ("wait_min", "Expected waiting time", "min"),
        ("energy_kwh", "Energy charged", "kWh"),
        ("range_added_km", "Range added", "km"),
        ("relative_wear", "Battery wear of this charge (vs. recommended)", "x"),
        ("capacity_loss_per_100", "Capacity lost per 100 such charges", "% points"),
        ("parked_ageing", "Ageing while parked at this level (vs. 50%)", "x"),
        ("driving_stress", "Driving stress (Battery Stress Index)", "/100"),
        ("energy_kwh_per_km", "Energy use", "kWh/km"),
        ("range_km", "Range after charging", "km"),
    ]
    rows = []
    for key, label, unit in compare:
        b, a = base[key], alt[key]
        rows.append(dict(key=key, label=label, unit=unit, recommended=b, scenario=a,
                         difference=None if a is None or b is None else round(a - b, 4)))
    return dict(recommended=base, scenario=alt, comparison=rows)
