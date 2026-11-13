import json
import os
import re
import time

import numpy as np
import pandas as pd
import shap
import xgboost as xgb

import federated


# ============================================================
# SHAP EXPLANATIONS AND RECOMMENDATIONS
#
# For the latest prediction of each module, SHAP measures how
# much each input factor pushed the prediction up or down
# compared with an average case. The driver's recommendations
# are built from these contributions: a factor becomes advice
# only if (1) the driver can change it and (2) SHAP shows that
# it is making the prediction worse for this driver now.
#
#   Module 1 battery stress     XGBoost        TreeSHAP (exact)
#   Module 2 charge need        federated MLP  Kernel SHAP, 40 background
#                                              rows (as in the Module 2 notebook)
#   Module 3 SOH forecast       federated MLP  Kernel SHAP over 14 factor groups
#   Module 4 energy use         Elastic Net    linear SHAP (exact)
#
# Modules 2-4 are explained with the server's current global
# federated model (the deployed model before the first round),
# which is the model the car installs after each round.
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

KERNEL_NSAMPLES = 500
TOP_K = 6


def action(text):
    """Advice: a controllable factor that SHAP shows is making the prediction worse."""
    return dict(text=text, kind="action")


def info(text):
    """Explanation without a needed change."""
    return dict(text=text, kind="info")


def contribution_list(names, phi, values=None, k=TOP_K):
    order = np.argsort(-np.abs(phi))[:k]
    return [dict(factor=names[i], shap=round(float(phi[i]), 4),
                 value=None if values is None else values[i])
            for i in order]


def grouped_kernel_shap(predict, x, background, groups, nsamples=KERNEL_NSAMPLES, seed=0):
    """Kernel SHAP where each player is a group of input columns.
    A group that is 'off' takes its values from the background rows
    (interventional SHAP, averaged over the background)."""
    d = len(x)
    membership = np.zeros((len(groups), d), dtype=bool)
    for g, cols in enumerate(groups):
        membership[g, cols] = True

    def f(Z):
        on = (Z @ membership) > 0                                    # (k, d)
        X = np.where(on[:, None, :], x[None, None, :], background[None, :, :])
        out = predict(X.reshape(-1, d)).reshape(len(Z), len(background))
        return out.mean(axis=1)

    np.random.seed(seed)
    explainer = shap.KernelExplainer(f, np.zeros((1, len(groups))))
    phi = explainer.shap_values(np.ones((1, len(groups))), nsamples=nsamples, silent=True)
    return float(np.ravel(explainer.expected_value)[0]), np.asarray(phi).reshape(-1)


# ============================================================
# MODULE 1 - BATTERY STRESS (XGBoost, TreeSHAP)
# ============================================================

model1 = xgb.XGBRegressor()
model1.load_model(os.path.join(MODELS_DIR, "model1_battery_stress_xgb.json"))
M1_FEATURES = model1.get_booster().feature_names
tree_explainer = shap.TreeExplainer(model1)

M1_NAMES = {
    "avg_speed_kmh": "average speed", "speed_std_kmh": "speed variation",
    "max_speed_kmh": "top speed", "accel_mean": "average acceleration",
    "accel_std": "acceleration variation", "accel_max": "strongest acceleration",
    "accel_min": "strongest braking", "jerk_std": "jerky pedal use",
    "hard_accel_count": "hard accelerations", "hard_brake_count": "hard braking events",
    "idle_time_pct": "idling time", "regen_braking_pct": "regenerative braking share",
    "driver_aggressiveness_score": "aggressiveness score",
    "ambient_temp_mean_c": "outside temperature", "route": "route", "weather": "weather",
}

# controllable factor -> advice (used only when SHAP shows it raises the stress)
M1_ADVICE = {
    "hard_accel_count": "Accelerate more gently and evenly",
    "accel_max": "Accelerate more gently and evenly",
    "accel_std": "Accelerate more gently and evenly",
    "accel_mean": "Accelerate more gently and evenly",
    "jerk_std": "Accelerate more gently and evenly",
    "hard_brake_count": "Brake earlier and more gently",
    "accel_min": "Brake earlier and more gently",
    "avg_speed_kmh": "Lower your speed, especially the top speed",
    "max_speed_kmh": "Lower your speed, especially the top speed",
    "speed_std_kmh": "Keep a steadier speed",
    "regen_braking_pct": "Use regenerative braking more (lift off earlier instead of braking)",
    "driver_aggressiveness_score": "Drive more calmly overall",
    "idle_time_pct": "Avoid long idling",
}
M1_MAX_ACTIONS = 3


def explain_module1(features, min_points=0.5):
    X = pd.DataFrame([[features[f] for f in M1_FEATURES]], columns=M1_FEATURES)
    phi = np.asarray(tree_explainer.shap_values(X)).reshape(-1)
    base = float(np.ravel(tree_explainer.expected_value)[0])

    # one-hot route and weather columns are reported as one factor each
    merged, values = {}, {}
    for f, p in zip(M1_FEATURES, phi):
        key = f.split("_")[0] if f.startswith(("route_", "weather_")) else f
        merged[key] = merged.get(key, 0.0) + float(p)
        if key == f:
            values[key] = round(float(features[f]), 2)
        elif features[f] == 1:
            values[key] = f.split("_", 1)[1]
    keys = list(merged)
    names = [M1_NAMES.get(k, k) for k in keys]
    contribs = contribution_list(names, np.array([merged[k] for k in keys]),
                                 [values.get(k) for k in keys])

    actions = {}
    for k in keys:
        if k in M1_ADVICE and merged[k] >= min_points:
            a = actions.setdefault(M1_ADVICE[k], [])
            a.append((M1_NAMES[k], merged[k], values.get(k)))
    advice = []
    ranked = sorted(actions.items(), key=lambda t: -sum(p[1] for p in t[1]))[:M1_MAX_ACTIONS]
    for text, parts in ranked:
        total = sum(p[1] for p in parts)
        pts = round(total)
        advice.append(action(f"{text}. This adds about {pts} point{'' if pts == 1 else 's'} to your "
                             f"battery stress score compared with an average trip."))
    if not advice:
        advice.append(info("Your driving style is gentle on the battery. Keep it up."))
    pred = base + float(phi.sum())
    return dict(module="Module 1", title="Battery stress", method="TreeSHAP (exact)",
                unit="stress points", base_value=round(base, 2), prediction=round(pred, 2),
                contributions=contribs, advice=advice)


# ============================================================
# MODULE 2 - CHARGE NEED (federated MLP, Kernel SHAP)
# ============================================================

M2_NAMES = {
    "hour": "arrival hour", "dow": "day of week", "month": "month", "weekend": "weekend",
    "hour_sin": "arrival hour", "hour_cos": "arrival hour", "battery_capacity": "battery capacity",
    "charging_power": "charger power", "cur_soc_need": "battery used this time",
    "cur_energy": "energy this session", "cur_charge_time": "charging time this session",
    "n_prev": "number of past charges", "hrs_since_prev_plugout": "hours since last unplug",
    "hist_hour_mean": "usual arrival hour", "soc_lag1": "battery used last time",
    "soc_lag2": "battery used 2 charges ago", "soc_lag3": "battery used 3 charges ago",
    "soc_roll5": "average use, last 5 charges", "soc_hist_mean": "how much battery you usually use",
    "soc_hist_med": "your typical usage", "soc_hist_q75": "your busier days",
    "soc_hist_q90": "your busiest days", "soc_hist_std": "how much your usage varies",
    "soc_hist_dow_mean": "your usual use on this weekday", "soc_per_day_hist": "your daily usage",
    "gap_prev_days": "days since last charge", "gap_hist_mean": "usual days between charges",
    "gap_hist_med": "typical days between charges", "after_dow_gap_mean": "usual gap after this weekday",
    "after_dow_soc_mean": "usual use after this weekday",
    "after_dow_soc_med": "typical use after this weekday",
    "after_hour_soc_mean": "usual use after this time of day", "after_soc_mean": "usual use until the next charge",
    "conn_hist_mean": "usual plug-in duration", "conn_hist_med": "typical plug-in duration",
    "conn_hist_dow_mean": "plug-in duration on this weekday",
    "conn_hist_hour_mean": "plug-in duration at this hour", "conn_lag1": "last plug-in duration",
    "slack_lag1": "idle plug-in time, last charge", "slack_hist_med": "typical idle plug-in time",
}

M2_TASK = federated.TASKS["m2c"]
M2_CFG = json.load(open(os.path.join(MODELS_DIR, "model2_charge_level_fedavg.json")))
_z2 = np.load(os.path.join(federated.FL_DIR, "m2_charge_fl.npz"))
_pool2 = np.concatenate([_z2[f"fleet_X_{k}"] for k in range(M2_TASK.cfg["n_fleet"])]).astype(np.float32)
M2_BACKGROUND = _pool2[np.sort(np.random.default_rng(42).choice(len(_pool2), 40, replace=False))]


def _m2_groups():
    sc = M2_CFG["scaler"]
    feats, miss = sc["feats"], sc["miss_idx"]
    groups, names = {}, []
    for j, f in enumerate(feats):
        name = M2_NAMES.get(f, f.replace("_", " "))
        groups.setdefault(name, []).append(j)
        if j in miss:
            groups[name].append(len(feats) + miss.index(j))   # its missing-value flag
    site = list(range(len(feats) + len(miss), len(feats) + len(miss) + len(sc["locations"])))
    groups["charging site"] = site
    names = list(groups)
    return names, [groups[n] for n in names]


M2_GROUP_NAMES, M2_GROUPS = _m2_groups()


def m2_charge_need(X):
    """Charge need (%) of the current global charge-need model for scaled inputs."""
    G, anc = M2_TASK.G, M2_CFG["spec"]["anchor"]
    h = X
    for l in range(3):
        h = h @ G[2 * l] + G[2 * l + 1]
        if l < 2:
            h = np.maximum(h, 0)
    mean_need = X[:, anc["col"]] * anc["std"] + anc["mean"]
    if anc["miss_col"] is not None:
        mean_need = np.where(X[:, anc["miss_col"]] > 0.5, anc["fill"], mean_need)
    return (h[:, 0] + mean_need / anc["scale"]) * anc["scale"]


def explain_module2(x_scaled, raw_row, target_soc):
    base, phi = grouped_kernel_shap(m2_charge_need, np.asarray(x_scaled, np.float32),
                                    M2_BACKGROUND, M2_GROUPS)
    sc = M2_CFG["scaler"]
    values = []
    for name, cols in zip(M2_GROUP_NAMES, M2_GROUPS):
        f = sc["feats"][cols[0]] if cols[0] < len(sc["feats"]) else None
        v = raw_row.get(f) if f else None
        values.append(None if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), 2))
    contribs = contribution_list(M2_GROUP_NAMES, phi, values)
    up = [c for c in contribs if c["shap"] > 0][:3]
    down = [c for c in contribs if c["shap"] < 0][:2]
    pred = base + float(phi.sum())
    main = contribs[0]["factor"] if contribs else "your past charging"
    text = (f"Based on your own charging habits, you may use up to {pred:.0f}% of your battery "
            f"before your next charge (a typical driver uses {base:.0f}%), so you should have at least "
            f"{target_soc:.0f}% in your battery when you set off. The biggest reason is {main}.")
    advice = [info(text)]
    return dict(module="Module 2", title="Charge need before the next charge",
                method="Kernel SHAP (federated MLP)", unit="% battery",
                base_value=round(base, 2), prediction=round(pred, 2),
                model_round=M2_TASK.round, contributions=contribs, advice=advice)


# ============================================================
# MODULE 3 - SOH FORECAST (federated MLP, grouped Kernel SHAP)
# ============================================================

M3_TASK = federated.TASKS["m3"]
M3_CFG = M3_TASK.cfg
M3_BACKGROUND = np.load(os.path.join(federated.FL_DIR, "m3_fl.npz"))["shap_background"].astype(np.float32)
_z3 = np.load(os.path.join(federated.FL_DIR, "m3_fl.npz"))
M3_W0, M3_B0 = _z3["G0"].astype(np.float32), _z3["G1"].astype(np.float32)   # frozen first layer

M3_GROUP_RULES = [
    ("charging temperature", r"^temp_ch|^temp_deviation_25|^temp_voltage_interaction"),
    ("battery temperature at rest", r"^temp_rest"),
    ("charging current", r"^curr_0|^abs_current|^current_voltage_product"),
    ("how full or empty you charge", r"^soc_start|^soc_end|^delta_soc|^soc_range"),
    ("current battery health and trend", r"^soh|degradation|^acceleration_ratio|^relative_rate_increase"),
    ("battery capacity", r"^cap"),
    ("internal resistance", r"dcr|^dV_|^rest_dv"),
    ("battery voltage", r"volt|delta_voltage"),
    ("internal battery measurements", r"^vf|^zf"),
    ("battery age and mileage", r"^seg_num|^cycle_lag|^cycle$|^mileage"),
    ("time of day you charge", r"^start_|^cycle_duration"),
]
M3_ADVICE = {
    "charging temperature": "Charge when the battery is cooler: avoid charging right after a long drive or in direct sun",
    "battery temperature at rest": "Park in the shade or a garage to keep the battery cool",
    "charging current": "Use slower (AC) charging when you have time",
    "how full or empty you charge": "Avoid very deep discharges and long stays near 100%; keep the charge between about 20 and 80%",
    "fast or slow charging": "Use slow charging instead of fast charging when possible",
}


def _m3_groups():
    pre = M3_CFG["preprocessing"]
    groups = {}
    for j, f in enumerate(pre["numeric_features"]):
        name = next((n for n, rx in M3_GROUP_RULES if re.search(rx, f)), "other measurements")
        groups.setdefault(name, []).append(j)
    j = len(pre["numeric_features"])
    for col, cats in zip(pre["categorical_features"], pre["categories"]):
        name = {"charge_type": "fast or slow charging", "battery_type": "battery chemistry",
                "car_identifier": "vehicle"}.get(col, col)
        groups[name] = list(range(j, j + len(cats)))
        j += len(cats)
    names = list(groups)
    return names, [groups[n] for n in names]


M3_GROUP_NAMES, M3_GROUPS = _m3_groups()


def m3_soh_change(X):
    """SOH change over the next 10 cycles (percentage points), current global model."""
    G = M3_TASK.G
    h = np.maximum(X @ M3_W0 + M3_B0, 0)
    h = np.maximum(h @ G[2] + G[3], 0)
    out = (h @ G[4] + G[5])[:, 0]
    return 100 * (out * M3_CFG["delta_std"] + M3_CFG["delta_mean"])


def explain_module3(driver_id, soh_now_pct, soh_future_pct, risk, min_points=0.02):
    x = np.load(os.path.join(federated.FL_DIR, "m3_fl.npz"))[f"car{driver_id}_now_X"].astype(np.float32)
    base, phi = grouped_kernel_shap(m3_soh_change, x, M3_BACKGROUND, M3_GROUPS)
    contribs = contribution_list(M3_GROUP_NAMES, phi)
    harmful = sorted([(n, p) for n, p in zip(M3_GROUP_NAMES, phi) if p < 0], key=lambda t: t[1])
    state = {"Low": "Your battery is in good health.",
             "Medium": "Your battery is ageing, so the tips below help it last longer.",
             "High": "Your battery health is low. Please plan a service check."}.get(risk, "")
    advice = [(info if risk == "Low" else action)(
        f"Your battery health is {soh_now_pct:.1f}% now and should be about {soh_future_pct:.1f}% "
        f"after your next 10 charges. {state}".strip())]
    for name, p in harmful:
        if name in M3_ADVICE and p <= -min_points:
            advice.append(action(f"{M3_ADVICE[name]}. Right now, {name} lowers your expected battery "
                                 f"health by {abs(p):.2f}%."))
    if len(advice) == 1:
        advice.append(info("Your charging habits are good for the battery. Keep it up."))
    pred = base + float(phi.sum())
    return dict(module="Module 3", title="SOH change over the next 10 cycles",
                method="Kernel SHAP over factor groups (federated MLP)", unit="SOH points",
                base_value=round(base, 3), prediction=round(pred, 3),
                model_round=M3_TASK.round, contributions=contribs, advice=advice)


# ============================================================
# MODULE 4 - ENERGY USE (Elastic Net, exact linear SHAP)
# ============================================================

M4_TASK = federated.TASKS["m4"]
M4_CFG = M4_TASK.cfg
M4_GROUP_OF = {
    "heating_power_mean": "heating", "requested_heating_mean": "heating",
    "requested_heating_max": "heating", "requested_heating_min": "heating",
    "aircon_power_mean": "air conditioning", "aircon_power_min": "air conditioning",
    "aircon_power_max": "air conditioning",
    "cabin_temp_min": "cabin temperature", "cabin_temp_max": "cabin temperature",
    "ambient_temp_min": "outside temperature", "heat_exchanger_temp_mean": "heating system temperature",
    "speed_above_50_fraction": "time above 50 km/h",
    "acceleration_mean": "acceleration", "acceleration_max": "acceleration",
    "acceleration_min": "acceleration", "jerk_mean_abs": "jerky driving", "throttle_mean": "throttle use",
    "battery_current_mean": "power drawn from the battery", "battery_current_min": "power drawn from the battery",
    "battery_current_max": "power drawn from the battery", "elevation_std": "hills on the route",
    "Duration [min]": "trip duration", "soc_max": "battery level at the start",
}
# features where a higher value physically means more energy use; advice is given
# only when SHAP and this direction agree (the model is a correlation model)
M4_ADVICE = {
    "heating": ("heating_power_mean", "Use less cabin heating (pre-heat while plugged in, use seat heating)"),
    "air conditioning": ("aircon_power_mean", "Use less air conditioning (pre-cool while plugged in)"),
    "time above 50 km/h": ("speed_above_50_fraction", "Drive a little slower on fast roads"),
    "jerky driving": ("jerk_mean_abs", "Drive more smoothly"),
    "throttle use": ("throttle_mean", "Press the accelerator more gently"),
}


def explain_module4(energy_features):
    feats = M4_CFG["features"]
    x = np.array([energy_features[f] for f in feats], float)
    z = (x - np.array(M4_CFG["scaler_mean"])) / np.array(M4_CFG["scaler_scale"])
    coef, intercept = M4_TASK.G[0].astype(float), float(M4_TASK.G[1][0])
    phi_f = 1000 * coef * z                       # Wh/km; exact SHAP of a linear model
    groups = {}
    for f, p in zip(feats, phi_f):
        g = M4_GROUP_OF.get(f, f)
        groups[g] = groups.get(g, 0.0) + p
    names = list(groups)
    contribs = contribution_list(names, np.array([groups[n] for n in names]))
    advice = []
    for g, (feat, text) in M4_ADVICE.items():
        zi = z[feats.index(feat)]
        if groups.get(g, 0) > 1.0 and zi > 0:
            advice.append(action(f"{text}. On this trip, {g} adds about {groups[g]:.0f} Wh/km "
                                 f"to your energy use."))
    if not advice:
        advice.append(info("Your trip is already energy efficient. Nothing you control is adding "
                           "extra energy use."))
    pred = 1000 * intercept + float(phi_f.sum())
    return dict(module="Module 4", title="Energy use of the trip", method="Linear SHAP (exact, Elastic Net)",
                unit="Wh/km", base_value=round(1000 * intercept, 1), prediction=round(pred, 1),
                model_round=M4_TASK.round, contributions=contribs, advice=advice)


# ============================================================
# ALL MODULES, WITH A CACHE
#
# The demo inputs of a driver do not change, so the explanation
# only has to be recomputed when a federated round changes a model.
# ============================================================

_cache = {}


def explain_all(driver_id, m1_features, m2_x, m2_raw, target_soc, soh_now_pct, soh_future_pct,
                risk, m4_features):
    key = (driver_id, round(target_soc, 2), round(soh_future_pct, 3),
           M2_TASK.round, M3_TASK.round, M4_TASK.round)
    if key in _cache:
        return _cache[key]
    start = time.perf_counter()
    result = dict(
        stress=explain_module1(m1_features),
        charging=explain_module2(m2_x, m2_raw, target_soc),
        battery=explain_module3(driver_id, soh_now_pct, soh_future_pct, risk),
        energy=explain_module4(m4_features),
    )
    result["shap_ms"] = round(1000 * (time.perf_counter() - start), 1)
    _cache.clear()
    _cache[key] = result
    return result


print(f"SHAP explanations ready (Module 1 TreeSHAP, Modules 2-3 Kernel SHAP, Module 4 linear)")
