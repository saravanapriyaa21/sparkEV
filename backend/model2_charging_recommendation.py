import json
import os
import time

import joblib
import numpy as np
import pandas as pd


# ============================================================
# MODEL 2 - PREDICTIVE CHARGING RECOMMENDATION
#
# Three models feed one decision layer:
#
#   Model A2  charge level  : 85th-percentile battery need (%)
#                             until the next charge -> target SOC
#                             (federated MLP, FedAvg)
#   Model B   urgency       : Time-critical / Moderate / Flexible
#                             (federated MLP, FedPer: shared base
#                             + the user's own personal head)
#   Model C   congestion    : site utilisation one hour ahead
#                             (gradient-boosted trees, centralised)
#
# The ESP32 runs Models A2 and B. The server runs Model C and the
# decision layer, which ranks the candidate stations.
#
# Source: notebooks/module2_charging_recommendation/
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

DATA_PATH = os.path.join(
    PROJECT_ROOT, "data", "raw", "Dataset1_charging_reports.csv"
)

RESERVE_PCT = 15        # safety reserve added to the predicted need
SOC_MAX = 95            # the dataset charges to at most 95 %
CHARGE_EFF = 0.88       # energy drawn -> energy stored in the battery

URGENCY_NAMES = ["Time-critical", "Moderate", "Flexible"]
WAIT_NAMES = ["Low", "Medium", "High"]


# ============================================================
# FEDERATED MLP (same maths as the notebook and the ESP32)
# ============================================================

class FederatedMLP:
    """A saved federated network: scaler statistics, weights and,
    for FedPer, one personal head per user."""

    def __init__(self, name):
        meta = json.load(open(os.path.join(MODELS_DIR, f"{name}.json")))
        arrays = np.load(os.path.join(MODELS_DIR, f"{name}.npz"))

        scaler = meta["scaler"]
        self.feats = scaler["feats"]
        self.locations = scaler["locations"]
        self.clip = scaler["clip"]
        self.mean = np.array(scaler["mean"])
        self.std = np.array(scaler["std"])
        self.miss_idx = np.array(scaler["miss_idx"], dtype=int)

        self.n_pers = meta["n_pers"]
        self.spec = meta["spec"]
        self.anchor = self.spec.get("anchor")

        n_layers = sum(k.startswith("G") for k in arrays.files)
        self.G = [arrays[f"G{i}"] for i in range(n_layers)]

        heads = {}
        for key in arrays.files:
            if key.startswith("H__"):
                _, user, i = key.split("__")
                heads.setdefault(user, {})[int(i)] = arrays[key]
        self.heads = {u: [h[i] for i in sorted(h)] for u, h in heads.items()}

    def transform(self, row, location):
        """Raw feature values (dict) -> scaled input vector."""
        x = np.array([
            np.nan if row.get(f) is None else float(row[f])
            for f in self.feats
        ])
        ok = np.isfinite(x)
        z = np.clip(
            np.where(ok, (np.where(ok, x, 0.0) - self.mean) / self.std, 0.0),
            -self.clip, self.clip
        )
        missing = (~ok[self.miss_idx]).astype(float)
        site = np.array([float(location == s) for s in self.locations])
        return np.concatenate([z, missing, site])

    def params(self, user=None):
        if user is not None and user in self.heads:
            return self.G[:len(self.G) - self.n_pers] + self.heads[user]
        return self.G

    def output(self, x, user=None):
        P = self.params(user)
        h = x
        n = len(P) // 2
        for layer in range(n):
            h = h @ P[2 * layer] + P[2 * layer + 1]
            if layer < n - 1:
                h = np.maximum(h, 0.0)
        if self.anchor is not None:
            # The network predicts the difference from the user's own
            # mean need; add that mean back (read from the input).
            a = self.anchor
            mean_need = x[a["col"]] * a["std"] + a["mean"]
            if a["miss_col"] is not None and x[a["miss_col"]] > 0.5:
                mean_need = a["fill"]
            h = h + mean_need / a["scale"]
        return h


charge_level_model = FederatedMLP("model2_charge_level_fedavg")
urgency_model = FederatedMLP("model2_urgency_fedper")

print("Model 2 federated networks loaded (charge level: FedAvg, urgency: FedPer)")


def predict_charge_need(row, location):
    """Model A2: 85th-percentile battery need (%) until the next charge."""
    x = charge_level_model.transform(row, location)
    out = charge_level_model.output(x)
    return float(np.clip(out[0] * charge_level_model.spec["scale"], 0, 90))


def predict_urgency(row, location, user):
    """Model B: urgency class and probabilities, with the user's own head."""
    x = urgency_model.transform(row, location)
    logits = urgency_model.output(x, user)
    p = np.exp(logits - logits.max())
    p = p / p.sum()
    return int(np.argmax(p)), p.tolist()


# ============================================================
# MODEL C - SITE CONGESTION ONE HOUR AHEAD
# ============================================================

congestion = joblib.load(
    os.path.join(MODELS_DIR, "model2_site_congestion.joblib")
)


def build_occupancy(sessions):
    """Hourly utilisation per site (same as notebook 01, Model C)."""
    d = sessions.dropna(subset=["plugout_time"]).copy()
    d["pt"] = pd.to_datetime(d.plugin_time)
    d["po"] = pd.to_datetime(d.plugout_time)
    d = d[d.po > d.pt]
    frames = []
    for loc, g in d.groupby("location"):
        n_users = g.user_id.nunique()
        idx = pd.date_range(g.pt.min().floor("h"), g.po.max().ceil("h"), freq="h")
        ev = pd.Series(0, index=idx, dtype=float)
        s = g.pt.dt.floor("h").value_counts()
        e = (g.po.dt.floor("h") + pd.Timedelta(hours=1)).value_counts()
        ev = ev.add(s, fill_value=0).sub(e, fill_value=0)
        occ = ev.reindex(idx, fill_value=0).cumsum().clip(lower=0)
        f = pd.DataFrame({"ts": idx, "location": loc, "occ": occ.values, "n_users": n_users})
        f["util"] = f.occ / n_users
        frames.append(f)
    h = pd.concat(frames, ignore_index=True)
    h["loc_code"] = h.location.astype("category").cat.codes
    h = h.sort_values(["location", "ts"]).reset_index(drop=True)
    g = h.groupby("location", sort=False)["util"]
    h["util_lag1"] = g.shift(1)
    h["util_lag2"] = g.shift(2)
    h["util_lag3"] = g.shift(3)
    h["util_lag24"] = g.shift(24)
    h["util_lag168"] = g.shift(168)
    h["d1"] = h.util_lag1 - h.util_lag2
    h["d2"] = h.util_lag2 - h.util_lag3
    h["util_roll24"] = g.transform(lambda s: s.shift(1).rolling(24, min_periods=6).mean())
    h["hour"] = h.ts.dt.hour
    h["dow"] = h.ts.dt.dayofweek
    h["month"] = h.ts.dt.month
    h["weekend"] = (h.dow >= 5).astype(int)
    h["hour_sin"] = np.sin(2 * np.pi * h.hour / 24)
    h["hour_cos"] = np.cos(2 * np.pi * h.hour / 24)
    return h


occupancy = build_occupancy(
    pd.read_csv(DATA_PATH, sep=";", decimal=",", na_values=["NA"])
).dropna(subset=["util_lag1", "util_lag24"])

print("Model 2 site occupancy history loaded:",
      occupancy.location.nunique(), "sites")


def predict_site_utilisation(site, now=None):
    """Predicted utilisation (0..1) of a site for the next hour.

    The dataset ends in 2021, so the current weekday and hour are
    mapped onto the latest matching hour in that site's history."""
    now = now or pd.Timestamp.now()
    site_rows = occupancy[occupancy.location == site]
    rows = site_rows[
        (site_rows.dow == now.dayofweek) & (site_rows.hour == now.hour)
    ]
    row = (rows if len(rows) else site_rows).iloc[[-1]]
    change = congestion["reg"].predict(row[congestion["features"]])[0]
    util = float(np.clip(row["util_lag1"].iloc[0] + change, 0, 1.5))
    lo, hi = congestion["thresholds"]
    risk = WAIT_NAMES[int(np.digitize(util, [lo, hi]))]
    return util, risk


# ============================================================
# CANDIDATE STATIONS (demo configuration)
#
# Distance and charger power are set for the demo. Each station
# takes its expected busyness from the occupancy history of one
# real site in the dataset, predicted by Model C.
# ============================================================

STATIONS = [
    dict(name="Station 1", site="OSL_S", distance_km=1.0, power_kw=7.0),
    dict(name="Station 2", site="OSL_T", distance_km=3.5, power_kw=22.0),
    dict(name="Station 3", site="TRO_R", distance_km=6.0, power_kw=11.0),
]


# ============================================================
# DECISION LAYER (same rules as the notebooks)
# ============================================================

def recommend_option(current_soc, battery_kwh, pred_demand_q_pct, urgency_class, candidates,
                     vehicle_max_kw=11.0, urgency_weight=(1.6, 1.0, 0.5), avg_speed_kmh=30):
    target = float(np.clip(pred_demand_q_pct + RESERVE_PCT, 30, SOC_MAX))
    if current_soc >= target:
        return [], target, 0.0, (
            f"Your battery is at {current_soc:.0f}%, which is enough until your next "
            f"usual charge. No need to charge now.")
    need_kwh = (target - current_soc) / 100 * battery_kwh / CHARGE_EFF
    w_t = urgency_weight[int(urgency_class)]
    out = []
    for c in candidates:
        kw = min(c["power_kw"], vehicle_max_kw)
        charge_min = need_kwh / kw * 60
        util = min(c["pred_utilisation"], 0.95)
        wait_min = (charge_min * util / (1 - util)) * 0.5
        travel_min = c["distance_km"] / avg_speed_kmh * 60
        c_rate = kw / battery_kwh
        stress_pen = (c_rate ** 1.5) * 25 * (1.0 if urgency_class == 2 else 0.4 if urgency_class == 1 else 0.0)
        score = w_t * (travel_min + wait_min + charge_min) + stress_pen
        out.append(dict(name=c["name"], site=c["site"], distance_km=c["distance_km"],
                        power_kw=c["power_kw"], wait_risk=c["wait_risk"],
                        travel_min=round(travel_min, 1), wait_min=round(wait_min, 1),
                        charge_min=round(charge_min, 1),
                        total_min=round(travel_min + wait_min + charge_min, 1),
                        stress_penalty=round(stress_pen, 1), score=round(score, 1)))
    out.sort(key=lambda r: r["score"])
    when = ["Please charge soon", "Charge sometime today", "No rush"][int(urgency_class)]
    msg = f"{when}: charge to {target:.0f}% at {out[0]['name']} (about {need_kwh:.1f} kWh)."
    return out, target, need_kwh, msg


def evaluate_charging_option(current_soc, battery_kwh, target_soc, station_name,
                             power_kw=None, vehicle_max_kw=11.0, avg_speed_kmh=30, now=None):
    """Time and energy of one chosen option (used by the What-If Simulator).

    Same travel / waiting / charging formulas as recommend_option, but for a
    station and target the user picks instead of the best-ranked one."""
    station = next(s for s in STATIONS if s["name"] == station_name)
    util, risk = predict_site_utilisation(station["site"], now)
    charger_kw = float(power_kw or station["power_kw"])
    kw = min(charger_kw, vehicle_max_kw)
    need_kwh = max(0.0, (target_soc - current_soc) / 100 * battery_kwh / CHARGE_EFF)
    charge_min = need_kwh / kw * 60
    util = min(util, 0.95)
    wait_min = (charge_min * util / (1 - util)) * 0.5
    travel_min = station["distance_km"] / avg_speed_kmh * 60
    return dict(
        station=station["name"], site=station["site"], distance_km=station["distance_km"],
        charger_kw=charger_kw, effective_kw=kw, wait_risk=risk,
        target_soc=round(float(target_soc), 1), energy_kwh=round(need_kwh, 2),
        energy_stored_kwh=round(need_kwh * CHARGE_EFF, 2),
        c_rate=round(kw / battery_kwh, 3),
        travel_min=round(travel_min, 1), wait_min=round(wait_min, 1),
        charge_min=round(charge_min, 1), total_min=round(travel_min + wait_min + charge_min, 1),
    )


def recommend_charging(current_soc, battery_kwh, charge_need_pct, urgency_class, now=None):
    """Rank the candidate stations with Model C and the decision layer."""
    start = time.perf_counter()
    candidates = []
    for station in STATIONS:
        util, risk = predict_site_utilisation(station["site"], now)
        candidates.append(dict(station, pred_utilisation=util, wait_risk=risk))
    ranked, target, need_kwh, message = recommend_option(
        current_soc, battery_kwh, charge_need_pct, urgency_class, candidates)
    return dict(
        target_soc=round(target, 1),
        need_kwh=round(need_kwh, 2),
        current_soc=round(float(current_soc), 1),
        recommended_station=ranked[0]["name"] if ranked else None,
        message=message,
        stations=ranked,
        latency_ms=round((time.perf_counter() - start) * 1000, 3),
    )
