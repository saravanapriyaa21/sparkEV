"""Train and export the Module 4 deployment models and the two demo trips.

Usage (from the project root):

    python esp32/tools/export_model4_models.py --trips "/path/to/TUM trip data"

The trip folder holds Overview.xlsx and TripA01.csv ... TripB38.csv. The notebooks use
np.trapz, so run this with numpy < 2.3 (for example numpy 2.2.6, scikit-learn 1.6.1).

It runs the code cells of the Module 4 notebooks up to the point where the final features
and settings are defined, then writes:

    models/model4_energy_arrival_soc.json
        energy:          Stable-23 Elastic Net (alpha 0.001, l1_ratio 0.3), fitted on all
                         valid trips, as in notebook 01
        arrival_soc_50:  the notebook 03 Elastic Net that predicts arrival SOC from the
                         first 50 % of a trip (fitted on its training trips)
    esp32/tools/vectors/model4_demo_trips.json
        driver 17: the notebook 03 test trip with the highest predicted energy consumption
        driver 24: the notebook 03 test trip with the lowest predicted energy consumption
"""
import argparse
import json
import os
import warnings

import matplotlib
matplotlib.use("Agg")
warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NB_DIR = os.path.join(PROJECT_ROOT, "notebooks", "module4_energy_and_soc")

parser = argparse.ArgumentParser()
parser.add_argument("--trips", required=True, help="folder with Overview.xlsx and the Trip*.csv files")
args = parser.parse_args()
TRIPS = os.path.abspath(args.trips)


def run_cells(path, stop_after, replacements):
    nb = json.load(open(path))
    ns = {"display": print, "__name__": "__main__", "TRIPS_DIR": TRIPS}
    for i, cell in enumerate(nb["cells"]):
        if i > stop_after:
            break
        if cell["cell_type"] != "code":
            continue
        src = "\n".join(l for l in "".join(cell["source"]).splitlines()
                        if not l.lstrip().startswith(("!", "%")))
        for old, new in replacements:
            src = src.replace(old, new)
        exec(compile(src, f"{os.path.basename(path)}[{i}]", "exec"), ns)
    return ns


def pipeline_json(pipe, feats):
    scaler = pipe.named_steps["scaler"]
    net = [step for name, step in pipe.named_steps.items() if name != "scaler"][0]
    return dict(features=[str(f) for f in feats], mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist(),
                coef=net.coef_.tolist(), intercept=float(net.intercept_),
                alpha=float(net.alpha), l1_ratio=float(net.l1_ratio))


# ---- energy consumption (notebook 01)
nb01 = run_cells(os.path.join(NB_DIR, "01_energy_consumption_prediction.ipynb"), 139,
                 [(r"C:\Users\gayat\projects\get_into\FIP\module_01", TRIPS)])
from sklearn.linear_model import ElasticNet  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

model_df, X_compact = nb01["model_df"], nb01["X_compact"]
stable = list(nb01["stable_23_features"])
y_energy = model_df.loc[X_compact.index, "Energy Consumption [kWh/km]"]
energy = Pipeline([("scaler", StandardScaler()),
                   ("elastic_net", ElasticNet(alpha=0.001, l1_ratio=0.3, max_iter=100000, random_state=42))])
energy.fit(X_compact[stable], y_energy)
trip_of = model_df.loc[X_compact.index, "Trip"].astype(str)

# ---- arrival SOC from the first 50 % of a trip (notebook 03)
nb03 = run_cells(os.path.join(NB_DIR, "03_arrival_soc_forecasting.ipynb"), 9,
                 [('MODULE_DIR = Path.cwd() / "module_01"', "MODULE_DIR = Path(TRIPS_DIR)"),
                  ("MODULE_DIR = Path.cwd()", "MODULE_DIR = Path(TRIPS_DIR)")])
X50, y50, _ = nb03["datasets"][0.50]
arrival = nb03["central_models"][0.50]
test_trips = sorted(str(t) for t in nb03["test_ids"])

bundle = dict(
    energy=dict(**pipeline_json(energy, stable), target="energy consumption [kWh/km]",
                trained_on=f"all {len(X_compact)} valid trips"),
    arrival_soc_50=dict(**pipeline_json(arrival, X50.columns), target="arrival SOC (fraction 0-1)",
                        horizon="first 50% of the trip", trained_on="training trips of notebook 03",
                        test_trips=test_trips),
)
json.dump(bundle, open(os.path.join(PROJECT_ROOT, "models", "model4_energy_arrival_soc.json"), "w"), indent=1)

# ---- demo trips: notebook 03 test trips with the highest / lowest predicted energy use
candidates = []
for idx in X_compact.index:
    trip = trip_of.loc[idx]
    if trip in test_trips and trip in [str(t) for t in X50.index]:
        e_pred = float(energy.predict(X_compact.loc[[idx], stable])[0])
        candidates.append((e_pred, trip, idx))
candidates.sort()
demo = {}
for driver, (e_pred, trip, idx) in ((17, candidates[-1]), (24, candidates[0])):
    x50 = X50.loc[[t for t in X50.index if str(t) == trip]]
    demo[str(driver)] = dict(
        trip=trip,
        energy_features=dict(zip(stable, X_compact.loc[idx, stable].astype(float).tolist())),
        arrival_features=dict(zip([str(c) for c in X50.columns], x50.iloc[0].astype(float).tolist())),
        reference=dict(energy_kwh_per_km=e_pred, arrival_soc=float(arrival.predict(x50)[0])),
        actual=dict(energy_kwh_per_km=float(y_energy.loc[idx]), arrival_soc=float(y50.loc[x50.index[0]])),
    )
out = os.path.join(PROJECT_ROOT, "esp32", "tools", "vectors", "model4_demo_trips.json")
json.dump(demo, open(out, "w"), indent=1)
for driver, d in demo.items():
    print(f"driver {driver}: {d['trip']} | energy {d['reference']['energy_kwh_per_km']:.4f} kWh/km "
          f"(actual {d['actual']['energy_kwh_per_km']:.4f}) | arrival SOC {d['reference']['arrival_soc'] * 100:.1f}% "
          f"(actual {d['actual']['arrival_soc'] * 100:.1f}%)")
print("written: models/model4_energy_arrival_soc.json,", os.path.relpath(out, PROJECT_ROOT))
