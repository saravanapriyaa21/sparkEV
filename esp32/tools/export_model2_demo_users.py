"""Pick the two Module 2 demo users and export their input vectors.

Usage (from the project root, data in data/raw/):

    python esp32/tools/export_model2_demo_users.py

It runs the feature code of notebooks/module2_charging_recommendation/
02_charging_recommendation_federated.ipynb, checks that the backend implementation
(backend/model2_charging_recommendation.py) gives the same predictions as the notebook's
model code, and writes esp32/tools/vectors/model2_demo_users.json.

Demo users are taken from the test sessions:
    driver 17: high predicted charge need (q85 >= 55 %) and predicted Time-critical
    driver 24: low predicted charge need (q85 <= 25 %) and predicted Flexible
"""
import json
import os
import sys
import tempfile

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NB_DIR = os.path.join(PROJECT_ROOT, "notebooks", "module2_charging_recommendation")
NOTEBOOK = os.path.join(NB_DIR, "02_charging_recommendation_federated.ipynb")
OUT_PATH = os.path.join(PROJECT_ROOT, "esp32", "tools", "vectors", "model2_demo_users.json")

sys.path.insert(0, os.path.join(PROJECT_ROOT, "backend"))
import model2_charging_recommendation as m2  # noqa: E402


def notebook_namespace():
    """Execute the notebook's code cells up to the data/federation build."""
    nb = json.load(open(NOTEBOOK))
    ns = {"__name__": "__main__"}
    cwd = os.getcwd()
    os.chdir(NB_DIR)
    try:
        for cell in nb["cells"]:
            if cell["cell_type"] != "code":
                continue
            src = "".join(cell["source"])
            src = src.replace('OUT_DIR = "./module2_fl_outputs"', f'OUT_DIR = {tempfile.mkdtemp()!r}')
            exec(compile(src, "notebook", "exec"), ns)
            if "FRAMES, SCALERS = {}, {}" in src:
                break
        # the bundle helpers are defined later, next to code that needs trained models
        for cell in nb["cells"]:
            src = "".join(cell["source"])
            if cell["cell_type"] == "code" and "def load_bundle" in src:
                exec(compile(src.split("# Deployed federated model")[0], "notebook", "exec"), ns)
    finally:
        os.chdir(cwd)
    return ns


ns = notebook_namespace()
FRAMES = ns["FRAMES"]

# ---- parity: backend implementation vs the notebook's own model code
predA, scA, spA = ns["load_bundle"](m2.MODELS_DIR, "model2_charge_level_fedavg")
predB, scB, spB = ns["load_bundle"](m2.MODELS_DIR, "model2_urgency_fedper")
worst_a, worst_b = 0.0, 0.0
for uid in sorted(FRAMES["A"])[:40]:
    te = FRAMES["A"][uid]["te"]
    ref = ns["fl_predict"](predA, scA, spA, te, uid)
    for k in range(len(te)):
        row = te.iloc[k]
        got = m2.predict_charge_need(row.to_dict(), row["location"])
        worst_a = max(worst_a, abs(got - ref[k]))
for uid in sorted(FRAMES["B"])[:40]:
    te = FRAMES["B"][uid]["te"]
    ref = ns["fl_predict"](predB, scB, spB, te, uid)
    for k in range(len(te)):
        row = te.iloc[k]
        cls, _ = m2.predict_urgency(row.to_dict(), row["location"], str(uid))
        worst_b = max(worst_b, float(cls != ref[k]))
print(f"parity with notebook: charge need max abs diff {worst_a:.2e} %, urgency mismatches {worst_b:.0f}")
assert worst_a < 1e-6 and worst_b == 0

# ---- pick the demo sessions
profiles = {
    17: dict(rule="q85 >= 55 and Time-critical", ok=lambda q, u: q >= 55 and u == 0),
    24: dict(rule="q85 <= 25 and Flexible", ok=lambda q, u: q <= 25 and u == 2),
}
demo = {}
for uid in sorted(set(FRAMES["A"]) & set(FRAMES["B"]) & set(m2.urgency_model.heads)):
    te_a, te_b = FRAMES["A"][uid]["te"], FRAMES["B"][uid]["te"]
    for k in range(len(te_a)):
        row_a = te_a.iloc[k]
        match = te_b[te_b.pt == row_a.pt]
        if match.empty:
            continue
        row_b = match.iloc[0]
        q = m2.predict_charge_need(row_a.to_dict(), row_a["location"])
        u, probs = m2.predict_urgency(row_b.to_dict(), row_b["location"], str(uid))
        for driver, prof in profiles.items():
            if driver not in demo and prof["ok"](q, u):
                demo[driver] = dict(
                    user_id=str(uid), location=str(row_a["location"]),
                    session_plugin_time=str(row_a.pt),
                    rule=prof["rule"],
                    battery_capacity_kwh=float(row_a["battery_capacity"]),
                    charge_level_features={f: (None if not np.isfinite(float(row_a[f])) else float(row_a[f]))
                                           for f in m2.charge_level_model.feats},
                    urgency_features={f: (None if not np.isfinite(float(row_b[f])) else float(row_b[f]))
                                      for f in m2.urgency_model.feats},
                    reference=dict(charge_need_q85_pct=q,
                                   target_soc_pct=float(np.clip(q + m2.RESERVE_PCT, 30, m2.SOC_MAX)),
                                   urgency_class=u, urgency=m2.URGENCY_NAMES[u], urgency_probs=probs),
                    actual_next_need_pct=float(row_a["next_soc_need"]),
                    actual_urgency=m2.URGENCY_NAMES[int(row_b["urg_class"])],
                )
    if len(demo) == len(profiles):
        break

assert len(demo) == len(profiles), f"found only {list(demo)}"
json.dump({str(k): v for k, v in sorted(demo.items())}, open(OUT_PATH, "w"), indent=1)
for driver, d in sorted(demo.items()):
    r = d["reference"]
    print(f"driver {driver}: user {d['user_id']} at {d['location']} | need q85 {r['charge_need_q85_pct']:.1f}% "
          f"-> target {r['target_soc_pct']:.0f}% | {r['urgency']} (actual next need "
          f"{d['actual_next_need_pct']:.0f}%, actual {d['actual_urgency']})")
print("written:", os.path.relpath(OUT_PATH, PROJECT_ROOT))
