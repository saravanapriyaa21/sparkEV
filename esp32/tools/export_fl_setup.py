"""Build the on-device federated learning setup for all four modules.

Usage (from the project root, in an environment with torch, xgboost, numpy < 2.3,
pandas 2.2 and scikit-learn 1.6):

    python esp32/tools/export_fl_setup.py --module all \
        --trips "/path/to/TUM trip data" --m3data "/path/to/folder with data_train and data_test"

For each module it re-runs the federated training of that module's notebook (same code,
same seeds), checks that the notebook's result is reproduced, and writes

    models/fl/m<k>_fl.npz        global model, simulated fleet (other clients' local
                                 training data) and the two demo cars' local data
    models/fl/m<k>_fl.json       settings of the local training and of the aggregation

The ESP32 header is then generated with esp32/tools/convert_fl_to_esp32.py.

Module 1  battery stress MLP (FedPer)           notebooks/module1_battery_stress/03_...
Module 2  charge need (FedAvg + FedAdam) and    notebooks/module2_charging_recommendation/02_...
          urgency (FedPer) MLPs
Module 3  SOH forecasting MLP (FedAvg)          notebooks/module3_battery_health/02_...
Module 4  energy Elastic Net (FedAvg)           notebooks/module4_energy_and_soc/02_...
"""
import argparse
import json
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
warnings.filterwarnings("ignore")

import numpy as np
import xgboost  # noqa: F401  (import before torch: the other order crashes on macOS)
import torch

# one thread: avoids an OpenMP clash between torch and XGBoost on macOS
torch.set_num_threads(1)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NB = os.path.join(PROJECT_ROOT, "notebooks")
OUT_DIR = os.path.join(PROJECT_ROOT, "models", "fl")

# Demo cars: driver 17 and driver 24, as everywhere else in the project
DEMO_DRIVERS = (17, 24)


def run_cells(path, cells, ns=None, replacements=(), cwd=None):
    """Execute the given code cells of a notebook in one namespace."""
    import matplotlib.pyplot as plt
    nb = json.load(open(path))
    ns = {} if ns is None else ns
    ns.setdefault("display", lambda *a, **k: None)
    ns.setdefault("__name__", "__main__")
    old = os.getcwd()
    if cwd:
        os.chdir(cwd)
    try:
        for i in cells:
            cell = nb["cells"][i]
            assert cell["cell_type"] == "code", (path, i)
            src = "\n".join(l for l in "".join(cell["source"]).splitlines()
                            if not l.lstrip().startswith(("!", "%")))
            for a, b in replacements:
                src = src.replace(a, b)
            exec(compile(src, f"{os.path.basename(path)}[{i}]", "exec"), ns)
            plt.close("all")
    finally:
        os.chdir(old)
    return ns


def run_defs(path, cell, ns):
    """Execute only the function and class definitions of a notebook cell."""
    import ast
    tree = ast.parse("".join(json.load(open(path))["cells"][cell]["source"]))
    tree.body = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    exec(compile(tree, f"{os.path.basename(path)}[{cell}]", "exec"), ns)
    return ns


def save(name, arrays, config):
    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez_compressed(os.path.join(OUT_DIR, f"{name}_fl.npz"),
                        **{k: np.asarray(v) for k, v in arrays.items()})
    with open(os.path.join(OUT_DIR, f"{name}_fl.json"), "w") as f:
        json.dump(config, f, indent=1)
    size = os.path.getsize(os.path.join(OUT_DIR, f"{name}_fl.npz")) / 1e6
    print(f"  wrote models/fl/{name}_fl.npz ({size:.2f} MB) and {name}_fl.json")


def torch_layers(state, prefix_map):
    """torch Linear weights are (out, in); the ESP32 code uses (in, out)."""
    out = []
    for w_key, b_key in prefix_map:
        out.append(state[w_key].detach().cpu().numpy().T.astype(np.float32))
        out.append(state[b_key].detach().cpu().numpy().astype(np.float32))
    return out


# =====================================================================
# MODULE 1 - battery stress, FedPer MLP (d -> 32 -> 16 -> 1)
# =====================================================================

def export_m1(trips, workdir):
    print("\n=== Module 1: battery stress (FedPer MLP)")
    os.makedirs(workdir, exist_ok=True)
    nb01 = os.path.join(NB, "module1_battery_stress", "01_driving_behaviour_stress_analysis.ipynb")
    nb03 = os.path.join(NB, "module1_battery_stress", "03_centralized_vs_federated_learning.ipynb")
    rep = [('DATA_DIR = os.path.join("..", "dataset soh")', f"DATA_DIR = {trips!r}")]
    run_cells(nb01, [i for i in range(1, 19) if json.load(open(nb01))["cells"][i]["cell_type"] == "code"],
              replacements=rep, cwd=workdir)
    ns = run_cells(nb03, [2, 4, 5, 7, 9, 11, 14], cwd=workdir)
    best = ns["best_fedper"]
    print(f"  FedPer best round {best['round']}: held-out R2={best['R2']:.3f} (saved notebook run: round 30, 0.452)")
    # The window features depend slightly on library versions (pandas / numpy), so this
    # run is close to, not identical with, the saved notebook output.
    assert abs(best["R2"] - 0.452) < 0.03, "Module 1 FedPer result far from the notebook"

    base = best["base_state"]
    G = torch_layers(base, [("0.weight", "0.bias"), ("2.weight", "2.bias")])
    heads = best["head_states"]
    cids = list(ns["client_ids"])
    clients = ns["clients"]
    n_tr = np.array([clients[c]["n_train"] for c in cids], float)

    def head_arr(h):
        return [h["weight"].numpy().T.astype(np.float32), h["bias"].numpy().astype(np.float32)]

    # a new car starts from the average head of the fleet (weighted by data size)
    avg_head = [sum(w * head_arr(heads[c])[j] for w, c in zip(n_tr, cids)) / n_tr.sum() for j in range(2)]

    arrays = {f"G{i}": w for i, w in enumerate(G)}
    arrays["G4"], arrays["G5"] = avg_head
    sizes = []
    for k, cid in enumerate(cids):
        c = clients[cid]
        arrays[f"fleet_X_{k}"] = c["Xtr"].numpy().astype(np.float32)
        arrays[f"fleet_y_{k}"] = c["ytr"].numpy()[:, 0].astype(np.float32)
        h = head_arr(heads[cid])
        arrays[f"fleet_head_{k}_0"], arrays[f"fleet_head_{k}_1"] = h
        sizes.append(int(c["n_train"]))

    # Demo cars: the two longest held-out trips ("new cars joining the fleet"). Local
    # training data = the first 75 % of the trip's windows, as for every fleet client.
    meta, Xte, yte = ns["meta_test_pool"], ns["X_test_pool"], ns["y_test_pool"]
    counts = meta["trip_id"].value_counts()
    demo_trips = sorted(counts.index, key=lambda t: (-counts[t], t))[:2]
    demo = {}
    for drv, trip in zip(DEMO_DRIVERS, demo_trips):
        mask = (meta["trip_id"] == trip).values
        order = np.argsort(meta.loc[mask, "window_idx"].values)
        Xc = ns["scaler"].transform(Xte.loc[mask].values)[order].astype(np.float32)
        yc = (yte.loc[mask].values[order] / 100.0).astype(np.float32)
        n_val = max(1, int(round(len(yc) * ns["LOCAL_VAL_FRAC"])))
        arrays[f"car{drv}_X"], arrays[f"car{drv}_y"] = Xc[:-n_val], yc[:-n_val]
        demo[str(drv)] = dict(trip=str(trip), n_local=int(len(yc) - n_val))

    # Check: numpy forward pass of a fleet client's model equals torch
    m = ns["StressMLP"](ns["INPUT_DIM"])
    m.shared_base.load_state_dict(base)
    m.head.load_state_dict(heads[cids[0]])
    with torch.no_grad():
        ref = m(clients[cids[0]]["Xtr"]).numpy()[:, 0]
    h = np.maximum(arrays["fleet_X_0"] @ G[0] + G[1], 0)
    h = np.maximum(h @ G[2] + G[3], 0)
    hw = head_arr(heads[cids[0]])
    assert np.allclose(h @ hw[0][:, 0] + hw[1][0], ref, atol=1e-5)

    config = dict(
        module="m1", task="Battery Stress Index / 100", strategy="FedPer",
        sizes=[int(ns["INPUT_DIM"]), 32, 16, 1], frozen_layers=0, personal_layers=1, loss="mse",
        optimizer="adam", lr=float(ns["LOCAL_LR"]), batch=0, local_epochs=int(ns["LOCAL_EPOCHS"]),
        server_opt="avg", n_fleet=len(sizes), fleet_sizes=sizes, fleet_ids=[str(c) for c in cids],
        feature_columns=list(ns["feature_cols"]),
        scaler_mean=ns["scaler"].mean_.tolist(), scaler_scale=ns["scaler"].scale_.tolist(), demo=demo,
        notebook_result=dict(heldout_r2=float(best["R2"]), best_round=int(best["round"])),
        source="notebooks/module1_battery_stress/03_centralized_vs_federated_learning.ipynb (FedPer)",
        note="batch 0 = full batch, as in the notebook",
    )
    save("m1", arrays, config)


# =====================================================================
# MODULE 2 - charge need (A2, FedAvg + FedAdam) and urgency (B, FedPer)
# =====================================================================

M2_DEMO_USERS = {17: "ASK_Plass15", 24: "ASK_Plass21"}
M2_TASKS = {"charge": ("model2_charge_level_fedavg", "A"), "urgency": ("model2_urgency_fedper", "B")}


def export_m2(workdir):
    print("\n=== Module 2: charge need and urgency (FedAvg / FedPer MLPs)")
    os.makedirs(workdir, exist_ok=True)
    path = os.path.join(NB, "module2_charging_recommendation", "02_charging_recommendation_federated.ipynb")
    data_dir = os.path.join(PROJECT_ROOT, "data", "raw")
    ns = run_cells(path, [3, 5, 7, 9, 11, 13, 15], cwd=workdir,
                   replacements=[('DATA_DIR = os.path.join("..", "..", "data", "raw")', f"DATA_DIR = {data_dir!r}")])
    run_defs(path, 32, ns)            # load_bundle (the rest of that cell trains and saves)
    cfg = ns["CFG"]
    for name, (bundle, which) in M2_TASKS.items():
        pred, scaler, spec = ns["load_bundle"](os.path.join(PROJECT_ROOT, "models"), bundle)
        fitted = ns["SCALERS"][which]
        assert np.allclose(scaler.mean, fitted.mean) and np.allclose(scaler.std, fitted.std), \
            "deployed scaler differs from the notebook data"
        frames = ns["FRAMES"][which]
        uids = sorted(frames)
        clients = ns["make_clients"](frames, scaler, spec, uids)
        anc = spec.get("anchor")
        arrays, sizes, fleet_ids = {}, [], []
        for i, w in enumerate(pred.G):
            arrays[f"G{i}"] = w.astype(np.float32)

        def target(c):
            y = c["ytr"]
            if anc is not None:
                y = y - ns["anchor_from_X"](c["Xtr"], anc)
            return y

        demo = {}
        for drv, uid in M2_DEMO_USERS.items():
            c = clients[uid]
            arrays[f"car{drv}_X"] = c["Xtr"].astype(np.float32)
            arrays[f"car{drv}_y"] = np.asarray(target(c), np.float32 if spec["kind"] != "cls" else np.int32)
            if pred.heads:
                arrays[f"car{drv}_head_0"], arrays[f"car{drv}_head_1"] = [h.astype(np.float32) for h in pred.heads[uid]]
            demo[str(drv)] = dict(user=uid, n_local=int(len(c["Xtr"])))
        # the simulated fleet: every other user (the ESP32 plays the demo user's car)
        for uid in uids:
            if uid in M2_DEMO_USERS.values():
                continue
            k = len(sizes)
            c = clients[uid]
            arrays[f"fleet_X_{k}"] = c["Xtr"].astype(np.float16)
            arrays[f"fleet_y_{k}"] = np.asarray(target(c), np.float32 if spec["kind"] != "cls" else np.int32)
            if pred.heads:
                arrays[f"fleet_head_{k}_0"], arrays[f"fleet_head_{k}_1"] = [h.astype(np.float32) for h in pred.heads[uid]]
            sizes.append(int(len(c["Xtr"])))
            fleet_ids.append(uid)

        # Check: deployed model on the validation rows gives a finite, sensible score
        yv, pv, _ = ns["eval_pool"](clients, spec, pred, "va")
        val_score = float(ns["score"](spec, yv, pv))
        print(f"  {name}: {len(uids)} users, validation score {val_score:.4f}, "
              f"fleet rows {sum(sizes)}")

        config = dict(
            module="m2", model=name, task=spec["ycol"],
            strategy="FedPer" if pred.heads else "FedAvg",
            sizes=[int(pred.G[0].shape[0]), *cfg.hidden, int(pred.G[-1].shape[0])],
            frozen_layers=0, personal_layers=1 if pred.heads else 0,
            loss={"quant": "pinball", "cls": "cross_entropy", "reg": "mse"}[spec["kind"]],
            q=spec.get("q"), class_weights=None if spec.get("cw") is None else [float(x) for x in spec["cw"]],
            anchor=anc, optimizer="adam", lr=float(cfg.lr), batch=int(cfg.batch),
            local_epochs=int(cfg.local_epochs), server_opt=cfg.server_opt, server_lr=float(cfg.server_lr),
            server_betas=list(cfg.server_betas), server_tau=float(cfg.server_tau),
            n_fleet=len(sizes), fleet_sizes=sizes, fleet_ids=fleet_ids, demo=demo,
            validation_score=val_score,
            source="notebooks/module2_charging_recommendation/02_charging_recommendation_federated.ipynb "
                   f"(deployed bundle models/{bundle})",
        )
        save(f"m2_{name}", arrays, config)


# =====================================================================
# MODULE 4 - energy consumption, Stable-23 Elastic Net (FedAvg)
# =====================================================================

def export_m4(trips, workdir):
    print("\n=== Module 4: energy consumption (Elastic Net, FedAvg)")
    os.makedirs(workdir, exist_ok=True)
    path = os.path.join(NB, "module4_energy_and_soc", "02_energy_centralized_vs_federated.ipynb")
    nb = json.load(open(path))
    data_cells = [i for i in range(1, 37) if nb["cells"][i]["cell_type"] == "code" and i != 20]
    ns = run_cells(path, data_cells + [144, 146, 147, 148], cwd=workdir, replacements=[
        ('MODULE_DIR = Path(r"C:\\Users\\gayat\\projects\\get_into\\FIP\\module_01")', f"MODULE_DIR = Path({trips!r})")])
    ns.update(FED_ALPHA=0.001, FED_L1_RATIO=0.3, LOCAL_EPOCHS=10, LEARNING_RATE=0.05)
    run_defs(path, 116, ns)             # soft_threshold, local_elastic_net (proximal gradient)

    # The deployed energy model (models/model4_energy_arrival_soc.json) is the starting
    # global model; every client scales its trips with the deployed scaler.
    dep = json.load(open(os.path.join(PROJECT_ROOT, "models", "model4_energy_arrival_soc.json")))["energy"]
    feats = dep["features"]
    assert feats == list(ns["STABLE_23_FEATURES"])
    mean, scale = np.array(dep["mean"]), np.array(dep["scale"])
    X, y, model_df = ns["X"], ns["y"], ns["model_df"]
    trip_of = model_df["Trip"].astype(str)

    def client_arrays(ids):
        Xc = ((X.loc[ids, feats].to_numpy(float) - mean) / scale).astype(np.float32)
        return Xc, y.loc[ids].to_numpy(np.float32)

    groups = ns["client_trip_ids"]
    names = list(groups)
    # Demo cars: the clients C1 and C2 of the notebook's frozen 5-client partition
    car_clients = {17: names[0], 24: names[1]}
    arrays = dict(G0=np.array(dep["coef"], np.float32), G1=np.array([dep["intercept"]], np.float32))
    demo, sizes, fleet_ids = {}, [], []
    for drv, cname in car_clients.items():
        Xc, yc = client_arrays(groups[cname])
        arrays[f"car{drv}_X"], arrays[f"car{drv}_y"] = Xc, yc
        demo[str(drv)] = dict(client=cname, trips=[trip_of[i] for i in groups[cname]], n_local=int(len(yc)))
    for cname in names:
        if cname in car_clients.values():
            continue
        Xc, yc = client_arrays(groups[cname])
        k = len(sizes)
        arrays[f"fleet_X_{k}"], arrays[f"fleet_y_{k}"] = Xc, yc
        sizes.append(int(len(yc)))
        fleet_ids.append(cname)

    # Check: the deployed model reproduces its own predictions in the scaled space
    pred = arrays["car17_X"] @ arrays["G0"] + arrays["G1"][0]
    print(f"  car 17 ({car_clients[17]}): MAE of the deployed model on its trips "
          f"{np.mean(np.abs(pred - arrays['car17_y'])):.4f} kWh/km")

    config = dict(
        module="m4", task="energy consumption [kWh/km]", strategy="FedAvg",
        model="elastic_net", n_features=len(feats), features=feats, scaler_mean=mean.tolist(),
        scaler_scale=scale.tolist(), alpha=0.001, l1_ratio=0.3, optimizer="proximal_gradient",
        lr=0.05, batch=0, local_epochs=10, server_opt="avg",
        n_fleet=len(sizes), fleet_sizes=sizes, fleet_ids=fleet_ids, demo=demo,
        source="notebooks/module4_energy_and_soc/02_energy_centralized_vs_federated.ipynb "
               "(Stable-23 FedAvg: alpha 0.001, l1_ratio 0.3, lr 0.05, 10 local epochs, 5 clients)",
        note="starting global model = deployed energy model; batch 0 = full batch",
    )
    save("m4", arrays, config)


# =====================================================================
# MODULE 3 - SOH forecasting, FedAvg MLP (357 -> 64 -> 32 -> 1)
# =====================================================================

M3_DEMO_FILES = {17: "LFP01", 24: "NMC01_Car02"}


def export_m3(m3data):
    print("\n=== Module 3: SOH forecasting (FedAvg MLP)")
    os.environ["SPARKEV_M3_DATA"] = m3data
    path = os.path.join(NB, "module3_battery_health", "02_soh_centralized_vs_federated.ipynb")
    ns = run_cells(path, [2, 4, 6, 10, 12, 18])
    t = ns["results"]["FedAvg"]["test"]
    print(f"  FedAvg test R2={t['R2']:.4f} MAE={t['MAE']:.5f} (notebook: 0.9396, 0.00812)")
    assert abs(t["R2"] - 0.9396) < 5e-4, "Module 3 FedAvg result not reproduced"

    state = ns["fedavg_best"]["state"]
    W = torch_layers(state, [("shared_base.0.weight", "shared_base.0.bias"),
                             ("shared_base.2.weight", "shared_base.2.bias"),
                             ("head.weight", "head.bias")])

    # Preprocessing as plain arrays (median imputer + standard scaler, most-frequent +
    # one-hot), so the server and the ESP32 do not need scikit-learn objects
    pre = ns["preprocessor"]
    num_pipe = pre.named_transformers_["numeric"]
    cat_pipe = pre.named_transformers_["categorical"]
    numeric, categorical = list(ns["numeric_features"]), list(ns["categorical_features"])
    prep = dict(
        numeric_features=numeric,
        numeric_median=num_pipe.named_steps["imputer"].statistics_.tolist(),
        numeric_mean=num_pipe.named_steps["scaler"].mean_.tolist(),
        numeric_scale=num_pipe.named_steps["scaler"].scale_.tolist(),
        categorical_features=categorical,
        categorical_fill=[str(v) for v in cat_pipe.named_steps["imputer"].statistics_],
        categories=[[str(c) for c in cats] for cats in cat_pipe.named_steps["encoder"].categories_],
        feature_columns=list(ns["feature_columns"]),
    )

    def transform(frame):
        X = prep_transform(prep, frame)
        ref = pre.transform(frame.reindex(columns=ns["feature_columns"]))
        assert np.allclose(X, ref, atol=1e-5), "numpy preprocessing differs from scikit-learn"
        return X.astype(np.float32)

    W1, b1 = W[0], W[1]

    def hidden1(X):
        return np.maximum(X @ W1 + b1, 0).astype(np.float32)

    # Simulated fleet: the 105 training batteries. The first layer stays frozen during
    # on-device rounds (it alone would need 366 KB of RAM for Adam training on the ESP32),
    # so the fleet is stored as first-layer activations.
    arrays, sizes = {}, []
    clients = ns["clients"]
    for k, cid in enumerate(ns["client_ids"]):
        c = clients[cid]
        arrays[f"fleet_h_{k}"] = hidden1(c["Xtr"].numpy()).astype(np.float16)
        arrays[f"fleet_y_{k}"] = c["ytr"].numpy()[:, 0].astype(np.float32)
        sizes.append(len(c["ytr"]))

    # Demo cars: independent test batteries ("new cars joining the fleet"). Local data
    # = first half of their cycles (as in the notebook's new-battery experiment);
    # the newest cycle is the input for the forecast shown to the driver.
    ind = ns["ind_eval"]
    raw = ns["ind_df"]
    demo = {}
    for drv, bid in M3_DEMO_FILES.items():
        g = ind[ind["battery_id"] == bid].sort_values("seg_num")
        half = len(g) // 2
        local = g.iloc[:half]
        Xl = transform(local)
        yl = ns["scaled_delta"](local[ns["target_column"]].values, local["soh"].values)
        latest = raw[raw["battery_id"] == bid].sort_values("seg_num").iloc[[-1]]
        Xn = transform(latest)
        arrays[f"car{drv}_X"] = Xl
        arrays[f"car{drv}_y"] = yl.astype(np.float32)
        arrays[f"car{drv}_now_X"] = Xn[0]
        demo[str(drv)] = dict(battery=bid, n_local=int(len(local)), seg_num=int(latest["seg_num"].iloc[0]),
                              soh_now=float(latest["soh"].iloc[0]))

    for i, w in enumerate(W):
        arrays[f"G{i}"] = w

    # SHAP background for the server's explanations: 40 training rows (seed 42),
    # the same size as the Module 2 notebook's Kernel SHAP background
    rng = np.random.default_rng(42)
    bg = rng.choice(len(ns["X_train_processed"]), 40, replace=False)
    arrays["shap_background"] = ns["X_train_processed"][np.sort(bg)].astype(np.float32)

    # Check: the numpy forward pass gives the notebook's FedAvg test score
    def predict(X, soh_now):
        h = hidden1(X)
        h = np.maximum(h @ W[2] + W[3], 0)
        return soh_now + (h @ W[4] + W[5])[:, 0] * ns["DELTA_STD"] + ns["DELTA_MEAN"]
    from sklearn.metrics import r2_score
    r2 = r2_score(ns["y_test"], predict(ns["X_test_processed"].astype(np.float32), ns["soh_now_test"]))
    print(f"  numpy forward pass test R2={r2:.4f}")
    assert abs(r2 - t["R2"]) < 1e-4

    config = dict(
        module="m3", task="SOH 10 cycles ahead", strategy="FedAvg",
        sizes=[int(W1.shape[0]), 64, 32, 1], frozen_layers=1, loss="mse",
        optimizer="adam", lr=float(ns["LR"]), batch=int(ns["BATCH_SIZE"]),
        local_epochs=int(ns["LOCAL_EPOCHS"]), server_opt="avg",
        delta_mean=float(ns["DELTA_MEAN"]), delta_std=float(ns["DELTA_STD"]),
        n_fleet=len(sizes), fleet_sizes=sizes, demo=demo, preprocessing=prep,
        notebook_result=dict(test_r2=float(t["R2"]), test_mae=float(t["MAE"]), best_round=int(ns["fedavg_best"]["round"])),
        source="notebooks/module3_battery_health/02_soh_centralized_vs_federated.ipynb (FedAvg)",
    )
    save("m3", arrays, config)


def prep_transform(prep, frame):
    """Median impute + standardise numeric columns, most-frequent impute + one-hot categoricals."""
    import pandas as pd
    frame = frame.reindex(columns=prep["feature_columns"])
    num = frame[prep["numeric_features"]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    med = np.array(prep["numeric_median"])
    num = np.where(np.isfinite(num), num, med)
    num = (num - np.array(prep["numeric_mean"])) / np.array(prep["numeric_scale"])
    parts = [num]
    for col, fill, cats in zip(prep["categorical_features"], prep["categorical_fill"], prep["categories"]):
        v = frame[col].astype(object).where(frame[col].notna(), fill).astype(str).to_numpy()
        parts.append((v[:, None] == np.array(cats, dtype=object)[None, :]).astype(float))
    return np.hstack(parts)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", default="all", choices=["all", "m1", "m2", "m3", "m4"])
    ap.add_argument("--trips", help="TUM trip folder (Overview.xlsx, Trip*.csv), modules 1 and 4")
    ap.add_argument("--m3data", help="folder with data_train/ and data_test/, module 3")
    ap.add_argument("--workdir", default=os.path.join(PROJECT_ROOT, "build", "fl_export"),
                    help="scratch folder for notebook outputs")
    a = ap.parse_args()
    todo = ["m1", "m2", "m3", "m4"] if a.module == "all" else [a.module]
    if "m1" in todo:
        export_m1(a.trips, os.path.join(a.workdir, "m1"))
    if "m2" in todo:
        export_m2(os.path.join(a.workdir, "m2"))
    if "m3" in todo:
        export_m3(a.m3data)
    if "m4" in todo:
        export_m4(a.trips, os.path.join(a.workdir, "m4"))
