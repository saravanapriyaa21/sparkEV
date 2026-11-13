"""Turn the ESP32 benchmark Serial output into the paper export files.

Usage (from the project root):

    python esp32/tools/parse_benchmark_log.py \
        --serial serial_normal.txt [--serial-weak serial_weak.txt] \
        [--server-log server.log] [--out paper_exports]

serial_normal.txt  Serial Monitor text from a BENCHMARK_MODE 1 run
serial_weak.txt    Serial Monitor text from a BENCHMARK_MODE 2 run (optional)
server.log         stdout of `uvicorn server:app ...` during the run (optional)

Writes esp32_benchmark.json, esp32_sends_normal.csv, esp32_parity.csv,
esp32_fl_training.csv, esp32_fl_rounds.csv and, if given, esp32_sends_weak.csv.
Anything that was not measured is written as "not measured".
"""
import argparse
import csv
import io
import json
import os
import re
import statistics

import numpy as np
import pandas as pd
import sys

import xgboost as xgb

PROJECT_ROOT = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
VECTORS_DIR = os.path.join(PROJECT_ROOT, "esp32", "tools", "vectors")
FIRMWARE_DIR = os.path.join(
    PROJECT_ROOT, "esp32", "firmware", "SparkEV_Integrated"
)

NOT_MEASURED = "not measured"

sys.path.insert(0, os.path.join(PROJECT_ROOT, "backend"))


def block(text, name):
    match = re.search(rf"BEGIN_{name}\s*\n(.*?)\nEND_{name}", text, re.S)
    return match.group(1).strip() if match else None


def c_array(path, name, length=None):
    """Read a `static const float NAME[...] = {...};` array from a C file.

    Like C, elements missing from the initializer are zero.
    """
    text = open(path, encoding="utf-8").read()
    body = re.search(rf"{name}\s*\[[^\]]*\]\s*=\s*\{{(.*?)\}};", text, re.S).group(1)
    body = re.sub(r"//[^\n]*", "", body)
    values = []
    for token in body.replace("\n", " ").split(","):
        token = token.strip().rstrip("f")
        if token:
            values.append(float("nan") if token.upper() == "NAN" else float(token))
    if length is not None:
        values += [0.0] * (length - len(values))
    return values


def pc_references():
    """PC predictions for the vectors stored on the ESP32."""
    refs = {}

    m1 = xgb.XGBRegressor()
    m1.load_model(os.path.join(MODELS_DIR, "model1_battery_stress_xgb.json"))
    m1_cols = m1.get_booster().feature_names
    m2_users = json.load(open(os.path.join(VECTORS_DIR, "model2_demo_users.json")))
    m4_trips = json.load(open(os.path.join(VECTORS_DIR, "model4_demo_trips.json")))

    for driver in (17, 24):
        v1 = c_array(os.path.join(FIRMWARE_DIR, "SparkEV_Integrated.ino"), f"MODEL1_DRIVER{driver}_VECTOR", len(m1_cols))
        refs[(driver, "m1", "stress")] = float(m1.predict(pd.DataFrame([v1], columns=m1_cols))[0])

        # Module 2 and 4 references come from the backend implementation
        # (export_model2_demo_users.py / export_model4_models.py)
        r2 = m2_users[str(driver)]["reference"]
        refs[(driver, "m2", "charge_need")] = r2["charge_need_q85_pct"]
        refs[(driver, "m2", "urgency_class")] = float(r2["urgency_class"])
        refs[(driver, "m2", "p_time_critical")] = r2["urgency_probs"][0]
        r4 = m4_trips[str(driver)]["reference"]
        refs[(driver, "m4", "energy")] = r4["energy_kwh_per_km"]
        refs[(driver, "m4", "arrival_soc_pct")] = float(np.clip(r4["arrival_soc"], 0, 1)) * 100

    # Module 3: federated SOH model (backend/model3_soh_federated.py)
    import model3_soh_federated as m3
    for driver, car in m3.DRIVER_BATTERIES.items():
        refs[(driver, "m3", "soh_forecast")] = m3.forecast(car["x"], car["soh_now"])
    return refs


def fl_references(driver):
    """Local loss before and after one training run from the deployed weights,
    computed with the server's reference implementation (backend/federated.py)."""
    import federated as F
    refs = {}
    for t, task in enumerate(["m1", "m2c", "m2u", "m3", "m4"]):
        st = F.TASKS[task]
        z = np.load(os.path.join(F.FL_DIR, f"{F.TASK_FILES[task]}_fl.npz"))
        X, y = z[f"car{driver}_X"], z[f"car{driver}_y"]
        if st.is_en:
            before = F.eval_loss_en(st.G[0], st.G[1][0], X, y)
            coef, b = F.local_train_en(st.G[0], st.G[1][0], X, y, st.cfg)
            refs[task] = (before, F.eval_loss_en(coef, b, X, y))
            continue
        P = [p.copy() for p in st.G]
        if st.personal and f"car{driver}_head_0" in z.files:
            for j, i in enumerate(st.personal):
                P[i] = z[f"car{driver}_head_{j}"]
        if st.first:
            X = np.maximum(X @ P[0] + P[1], 0).astype(np.float32)
        before = F.eval_loss_mlp(P, X, y, st.cfg, st.first)
        newP = F.local_train_mlp(P, X, y, st.cfg, seed=1000 * driver + t, first=st.first)
        refs[task] = (before, F.eval_loss_mlp(newP, X, y, st.cfg, st.first))
    return refs


FL_ROUND_RE = re.compile(
    r"^(?P<name>M\d [a-zA-Z]+)\s+n=\s*(?P<n>\d+)\s+loss (?P<before>[-+0-9.eE]+) -> local (?P<local>[-+0-9.eE]+) "
    r"-> global (?P<glob>[-+0-9.eE]+)\s+train\s+(?P<train>[-0-9.]+) ms\s+down\s+(?P<down>[-0-9.]+) ms\s+"
    r"up\+agg\s+(?P<up>[-0-9.]+) ms\s+(?P<bytes>\d+) B up\s+round (?P<round>\d+) \((?P<cars>\d+) cars\)\s+"
    r"(?P<status>online|offline)", re.M)


def fl_round_rows(serial_text):
    """Per-model lines of the federated learning sessions printed by run_fl_session()."""
    rows, session = [], 0
    for line in serial_text.splitlines():
        m = re.search(r"FEDERATED LEARNING SESSION (\d+)", line)
        if m:
            session = int(m.group(1))
            continue
        m = FL_ROUND_RE.search(line.strip())
        if m:
            rows.append(dict(session=session, model=m["name"], samples=int(m["n"]),
                             loss_before=float(m["before"]), loss_local=float(m["local"]),
                             loss_global=float(m["glob"]), train_ms=float(m["train"]),
                             download_ms=float(m["down"]), upload_aggregate_ms=float(m["up"]),
                             bytes_up=int(m["bytes"]), server_round=int(m["round"]),
                             participants=int(m["cars"]), status=m["status"]))
    return rows


def server_times(server_log_path, prediction_ids):
    if not server_log_path:
        return NOT_MEASURED
    text = open(server_log_path, encoding="utf-8", errors="replace").read()
    pairs = re.findall(r"FastAPI time\s*:\s*([0-9.]+)\s*ms\s*\n\s*Prediction ID:\s*(\d+)", text)
    by_id = {int(pid): float(ms) for ms, pid in pairs}
    values = [by_id[pid] for pid in prediction_ids if pid in by_id]
    if len(values) < 2:
        return NOT_MEASURED
    return {"mean": round(statistics.mean(values), 3), "std": round(statistics.stdev(values), 3),
            "n": len(values)}


def sends_rows(serial_text):
    raw = block(serial_text, "SENDS_CSV")
    return list(csv.DictReader(io.StringIO(raw))) if raw else []


def write_rows(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", required=True)
    parser.add_argument("--serial-weak")
    parser.add_argument("--server-log")
    parser.add_argument("--out", default=os.path.join(PROJECT_ROOT, "paper_exports"))
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    text = open(args.serial, encoding="utf-8", errors="replace").read()
    summary = json.loads(block(text, "SUMMARY_JSON"))

    cycle = block(text, "CYCLE_JSON")
    if cycle:
        cycle = json.loads(cycle)
        summary["cycle_ms"] = cycle["cycle_ms"]
        summary["cycle_ms_values"] = cycle["cycle_ms_values"]
        summary["cpu_busy_share_pct"] = cycle["cpu_busy_share_pct"]
    else:
        summary["cycle_ms"] = summary["cpu_busy_share_pct"] = NOT_MEASURED

    rows = sends_rows(text)
    write_rows(rows, os.path.join(args.out, "esp32_sends_normal.csv"))
    ids = [int(r["prediction_id"]) for r in rows if int(r["prediction_id"]) > 0]
    summary["server_ms"] = server_times(args.server_log, ids)

    if args.serial_weak:
        weak_text = open(args.serial_weak, encoding="utf-8", errors="replace").read()
        summary["weak_wifi"] = json.loads(block(weak_text, "SUMMARY_JSON"))["weak_wifi"]
        write_rows(sends_rows(weak_text), os.path.join(args.out, "esp32_sends_weak.csv"))
    else:
        summary["weak_wifi"] = NOT_MEASURED

    parity = block(text, "PARITY_CSV")
    if parity:
        refs = pc_references()
        rows_out, worst = [], {}
        for r in csv.DictReader(io.StringIO(parity)):
            key = (int(r["driver"]), r["model"], r["output"])
            device = float(r["value"])
            diff = abs(device - refs[key])
            worst[r["model"]] = max(worst.get(r["model"], 0.0), diff)
            rows_out.append({**r, "pc_reference": f"{refs[key]:.6f}", "abs_diff": f"{diff:.6g}"})
        write_rows(rows_out, os.path.join(args.out, "esp32_parity.csv"))
        summary["parity_max_abs_diff"] = {k: float(f"{v:.6g}") for k, v in worst.items()}
    else:
        summary["parity_max_abs_diff"] = NOT_MEASURED

    fl_csv = block(text, "FL_CSV")
    if fl_csv:
        rows_fl = list(csv.DictReader(io.StringIO(fl_csv)))
        driver = int(rows_fl[0]["driver"])
        refs = fl_references(driver)
        out, fl_summary = [], {}
        for r in rows_fl:
            ref_before, ref_after = refs[r["task"]]
            rel = abs(float(r["loss_after"]) - ref_after) / max(abs(ref_after), 1e-12)
            out.append({**r, "pc_loss_before": f"{ref_before:.8f}", "pc_loss_after": f"{ref_after:.8f}",
                        "loss_after_rel_diff": f"{rel:.3g}"})
            fl_summary[r["task"]] = dict(
                samples=int(r["samples"]), exchange_bytes=int(r["exchange_bytes"]),
                scratch_bytes=int(r["scratch_bytes"]), prepare_ms=float(r["prepare_ms"]),
                train_ms=dict(mean=float(r["train_mean_ms"]), median=float(r["train_median_ms"]),
                              p95=float(r["train_p95_ms"]), min=float(r["train_min_ms"]),
                              max=float(r["train_max_ms"])),
                loss_after_rel_diff_vs_pc=float(f"{rel:.3g}"))
        write_rows(out, os.path.join(args.out, "esp32_fl_training.csv"))
        summary["fl_training"] = fl_summary
    else:
        summary["fl_training"] = NOT_MEASURED

    fl_rounds = fl_round_rows(text)
    if fl_rounds:
        write_rows(fl_rounds, os.path.join(args.out, "esp32_fl_rounds.csv"))
        online = [r for r in fl_rounds if r["status"] == "online"]
        summary["fl_rounds"] = dict(
            sessions=max(r["session"] for r in fl_rounds), model_rounds=len(fl_rounds),
            online=len(online),
            median_download_ms=statistics.median(r["download_ms"] for r in online) if online else NOT_MEASURED,
            median_upload_aggregate_ms=statistics.median(r["upload_aggregate_ms"] for r in online) if online else NOT_MEASURED)
    else:
        summary["fl_rounds"] = NOT_MEASURED

    summary.setdefault("power", NOT_MEASURED)
    # getFreeSketchSpace() returns the size of the app partition, not the space left
    if "free_sketch_bytes" in summary:
        summary["app_partition_bytes"] = summary.pop("free_sketch_bytes")
        summary["free_sketch_bytes"] = summary["app_partition_bytes"] - summary["sketch_bytes"]
    # temperatureRead() on chips without a usable sensor returns fixed values
    # (53.33 C, and 29.44 C right after boot), so those are not measurements
    temps = (summary.get("chip_temp_c_start"), summary.get("chip_temp_c_end"))
    if any(isinstance(t, float) and (np.isnan(t) or round(t, 2) in (53.33, 29.44)) for t in temps):
        summary["chip_temp_raw_readings"] = list(temps)
        summary["chip_temp_c_start"] = summary["chip_temp_c_end"] = NOT_MEASURED

    with open(os.path.join(args.out, "esp32_benchmark.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
