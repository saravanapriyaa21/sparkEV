# API Reference

The SparkEV server (`backend/server.py`) is a FastAPI application. It receives prediction results
from the ESP32, ranks the charging stations (Module 2), turns everything into recommendations, and
provides the latest result to the dashboard.

Base URL when running locally: `http://localhost:8000`

FastAPI also provides interactive documentation at http://localhost:8000/docs while the server
is running.

| Method | Endpoint | Used by | Purpose |
|---|---|---|---|
| GET | `/` | Anyone | Check that the server is running |
| POST | `/esp32/result` | ESP32 | Send the results of the four on-device modules |
| GET | `/latest` | Dashboard | Get the most recent result with recommendations |
| POST | `/whatif` | Dashboard | Compare the driver's own choice with the recommendation (What-If Simulator) |
| POST | `/predict` | Developers | Run the models on the computer, for testing without hardware |
| GET | `/fl/global/{task}` | ESP32 | Download the current global model of one federated task |
| POST | `/fl/update/{task}` | ESP32 | Upload the car's trained weights; returns the new global model |
| POST | `/fl/report` | ESP32 | Report the results of an on-device federated learning session |
| GET | `/fl/status` | Dashboard | Federated learning status of the server and the car's latest report |

---

## GET /

Health check.

**Response**

```json
{
  "project": "SparkEV",
  "status": "running",
  "model_1": "loaded",
  "model_2": "loaded",
  "model_3": "loaded",
  "model_4": "loaded",
  "federated_learning": ["m1", "m2c", "m2u", "m3", "m4"],
  "edge_endpoint": "/esp32/result",
  "dashboard_endpoint": "/latest"
}
```

---

## POST /esp32/result

Called by the ESP32 after it has run the models of the four prediction modules (1 to 4). The server then runs the
part of Module 2 that needs data from all charging sites: it predicts how busy each candidate
station will be in the next hour (site congestion model) and ranks the stations (decision
layer). The predicted charge on arrival from Module 4 is the starting charge for this ranking.
The server then generates recommendations and stores the result for the dashboard.

**Request body**

| Field | Type | Description |
|---|---|---|
| `driver_id` | int | Demonstration driver, `17` or `24` |
| `stress` | float | Module 1: Battery Stress Index (0 to 100) |
| `stress_status` | string | Module 1: stress level |
| `model_1_latency_ms` | float | Module 1 prediction time on the ESP32 |
| `model_2_charge_need_pct` | float | Module 2: safe (85th-percentile) estimate of the battery the driver will use before the next charge (%) |
| `model_2_target_soc` | float | Module 2: recommended charge level, charge need + 15% reserve, between 30 and 95% |
| `model_2_urgency` | string | Module 2: `Time-critical`, `Moderate` or `Flexible` |
| `model_2_urgency_class` | int | Module 2: `0`, `1` or `2` (same order) |
| `model_2_battery_kwh` | float | Module 2: battery capacity of the vehicle (kWh) |
| `model_2_latency_ms` | float | Module 2 prediction time on the ESP32 |
| `soh` | float | Module 3: forecast State of Health 10 charge cycles ahead (%) |
| `soh_now` | float | Module 3: State of Health of the newest cycle (%) |
| `battery_risk` | string | Module 3: risk level of the forecast SOH (`LOW` at 85% or more, `MODERATE` from 80%, `HIGH` from 70%, `CRITICAL` below) |
| `model_3_latency_ms` | float | Module 3 prediction time on the ESP32 |
| `model_4_energy_kwh_per_km` | float | Module 4: expected energy consumption of the trip (kWh/km) |
| `model_4_arrival_soc_pct` | float | Module 4: expected battery charge on arrival (%) |
| `model_4_latency_ms` | float | Module 4 prediction time on the ESP32 |
| `edge_inference_latency_ms` | float | Total prediction time for Modules 1 to 4 |

**Example request**

```json
{
  "driver_id": 17,
  "stress": 62.39,
  "stress_status": "MODERATE",
  "model_1_latency_ms": 3.37,
  "model_2_charge_need_pct": 69.40,
  "model_2_target_soc": 84.4,
  "model_2_urgency": "Time-critical",
  "model_2_urgency_class": 0,
  "model_2_battery_kwh": 24.81,
  "model_2_latency_ms": 2.18,
  "soh": 88.30,
  "soh_now": 87.56,
  "battery_risk": "LOW",
  "model_3_latency_ms": 3.93,
  "model_4_energy_kwh_per_km": 0.2041,
  "model_4_arrival_soc_pct": 55.4,
  "model_4_latency_ms": 0.12,
  "edge_inference_latency_ms": 9.64
}
```

**Response**

```json
{
  "status": "received",
  "driver_id": 17,
  "prediction_id": 1,
  "recommendations": ["Frequent hard acceleration detected; use smoother acceleration.", "..."]
}
```

**Errors**

| Status | Reason |
|---|---|
| 400 | `driver_id` is not one of the configured demonstration drivers |
| 422 | A required field is missing or has the wrong type |

---

## GET /latest

Returns the most recent result. The dashboard calls this endpoint every second.

**Response before any result has arrived**

```json
{
  "status": "waiting",
  "message": "Waiting for ESP32 data..."
}
```

**Response after a result has arrived**

All fields from the request, plus:

| Field | Type | Description |
|---|---|---|
| `charging_recommendation` | object | Module 2 station ranking, see below |
| `recommendations` | list of strings | Driving, charging, battery and energy advice, built from SHAP (same texts as `insights`) |
| `insights` | list | The same advice with its source: `module`, `topic` (customer label: Driving, Charging, Battery or Trip), `text`, `kind` (`action`: SHAP shows a controllable factor making the prediction worse; `info`: explanation or decision) |
| `explanations` | object | SHAP explanation of each module's latest prediction: `stress`, `charging`, `battery`, `energy`, and `shap_ms` (time to compute) |
| `prediction_id` | int | Number of results received since the server started |
| `server_latency_ms` | float | Time the server took to process the result |

The server normalises `battery_risk` to `Low`, `Medium` or `High` before storing it.

Each entry of `explanations` contains `module`, `title`, `method` (for example `TreeSHAP (exact)`),
`unit`, `base_value` (prediction for the average case), `prediction` (base value plus all SHAP
values), `contributions` (the 6 largest factors: `factor`, `shap`, `value`) and `advice`.
See [Explainable recommendations](research.md#explainable-recommendations-shap).

`charging_recommendation` contains:

| Field | Type | Description |
|---|---|---|
| `target_soc` | float | Recommended charge level (%) |
| `current_soc` | float | Starting charge used for the ranking: the predicted charge on arrival (%) |
| `need_kwh` | float | Energy to charge to reach the target (kWh) |
| `recommended_station` | string or null | Best station; `null` when no charging is needed now |
| `message` | string | One-line recommendation |
| `stations` | list | Candidate stations, best first: `name`, `site`, `distance_km`, `power_kw`, `wait_risk` (`Low`/`Medium`/`High`), `travel_min`, `wait_min`, `charge_min`, `total_min`, `score` |
| `latency_ms` | float | Time the server took to predict congestion and rank the stations |

The candidate stations are a demonstration setup in `backend/model2_charging_recommendation.py`:
distance and charger power are set by hand, and each station's expected busyness is predicted by
the site congestion model from the history of one real site in the dataset.

---

## POST /whatif

Used by the What-If Simulator (Module 5) in the dashboard. It takes the driver's own choice, predicts its
consequences for the most recent result, and returns them next to the recommended option. The
reference is the recommended station and charge level; when no charging is recommended, it is
"no charging" at Station 1.

**Request body**

| Field | Type | Description |
|---|---|---|
| `station` | string | `Station 1`, `Station 2` or `Station 3` |
| `target_soc` | float | Charge level to charge to (%), 30 to 100 |
| `power_kw` | float or null | Charger power (kW); `null` uses the station's charger. The car charges at no more than 11 kW |
| `driving_style` | float | Driving intensity relative to now (0.5 to 1.5; 1.0 = unchanged). Scales acceleration, jerk, hard acceleration and braking, and the aggressiveness score |
| `speed` | float | Speed relative to now (0.5 to 1.5; the dashboard offers 0.8 to 1.2) |
| `climate` | float | Heating and air-conditioning use relative to now (0.5 to 1.5) |

**Example**

```bash
curl -X POST http://localhost:8000/whatif \
     -H "Content-Type: application/json" \
     -d '{"station": "Station 1", "target_soc": 100, "power_kw": 3.7}'
```

**Response**

| Field | Description |
|---|---|
| `recommended` | All predicted values for the recommended option |
| `scenario` | All predicted values for the driver's choice |
| `comparison` | One row per outcome: `key`, `label`, `unit`, `recommended`, `scenario`, `difference` |

Outcomes: total, charging and waiting time (min), energy charged (kWh), range added (km), battery
wear of the charge relative to the recommended charge, capacity lost per 100 such charges
(percentage points), ageing while parked at the reached level (relative to 50%), driving
stress (0 to 100), energy use (kWh/km) and range after charging (km).

**Errors**

| Status | Reason |
|---|---|
| 400 | No result received yet, unknown station, or a value outside its range |

---

## POST /predict

Runs the models of Modules 1 to 4 on the computer for one demonstration driver, using the
data in `data/raw/` and the demo inputs in `esp32/tools/vectors/`. It is intended for testing
without the ESP32. The result is also stored, so it
appears on the dashboard.

**Request body**

```json
{ "driver_id": 17 }
```

**Example**

```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"driver_id": 17}'
```

**Response:** the same kind of result as `GET /latest`, with prediction times measured on the
computer.

---

## On-device federated learning

While the car is parked or charging, the ESP32 takes part in one federated round for each
federated model. It downloads the global model, trains it on its own data, uploads only the
weights and installs the new global model that the server returns. The server aggregates the
car's update with a simulated fleet (the other clients of each module's notebook). The
implementation is in `backend/federated.py`, and the firmware side is in
`esp32/firmware/SparkEV_Integrated/fl_runtime.h` and `fl_tasks.h`.

| Task | Model | Strategy | Weights exchanged |
|---|---|---|---|
| `m1` | Module 1 battery stress MLP | FedPer | shared base (5,312 bytes); the head stays on the car |
| `m2c` | Module 2 charge need MLP | FedAvg, FedAdam server optimiser | full network (26,116 bytes) |
| `m2u` | Module 2 urgency MLP | FedPer, FedAdam server optimiser | shared base (19,840 bytes); the head stays on the car |
| `m3` | Module 3 SOH forecasting MLP | FedAvg | layers 2 and 3 (8,452 bytes); layer 1 is frozen on the car |
| `m4` | Module 4 energy Elastic Net | FedAvg | coefficients and intercept (96 bytes) |

Weights travel as raw little-endian float32 arrays (`application/octet-stream`), in the order
W, b of each exchanged layer, with W stored as `[inputs][outputs]`.

### GET /fl/global/{task}

Returns the current global weights of the task.

Response headers: `X-FL-Round` (rounds completed so far), `X-FL-Floats` (number of values).

### POST /fl/update/{task}?client=car17&n=86

Body: the car's trained weights, same layout as `/fl/global`. `n` is the number of local
training samples and is used as the aggregation weight. In each round the server also trains
`SPARKEV_FL_CLIENTS_PER_ROUND` simulated cars (default 10; Module 4 has 3 other clients) with the same local
procedure, then aggregates all updates. Module 2 uses the FedAdam server optimiser of its
notebook; the other tasks use weighted averaging.

Response: the new global weights. Response headers: `X-FL-Round`, `X-FL-Participants` (cars
in the round, including the ESP32), `X-FL-Server-Ms`.

| Status | Reason |
|---|---|
| 400 | Wrong number of values, values that are not finite, or `n` below 1 |
| 404 | Unknown task |

### POST /fl/report

Called by the ESP32 after a session.

```json
{
  "driver_id": 17,
  "session": 1,
  "session_ms": 1250.0,
  "free_heap_min": 120000,
  "tasks": [
    {
      "task": "m1", "trained": true, "online": true, "samples": 86,
      "loss_before": 0.0068, "loss_local": 0.0010, "loss_global": 0.0021,
      "download_ms": 45.0, "train_ms": 12.3, "upload_ms": 110.0,
      "bytes_up": 5312, "bytes_down": 10624, "server_round": 1, "participants": 11,
      "scratch_bytes": 16484
    }
  ]
}
```

The losses are measured on the car's own local data: at the start of the round, after local
training, and after installing the new global model.

### GET /fl/status

Returns, for each task, its label, strategy, the number of rounds so far, the size of the
weights exchanged, the size of the simulated fleet and the last round. Also returns the car's latest
report (`latest_report`, or `null` before the first session).

The global models are kept in memory. When the server restarts, they start again from the
deployed models.
