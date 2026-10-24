# API Reference

The SparkEV server (`backend/server.py`) is a FastAPI application. It receives prediction results
from the ESP32, turns them into recommendations, and provides the latest result to the dashboard.

Base URL when running locally: `http://localhost:8000`

FastAPI also provides interactive documentation at http://localhost:8000/docs while the server
is running.

| Method | Endpoint | Used by | Purpose |
|---|---|---|---|
| GET | `/` | Anyone | Check that the server is running |
| POST | `/esp32/result` | ESP32 | Send the results of the three on-device models |
| GET | `/latest` | Dashboard | Get the most recent result with recommendations |
| POST | `/predict` | Developers | Run the models on the computer, for testing without hardware |

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
  "edge_endpoint": "/esp32/result",
  "dashboard_endpoint": "/latest"
}
```

---

## POST /esp32/result

Called by the ESP32 after it has run all three models. The server does not run any model here.
It checks the driver, generates recommendations and stores the result for the dashboard.

**Request body**

| Field | Type | Description |
|---|---|---|
| `driver_id` | int | Demonstration driver, `17` or `24` |
| `stress` | float | Module 1: Battery Stress Index (0 to 100) |
| `stress_status` | string | Module 1: stress level |
| `model_1_latency_ms` | float | Module 1 prediction time on the ESP32 |
| `charging` | string | Module 2: charging result |
| `model_2_prediction` | int | Module 2: `0` = normal, `1` = faulty |
| `model_2_probability` | float | Module 2: probability that the session is faulty |
| `model_2_status` | string | Module 2: status message |
| `model_2_latency_ms` | float | Module 2 prediction time on the ESP32 |
| `soh` | float | Module 3: State of Health (%) |
| `battery_risk` | string | Module 3: risk level (`LOW`, `MODERATE`/`MEDIUM`, `HIGH`, `CRITICAL`) |
| `cap` | float | Module 3: capacity |
| `cap_corr` | float | Module 3: corrected capacity |
| `model_3_latency_ms` | float | Module 3 prediction time on the ESP32 |
| `edge_inference_latency_ms` | float | Total prediction time for all three models |

**Example request**

```json
{
  "driver_id": 17,
  "stress": 62.39,
  "stress_status": "MODERATE",
  "model_1_latency_ms": 3.35,
  "charging": "Faulty",
  "model_2_prediction": 1,
  "model_2_probability": 1.0,
  "model_2_status": "Abnormal charging session",
  "model_2_latency_ms": 1.92,
  "soh": 91.76,
  "battery_risk": "LOW",
  "cap": 129.04,
  "cap_corr": 125.21,
  "model_3_latency_ms": 3.38,
  "edge_inference_latency_ms": 8.65
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
| `recommendations` | list of strings | Combined driving, charging and battery advice |
| `prediction_id` | int | Number of results received since the server started |
| `server_latency_ms` | float | Time the server took to process the result |

The server normalises `battery_risk` to `Low`, `Medium` or `High` before storing it.

---

## POST /predict

Runs all three models on the computer for one demonstration driver, using the datasets in
`data/raw/`. It is intended for testing without the ESP32. The result is also stored, so it
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
