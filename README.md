# SparkEV

**A personalized, predictive and explainable EV intelligence system using federated personalization.**

SparkEV turns real-world electric vehicle data into guidance that is specific to each driver,
predicts how the battery will behave, and explains the reasons behind each prediction. Raw vehicle
data stays private through personalized federated learning.

This is the main repository of the project. It contains the research for all four modules, the
federated learning experiments, the trained models, and the on-device deployment on an ESP32
microcontroller with a server and dashboard.

Project website: https://spark-ev.vercel.app

---

## Contents

1. [Overview](#overview)
2. [Modules](#modules)
3. [Federated learning](#federated-learning)
4. [On-device deployment](#on-device-deployment)
5. [Research roadmap](#research-roadmap)
6. [Repository structure](#repository-structure)
7. [Quick start](#quick-start)
8. [Documentation](#documentation)
9. [Datasets](#datasets)
10. [Team](#team)
11. [Citation](#citation)
12. [License](#license)

---

## Overview

Electric vehicles today mostly report fixed information: state of charge, estimated range and
charging status. Every driver receives the same advice, regardless of how they drive, when they
charge, or the conditions their battery operates in. These systems do not predict how the battery
will age, and they do not explain why a battery is at risk. Personalizing this advice would
normally require collecting every driver's data on a central server, which raises serious privacy
concerns.

SparkEV addresses this with four prediction modules and three ideas that apply across all of them:

- **Explainability.** SHAP shows which factors caused each prediction, so the driver learns why,
  not only what.
- **Privacy through federated learning.** With FedPer, each vehicle shares a common base model
  trained across the fleet and keeps a small personal layer for its own driver. Only model
  updates are shared. Raw data stays in the vehicle.
- **On-device intelligence.** Federated learning needs the models to run on the vehicle itself,
  so SparkEV's models are tested on low-power hardware.

---

## Modules

### Module 1: Driving behaviour and battery stress

Measures how much stress each driving pattern places on the battery and turns it into advice the
driver can act on. A physics-informed Battery Stress Index (0 to 100) is built from battery
temperature, C-rate and depth of discharge, and an XGBoost model predicts it from driving
behaviour alone.

- **Result:** R² = 0.831 (trip-grouped cross-validation), 0.805 on unseen trips
- **Notebooks:** [`notebooks/module1_battery_stress/`](notebooks/module1_battery_stress/)

### Module 2: Charging-session anomaly detection

Detects charging sessions that are likely to be faulty. An XGBoost classifier labels each of
2,649 real charging sessions as Normal or Faulty from 57 session features, and is compared with
Isolation Forest, LSTM and TCN baselines.

- **Result:** 99.62% accuracy, F1 = 0.981, 52 of 54 faults found with no false alarms
- **Notebooks:** [`notebooks/module2_charging_anomaly/`](notebooks/module2_charging_anomaly/)

### Module 3: Battery health and degradation risk

Forecasts the battery's State of Health (SOH) 10 cycles ahead and warns when the battery starts
to wear out faster than its normal trend, with a risk level of LOW, MEDIUM, HIGH or CRITICAL.

- **Result:** R² = 0.959 with a direct autoregressive model
- **Notebooks:** [`notebooks/module3_battery_health/`](notebooks/module3_battery_health/)

### Module 4: Energy consumption and arrival SOC

Predicts how much energy a trip will use, and how much charge will be left on arrival from what
is observed during the first part of the trip. Both tasks use Elastic Net models.

- **Result:** R² = 0.897 for energy consumption; R² = 0.932 to 0.984 for arrival SOC
- **Notebooks:** [`notebooks/module4_energy_and_soc/`](notebooks/module4_energy_and_soc/)

The method, data and full results of each module are described in
[docs/research.md](docs/research.md).

---

## Federated learning

FedAvg and FedPer were compared with centralized training for Modules 1 and 4. For arrival SOC,
federated learning almost matches centralized training (R² 0.973 against 0.976). For battery
stress, FedPer gives a lower error than FedAvg for 32 of 51 clients.

Details: [docs/research.md](docs/research.md#federated-learning)

---

## On-device deployment

The XGBoost models of Modules 1, 2 and 3 run on an **ESP32** microcontroller (240 MHz, about
520 KB of RAM). Each model is converted into plain C arrays, so the chip needs no machine
learning library. The ESP32 shows the results on an OLED display and sends them over Wi-Fi to a
FastAPI server, which generates recommendations for a React dashboard.

```
+---------------------------+   Wi-Fi (JSON)   +------------------------+   every second   +-------------------+
|  ESP32 with OLED display  | ---------------> |  FastAPI server        | <--------------- |  React dashboard  |
|                           |                  |  backend/server.py     |                  |  dashboard/       |
|  Module 1: stress         |                  |                        |                  |                   |
|  Module 2: charging       |                  |  - validates results   |                  |  - scores         |
|  Module 3: battery health |                  |  - generates advice    |                  |  - risk levels    |
+---------------------------+                  +------------------------+                  |  - advice         |
                                                                                           +-------------------+
```

| Model | Prediction time on the ESP32 |
|---|---|
| Module 1: battery stress | 3.35 ms |
| Module 2: charging anomaly | 1.92 ms |
| Module 3: battery health | 3.38 ms |
| **All three** | **8.65 ms** |

A full prediction and recommendation cycle completes well within one second. The network, not
the model, is the slowest part.

Details: [docs/esp32_deployment.md](docs/esp32_deployment.md)

---

## Research roadmap

| Phase | Description | Status |
|---|---|---|
| 1 | Real-world dataset collection and preparation | Complete |
| 2 | Physics-informed battery stress index | Complete |
| 3 | Predictive models for all four modules | Complete |
| 4 | Federated learning (FedAvg and FedPer), evaluated in simulation | Complete |
| 5 | On-device deployment on a microcontroller (ESP32) | Complete |
| 6 | Real-world deployment: live vehicle data, on-road testing in campus and shuttle EVs | Planned |

---

## Repository structure

```
SparkEV/
├── notebooks/                       Research, one folder per module
│   ├── module1_battery_stress/
│   ├── module2_charging_anomaly/
│   ├── module3_battery_health/
│   └── module4_energy_and_soc/
│
├── models/                          Trained models used by the server and the ESP32
├── backend/                         FastAPI server
├── esp32/
│   ├── firmware/SparkEV_Integrated/ Arduino program and the models converted to C
│   └── tools/                       Scripts that convert models and input data to C
├── dashboard/                       React web dashboard
│
├── docs/                            Documentation
├── data/raw/                        Location for datasets (not included)
├── requirements.txt                 Python packages for the server and tools
└── requirements-notebooks.txt       Extra Python packages for the notebooks
```

---

## Quick start

The datasets are not included. Before starting the server, add them to `data/raw/` as described
in the [installation guide](docs/installation.md#3-add-the-datasets).

**Server**

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn server:app --app-dir backend --host 0.0.0.0 --port 8000
```

**Dashboard** (in a second terminal)

```bash
cd dashboard
npm install
npm run dev
```

Open http://localhost:5173.

**Test without the ESP32**

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"driver_id": 17}'
```

For the ESP32 setup, wiring and the notebooks, see the full
[installation guide](docs/installation.md).

---

## Documentation

| Document | Contents |
|---|---|
| [Installation guide](docs/installation.md) | Requirements, datasets, server, dashboard, ESP32 wiring and upload, notebooks, troubleshooting |
| [Research summary](docs/research.md) | Method, data and results of each module and of federated learning |
| [ESP32 deployment](docs/esp32_deployment.md) | How models are converted to C, firmware overview, measured performance, updating the models |
| [API reference](docs/api.md) | Server endpoints with request and response formats |

---

## Datasets

| Module | Dataset |
|---|---|
| 1 and 4 | Technical University of Munich, *Battery and heating data in real driving cycles* (70 trips, 1,355 km) |
| 1 (stress index calibration) | NASA battery ageing data and CALCE battery data |
| 2 | Autosun and The Hong Kong Polytechnic University, EV battery charging dataset (2,649 sessions) |
| 3 | Real-world EV charging and battery SOH datasets, Zenodo, [doi:10.5281/zenodo.16868546](https://doi.org/10.5281/zenodo.16868546) |

---

## Team

Amrita Vishwa Vidyapeetham, Coimbatore, India

| Name | Role |
|---|---|
| Dr. Ramya G. R. | Faculty, Department of Computer Science and Engineering |
| Dr. Umashankar S. | Faculty, Department of Electrical and Electronics Engineering |
| Shravanthi S. | Module 3: battery health and degradation risk |
| Saravana Priyaa C. R. | Hardware deployment (ESP32) |
| Gayathri U. | Federated learning (FedPer) and Module 4: energy consumption |
| Rithanya K. A. | Module 1: driving behaviour and battery stress |
| Irene Divya J. | Module 2: charging anomaly detection |

---

## Citation

If you use this software or the research it contains, please cite:

> Ramya G. R., Umashankar S., Shravanthi S., Saravana Priyaa C. R., Gayathri U., Rithanya K. A.,
> Irene Divya J. *SparkEV: A Personalized, Predictive, and Explainable Framework for Intelligent
> Electric Vehicle Management.*

Citation details are also available in [CITATION.cff](CITATION.cff).

---

## License

This project is licensed under the [Apache License 2.0](LICENSE).

You may use, modify and share this work under the terms of the license. Any redistribution must
keep the copyright notice, the license and the [NOTICE](NOTICE) file, and must state any changes
made.
