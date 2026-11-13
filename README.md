# SparkEV

**A personalized, predictive and explainable EV intelligence system using federated personalization.**

SparkEV turns real-world electric vehicle data into guidance that is specific to each driver,
predicts how the battery will behave, and explains the reasons behind each prediction. Raw vehicle
data stays private through personalized federated learning.

This is the main repository of the project. It contains the research for all five modules, the
federated learning experiments, the trained models, and the on-device deployment on an ESP32
microcontroller with a server and dashboard.

Project website: https://spark-ev.vercel.app

---

## Contents

1. [Overview](#overview)
2. [Modules](#modules)
3. [Federated learning](#federated-learning)
4. [On-device deployment](#on-device-deployment)
5. [Dashboard](#dashboard)
6. [Research roadmap](#research-roadmap)
7. [Repository structure](#repository-structure)
8. [Quick start](#quick-start)
9. [Documentation](#documentation)
10. [Datasets](#datasets)
11. [Team](#team)
12. [Citation](#citation)
13. [License](#license)

---

## Overview

Electric vehicles today mostly report fixed information: state of charge, estimated range and
charging status. Every driver receives the same advice, regardless of how they drive, when they
charge, or the conditions their battery operates in. These systems do not predict how the battery
will age, and they do not explain why a battery is at risk. Personalizing this advice would
normally require collecting every driver's data on a central server, which raises serious privacy
concerns.

SparkEV addresses this with four prediction modules, a What-If Simulator built on them (Module 5),
and three ideas that apply across all of them:

- **Explainability.** SHAP shows which factors caused each prediction, so the driver learns why,
  not only what. The dashboard's advice is built from the SHAP explanation of the driver's own
  latest predictions, not from fixed rules (see [Explainable recommendations](docs/research.md#explainable-recommendations-shap)).
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

### Module 2: Predictive charging recommendation

Recommends a suitable charging decision from the driver's needs, the battery and the available
charging options, instead of simply pointing to the nearest charger. If Station 1 is closer but
busy or slow, and Station 2 is faster or free, SparkEV can recommend Station 2. If the driver
only needs a little charge before the next trip, it recommends a lower charge level instead of a
full charge. Three models feed a decision layer that ranks the candidate stations:

- **Charge level:** how much battery the driver will need before the next charge, giving a
  recommended target charge level
- **Urgency:** whether the session is time-critical, moderate or flexible
- **Site congestion:** how busy a charging site will be in the next hour

Trained on 34,358 real residential charging sessions of 261 users at 12 sites in Norway.

- **Result:** the recommended charge level lasts until the next charge, with at least 5% left,
  in 94% of sessions, while charging 25 percentage points less than a full charge on average;
  urgency accuracy 70.5%
- **Notebooks:** [`notebooks/module2_charging_recommendation/`](notebooks/module2_charging_recommendation/)

### Module 3: Battery health and degradation risk

Forecasts the battery's State of Health (SOH) 10 cycles ahead and warns when the battery starts
to wear out faster than its normal trend, with a risk level of LOW, MEDIUM, HIGH or CRITICAL.

- **Result:** R² = 0.959 with a direct autoregressive model; the federated network (FedAvg, each
  battery a client) reaches R² = 0.940 on the chronological test set and is the model deployed on the ESP32
- **Notebooks:** [`notebooks/module3_battery_health/`](notebooks/module3_battery_health/)

### Module 4: Energy consumption and arrival SOC

Predicts how much energy a trip will use, and how much charge will be left on arrival from what
is observed during the first part of the trip. Both tasks use Elastic Net models.

- **Result:** R² = 0.897 for energy consumption; R² = 0.932 to 0.984 for arrival SOC
- **Notebooks:** [`notebooks/module4_energy_and_soc/`](notebooks/module4_energy_and_soc/)

### Module 5: What-If Simulator

SparkEV gives one recommendation, but the driver decides. The What-If Simulator in the
dashboard lets the driver try other choices before making them and compares the predicted
consequences with SparkEV's recommendation side by side. For example, if SparkEV recommends
Station 2 but Station 1 is more convenient, or the driver wants to charge to 100% instead of
84%, the simulator shows what that changes.

| The driver changes | Predicted consequence | Model |
|---|---|---|
| Charging station | travel, waiting, charging and total time | Module 2 (site congestion and decision layer) |
| Charge level and charger power | energy charged, range added, battery wear of the charge, ageing while parked at that level | Module 2 and a battery ageing model fitted on measured cell ageing data (KIT) |
| Driving style and speed | driving stress (Battery Stress Index) | Module 1 |
| Driving style, speed, heating and air conditioning | energy use (kWh/km) and range | Module 4 physics-based vehicle model |

Only the chosen settings change; everything else stays at the driver's current values, so the
results are model estimates of the trade-offs between convenience, efficiency and battery health.

- **Code:** [`backend/whatif.py`](backend/whatif.py) (server) and the Simulator tab of the dashboard
- **Notebooks:** the models it uses are built in [`notebooks/module1_battery_stress/04_charge_level_ageing_model.ipynb`](notebooks/module1_battery_stress/04_charge_level_ageing_model.ipynb) (battery wear) and [`notebooks/module4_energy_and_soc/04_physics_energy_model.ipynb`](notebooks/module4_energy_and_soc/04_physics_energy_model.ipynb) (energy use)

The method, data and full results of each module are described in
[docs/research.md](docs/research.md).

---

## Federated learning

FedAvg and FedPer were compared with centralized training for all four prediction modules. For arrival
SOC, federated learning almost matches centralized training (R² 0.973 against 0.976). For SOH
forecasting, with each battery as a client, FedAvg matches the same network trained centrally
(R² 0.940 against 0.938). For charging recommendation, with each user as a client, FedAvg
reaches R² 0.236 against 0.249 for the same network trained centrally. For battery stress,
FedPer gives a lower error than FedAvg for 32 of 51 clients.

The federated training also runs **on the ESP32 itself**, as it would in a car. While the car
is parked or charging, the ESP32 downloads the global model of each federated model, trains it
on its own data, uploads only the weights and installs the new global model. The server
aggregates the update with a simulated fleet (the other clients of each notebook). Raw data
never leaves the device, and FedPer heads stay on it.

| Federated model on the ESP32 | Strategy | Local training | Weights sent |
|---|---|---|---|
| Module 1: battery stress MLP | FedPer | 188 ms (86 samples, 10 epochs) | 5.2 KB |
| Module 2: charge need MLP | FedAvg, FedAdam server | 58 ms (27 sessions, 2 epochs) | 25.5 KB |
| Module 2: urgency MLP | FedPer, FedAdam server | 53 ms (31 sessions, 2 epochs) | 19.4 KB |
| Module 3: SOH forecasting MLP | FedAvg (first layer frozen) | 45 ms (62 cycles, 2 epochs) | 8.3 KB |
| Module 4: energy Elastic Net | FedAvg | 0.3 ms (11 trips, 10 epochs) | 96 B |

A full session for all five models, including download, upload and aggregation over Wi-Fi,
takes about 2 seconds. The training on the ESP32 matches the Python reference to within
6 × 10⁻⁶ (relative).

Details: [docs/research.md](docs/research.md#federated-learning) and
[docs/esp32_deployment.md](docs/esp32_deployment.md#on-device-federated-learning)

---

## On-device deployment

The models of the four prediction modules run on an **ESP32** microcontroller (240 MHz, 320 KB
of RAM): the Module 1 XGBoost model, the Module 2 federated networks, the Module 3 federated
network and the Module 4 Elastic Net models. Each model is converted into plain C arrays, so
the chip needs no machine learning library. The ESP32 shows the results on an OLED display and
sends them over Wi-Fi to a FastAPI server, which ranks the charging stations, generates
recommendations and serves a React dashboard. The ESP32 also trains the federated models
(see [Federated learning](#federated-learning)).

```
+---------------------------+   Wi-Fi (JSON)   +------------------------+   every second   +-------------------+
|  ESP32 with OLED display  | ---------------> |  FastAPI server        | <--------------- |  React dashboard  |
|                           |                  |  backend/server.py     |                  |  dashboard/       |
|  Module 1: stress         |                  |  - ranks stations      |                  |  - overview       |
|  Module 2: charging       |  weights (FL)    |  - SHAP explanations   |                  |  - charging plan  |
|  Module 3: battery health | <--------------> |    and advice          |                  |  - insights, SHAP |
|  Module 4: energy and SOC |                  |  - aggregates the      |                  |  - simulator      |
|                           |                  |    federated updates   |                  |  - technical      |
+---------------------------+                  +------------------------+                  +-------------------+
```

| Model | Prediction time on the ESP32 |
|---|---|
| Module 1: battery stress (XGBoost) | 3.37 ms |
| Module 2: charge need and urgency (federated MLPs) | 2.18 ms |
| Module 3: SOH forecast (federated MLP) | 3.93 ms |
| Module 4: energy and arrival SOC (Elastic Net) | 0.12 ms |
| **All four** | **9.64 ms** |

Mean of 100 runs on the ESP32. Sending a result to the server takes 248 ms on average over a
phone hotspot, so the network, not the models, is the slowest part.

Details: [docs/esp32_deployment.md](docs/esp32_deployment.md)

---

## Dashboard

The dashboard is written for the driver, in the style of a car companion app. It uses plain
language and does not show model names. It reads the latest result from the server every second
and has four tabs:

| Tab | What the driver sees |
|---|---|
| Overview | Whether to charge now, the battery level on arrival with the suggested charge level, battery health, driving style and energy efficiency, and the most important personal tips |
| Charging | The suggested charge level and every station with its total time, split into driving, waiting and charging, and the chance of a queue; the reasons for the plan |
| Insights | Tips to try and information in plain language, and the SHAP explanation of each prediction: how much each factor raises or lowers the driver's result compared with an average driver |
| Simulator | The What-If Simulator (Module 5): the driver changes the station, charge level, charger power, driving style, speed or heating and compares the outcome with SparkEV's suggestion |

Every tip comes from SHAP. A tip asks the driver to change something only when SHAP shows that a
factor the driver can control is making their own prediction worse.

The **Technical** button opens a side panel on the same page with the details for developers and
reviewers: the prediction time of each model on the ESP32 and the server processing time, the raw
model outputs, the SHAP method of each model, the latest on-device federated learning round and
the connection status.

- **Code:** [`dashboard/src/App.jsx`](dashboard/src/App.jsx), with the explanations built in
  [`backend/explain.py`](backend/explain.py)

---

## Research roadmap

| Phase | Description | Status |
|---|---|---|
| 1 | Real-world dataset collection and preparation | Complete |
| 2 | Physics-informed battery stress index | Complete |
| 3 | Predictive models for the four prediction modules, and the What-If Simulator (Module 5) | Complete |
| 4 | Federated learning (FedAvg and FedPer), evaluated in simulation and on the ESP32 | Complete |
| 5 | On-device deployment on a microcontroller (ESP32) | Complete |
| 6 | Real-world deployment: live vehicle data, on-road testing in campus and shuttle EVs | Planned |

---

## Repository structure

```
SparkEV/
├── notebooks/                       Research, one folder per module
│   ├── module1_battery_stress/
│   ├── module2_charging_recommendation/
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
| 5 (battery wear in the What-If Simulator) | Luh and Blank, *Comprehensive battery aging dataset: capacity and impedance fade measurements of a lithium-ion NMC/C-SiO cell*, Scientific Data (2024), data [doi:10.35097/1969](https://doi.org/10.35097/1969), CC BY 4.0 |
| 2 | Sørensen, Sartori, Lindberg and Andresen, *Electric vehicle charging dataset with 35,000 charging sessions from 12 residential locations in Norway*, Data in Brief 57 (2024) 110883, [doi:10.5281/zenodo.12730566](https://doi.org/10.5281/zenodo.12730566) |
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
| Irene Divya J. | Module 2: predictive charging recommendation |

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
