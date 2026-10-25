# Research Summary

This document describes the method and results of each SparkEV module and of the federated
learning experiments. The numbers come from the notebooks in `notebooks/`. For the on-device
results, see [ESP32 deployment](esp32_deployment.md).

## Contents

1. [Module 1: Driving behaviour and battery stress](#module-1-driving-behaviour-and-battery-stress)
2. [Module 2: Charging-session anomaly detection](#module-2-charging-session-anomaly-detection)
3. [Module 3: Battery health and degradation risk](#module-3-battery-health-and-degradation-risk)
4. [Module 4: Energy consumption and arrival SOC](#module-4-energy-consumption-and-arrival-soc)
5. [Federated learning](#federated-learning)

---

## Module 1: Driving behaviour and battery stress

**Goal.** Measure how much stress each driving pattern places on the battery, and turn this into
advice the driver can act on.

**Data.** 70 real trips of one electric vehicle in and around Munich (1,355 km, 30.4 hours). Each
trip is split into 60-second windows, giving 3,193 driving windows.

**Method.**
- Battery stress is not measured directly, so a **Battery Stress Index (BSI)** from 0 to 100 is
  built for each window from battery temperature, charge/discharge rate (C-rate), depth of
  discharge and time spent outside the 20 to 80% charge range.
- The temperature and C-rate parts are replaced by a **physics-based capacity-fade model**,
  fitted on NASA and CALCE battery ageing data. This gives a more realistic stress label.
- An XGBoost model predicts the BSI from 14 driving features only (speed, acceleration, jerk,
  hard acceleration and braking, idle time, regenerative braking, aggressiveness, ambient
  temperature) plus route and weather.
- SHAP identifies the driving habits that raise stress the most, and these become plain-language
  recommendations. The BSI is shown as LOW (below 40), MODERATE (40 to 70) or HIGH (70 and above).

**Results** (5-fold cross-validation grouped by trip, and 14 unseen test trips):

| Stress label | Model | Cross-validation R² | Test R² | Test MAE |
|---|---|---|---|---|
| Original | XGBoost | 0.708 | 0.737 | 4.58 |
| Physics-informed | XGBoost | **0.831** | **0.805** | **3.23** |

With the physics-informed label, the top factors become maximum speed, average speed and mean
acceleration, which are all under the driver's control.

**Notebooks:** `notebooks/module1_battery_stress/`

---

## Module 2: Charging-session anomaly detection

**Goal.** Detect charging sessions that are likely to be faulty, from the driver's point of view.

**Data.** 171,496 charging records grouped into 2,649 charging sessions (2,378 normal, 271 faulty),
with 57 features per session based on state of charge, voltage, current, power, temperature and
energy.

**Method.**
- An XGBoost classifier labels each session as Normal or Faulty.
- Only about 10% of sessions are faulty, so the faulty class is given more weight during training.
- Feature importance shows which session characteristics are behind each alert.
- A check with all energy features removed confirms that the model does not depend on a single
  type of signal.
- XGBoost is compared with three baselines: Isolation Forest (an unsupervised anomaly detector),
  and two deep learning models for sequences, LSTM and TCN (temporal convolutional network).

**Results** (530 test sessions, faulty-class metrics):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Isolation Forest | 95.28% | 0.939 | 0.574 | 0.713 | 0.929 |
| LSTM | 90.00% | 1.000 | 0.019 | 0.036 | 0.898 |
| TCN | 89.81% | 0.000 | 0.000 | 0.000 | 0.633 |
| XGBoost | **99.62%** | **1.000** | **0.963** | **0.981** | **0.999** |
| XGBoost without energy features | 99.62% | 1.000 | 0.963 | 0.981 | 0.9998 |

Only about 10% of sessions are faulty, so predicting every session as normal already gives
89.81% accuracy. This is why the faulty-class metrics matter most. TCN detects no faults and LSTM
only 1 of 54, despite accuracy around 90%. Isolation Forest misses 23 of 54. XGBoost finds 52 of
54 faulty sessions without a single false alarm.

**Notebooks:** `notebooks/module2_charging_anomaly/`

---

## Module 3: Battery health and degradation risk

**Goal.** Forecast the battery's State of Health (SOH) ahead of time, and warn when the battery
starts to wear out faster than its normal trend.

**Data.** Real-world charging-cycle records of NMC and LFP batteries. Sorting by battery and cycle
gives 18,065 forecasting examples with 282 features, split by time for each battery: the first
80% for training and the last 20% for testing.

**Method.**
- The target is the SOH **10 cycles ahead**, using only information available at the current
  cycle (past SOH values, SOH drops, degradation rates and rolling trends).
- Four models are compared. The direct autoregressive model predicts the 10-cycle value in one
  step, so errors do not build up as they do in step-by-step forecasting.
- **Accelerated-degradation detector:** the recent degradation rate (last 5 observations) is
  compared with the long-term rate (last 20). An alert is raised only if the speed-up holds in at
  least 2 of the last 3 observations. This gives a 0 to 100 risk score with levels LOW, MEDIUM,
  HIGH and CRITICAL.
- The forecast SOH is also shown as a health level: LOW (85% and above), MODERATE (80 to 85%),
  HIGH (70 to 80%) and CRITICAL (below 70%). 80% is the common end-of-life limit.

**Results** (SOH 10 cycles ahead, time-ordered test set):

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Random Forest | 0.00981 | 0.01819 | 0.9197 |
| XGBoost | 0.00964 | 0.01772 | 0.9238 |
| MLP | 0.02116 | 0.03937 | 0.6237 |
| Direct autoregressive | **0.00657** | **0.01248** | **0.9591** |

On a separate independent test of 1,152 forecasts, XGBoost reached R² = 0.9729.

**Notebooks:** `notebooks/module3_battery_health/`

---

## Module 4: Energy consumption and arrival SOC

**Goal.** Predict how much energy a trip will use, and how much charge will be left on arrival.

**Data.** The same 70 Munich trips used in Module 1. After quality checks, 67 trips are used for
arrival SOC prediction.

**Method.**
- **Energy consumption (kWh/km):** an Elastic Net model uses 23 features that were selected in
  all 50 cross-validation runs ("Stable-23"), which makes the model compact and reliable.
- **Arrival SOC:** a second Elastic Net predicts the battery level at the end of the trip from
  what is observed during the first 25%, 50% or 75% of the journey. All 85 features are computed
  only from data before that point, so the model never sees the future.

**Results.**

| Task | Model | R² | MAE |
|---|---|---|---|
| Energy consumption | Elastic Net, Stable-23 | **0.897** | 0.0077 kWh/km |
| Energy consumption | Elastic Net, 77 features | 0.854 | - |
| Arrival SOC, from first 25% of trip | Elastic Net | 0.932 | 2.54 percentage points |
| Arrival SOC, from first 50% of trip | Elastic Net | 0.976 | 1.48 percentage points |
| Arrival SOC, from first 75% of trip | Elastic Net | **0.984** | **1.22 percentage points** |
| Arrival SOC | Mean value only (baseline) | -0.09 | 11.6 percentage points |

The most important factors for energy use are battery current, cabin temperature, heating power
and the share of high-speed driving.

**Notebooks:** `notebooks/module4_energy_and_soc/`

---

## Federated learning

**Goal.** Check whether SparkEV can learn from many vehicles without collecting their raw data,
and whether a personal layer per vehicle helps.

**Method.**
- **FedAvg:** each client trains locally, and the server averages the full models.
- **FedPer:** the server averages only the shared base layers. Each client keeps its own personal
  layer, which never leaves the client.
- Tree models cannot be averaged, so the federated stress model (Module 1) is a small neural
  network. For the linear arrival SOC model (Module 4), the personal part is a per-client offset.
- The Munich data comes from one vehicle, so each trip acts as a separate client (51 clients for
  Module 1, 5 condition-based clients for Module 4).

**Results** (R²):

| Task | Centralized | FedAvg | FedPer |
|---|---|---|---|
| Module 1: battery stress | 0.737 | 0.541 | 0.452 (new vehicles) |
| Module 4: energy consumption | 0.837 | 0.740 | about 0.75 (client-local) |
| Module 4: arrival SOC (50% of trip) | 0.976 | 0.973 | 0.965 (client-local) |

- For arrival SOC, federated learning almost matches centralized training.
- For battery stress, most of the gap comes from the smaller model type: the same neural network
  trained centrally reaches 0.644, so FedAvg recovers 84% of what it can achieve.
- FedPer gives a lower error than FedAvg for 32 of 51 established clients, but new vehicles start
  with lower accuracy until their personal layer has learned.

**Notebooks:** `notebooks/module1_battery_stress/03_centralized_vs_federated_learning.ipynb`,
`notebooks/module4_energy_and_soc/02_energy_centralized_vs_federated.ipynb`,
`notebooks/module4_energy_and_soc/03_arrival_soc_forecasting.ipynb`
