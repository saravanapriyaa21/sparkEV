# Research Summary

This document describes the method and results of each SparkEV module and of the federated
learning experiments. The numbers come from the notebooks in `notebooks/`. For the on-device
results, see [ESP32 deployment](esp32_deployment.md).

## Contents

1. [Module 1: Driving behaviour and battery stress](#module-1-driving-behaviour-and-battery-stress)
2. [Module 2: Predictive charging recommendation](#module-2-predictive-charging-recommendation)
3. [Module 3: Battery health and degradation risk](#module-3-battery-health-and-degradation-risk)
4. [Module 4: Energy consumption and arrival SOC](#module-4-energy-consumption-and-arrival-soc)
5. [Module 5: What-If Simulator](#module-5-what-if-simulator)
6. [Explainable recommendations (SHAP)](#explainable-recommendations-shap)
7. [Federated learning](#federated-learning)

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

## Module 2: Predictive charging recommendation

**Goal.** Recommend a suitable charging decision instead of simply the nearest charger: which
station to use, and how much to charge, based on what the driver will need, how urgent the
session is and how busy the stations are.

**Data.** 34,358 real residential charging sessions of 261 users at 12 apartment-building sites in
Norway (Sørensen et al., 2024). Each session has plug-in and plug-out times, energy charged, and
estimated battery capacity, charging power and state of charge.

**Method.** Three models, all using only information known when the car is plugged in (user
history uses earlier sessions only). The data is split by time: the first 70% of sessions for
training, the next 15% for validation and tuning, and the last 15% for testing.

- **Model A, charge level:** predicts how much battery (%) the driver will use before the next
  charge. An 85th-percentile (quantile) model gives a safe estimate, and the recommended charge
  level is this estimate plus a 15% reserve.
- **Model B, urgency:** classifies a session as time-critical, moderate or flexible from how long
  the car usually stays plugged in compared with the time it needs to charge.
- **Model C, site congestion:** predicts how busy each charging site will be in the next hour,
  converted into a low, medium or high waiting risk.
- **Decision layer:** ranks the candidate stations by travel time, expected waiting time and
  charging time, weighted by the predicted urgency, and recommends a station and a charge level.

All three models are gradient-boosted trees (HistGradientBoosting), tuned on the validation set.

**Results** (test set, last 15% of sessions):

| Task | Model | Simple baseline |
|---|---|---|
| Charge needed before next charge | MAE 14.2 points, R² 0.311 | user's average: MAE 14.6, R² 0.263 |
| Recommended charge level | lasts until the next charge with at least 5% left in **94.2%** of sessions | always charge to 95% |
| Urgency (3 classes) | accuracy **70.5%**, macro F1 0.645 | |
| Time-critical vs. not | accuracy 84.8%, F1 0.576 | |
| Site congestion, next hour | R² 0.878 | same as last hour: R² 0.874 |
| Waiting risk (3 classes) | accuracy 91.7% | same as last hour: 91.5% |

- The recommended charge level is on average **69.8%**, 25 percentage points less than always
  charging to 95%, and a full charge is unnecessary in 86.7% of sessions.
- Predicting exactly how much a driver will need is hard (R² 0.31): it depends on future trips
  the data does not describe. This is why the recommendation uses a safe upper estimate rather
  than the average.
- Site congestion one hour ahead is only slightly better than assuming it stays the same as the
  last hour. Real station waiting-time data would make this part stronger.

**Explainability.** SHAP is applied to the federated models (see
[Federated learning](#federated-learning)). The user's own charging history drives the charge
level, while the time of day and the current session drive the urgency.

**Notebooks:** `notebooks/module2_charging_recommendation/`

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

The autoregressive model is scored on the 2,570 test cycles that have 20 earlier SOH values; the
other models on all 3,652.

On a separate independent test of 6 unseen batteries (1,152 forecasts), XGBoost reached
R² = 0.9729. On the 1,038 of these forecasts that the autoregressive model can make, XGBoost
reached R² = 0.975 and the autoregressive model R² = 0.967, so XGBoost generalizes better to new
batteries.

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

## Module 5: What-If Simulator

**Goal.** Let drivers explore other decisions than the recommended one and see the predicted
trade-offs between convenience, efficiency and battery health, instead of only receiving one
optimal answer.

**Method.** The simulator re-runs the project's own models with the driver's chosen settings and
compares the result with the recommended option. Only the settings the driver changes are
altered; all other inputs stay at the driver's current values.

| Setting | Outcome | How it is predicted |
|---|---|---|
| Station | travel, waiting, charging and total time | Module 2: predicted site congestion and the decision-layer formulas |
| Charge level, charger power | energy charged, range added | Module 2 (charging efficiency 0.88) and the Module 4 physics model |
| Charge level, charger power | battery wear of the charge, capacity lost per 100 such charges | battery ageing model fitted on the KIT cell ageing data (below): wear per full cycle by charge level reached and charging rate, at 25 °C |
| Charge level | ageing while parked at that level | storage tests of the same dataset, relative to 50% |
| Driving style, speed | driving stress | Module 1 XGBoost on the scaled driving features |
| Driving style, speed, heating and air conditioning | energy use and range | Module 4 physics-based vehicle model (below) |

**Energy use: a physics-based vehicle model.** The Module 4 Elastic Net cannot be used for
what-if questions: its acceleration, throttle and speed inputs are correlated with weather and
route (high-speed trips are also the warm summer trips), so its coefficients have signs that are
not physical, and smoother driving would wrongly raise the predicted energy use. The simulator
therefore uses a grey-box vehicle model built for this purpose
(`notebooks/module4_energy_and_soc/04_physics_energy_model.ipynb`): the standard driving-resistance
equation (inertia, rolling resistance, climbing, air drag), with drivetrain and regeneration
efficiency, an auxiliary load and the heating / air-conditioning power, fitted to the measured
second-by-second battery power of the Munich trips.

- Fitted constants: effective mass 1.55 t, rolling resistance 0.011, drag area 0.54 m²,
  drivetrain efficiency 0.83, regeneration efficiency 0.90, auxiliary load 0.66 kW.
- Validation on the 14 unseen test trips: second-by-second battery power R² 0.88; whole-trip
  consumption on winter trips R² 0.90 (error 0.006 kWh/km). Summer trips are predicted poorly
  because their logged air-conditioning power does not match what the battery supplies.
- The what-if changes the trip's speed profile (gentler or harsher accelerating and braking
  around the traffic flow, lower or higher speed) or the heating / AC power and recomputes the
  energy. Median effects over all trips: smoother driving -0.0045 kWh/km, 10% slower -0.0062,
  10% faster +0.0085, half the heating / AC -0.015.
- The Elastic Net stays the model that predicts a trip's consumption (test R² 0.837); the
  physics model is used only to estimate the effect of a change.

**Battery wear: measured cell ageing data.** Module 3's data cannot show the effect of the charge
level (almost every charge in it ends at 95-100%), and the Module 1 capacity-fade model has no
state-of-charge term. The simulator therefore uses an ageing model fitted on the open KIT dataset
(Luh and Blank, 2024; 228 NMC/C-SiO cells, CC BY 4.0) in
`notebooks/module1_battery_stress/04_charge_level_ageing_model.ipynb`:

- Cycling to 100% instead of 90% wears the battery 1.8 times faster per cycle; emptying it
  completely (0-100%) is about 5 to 8 times worse than 10-90%.
- Parking at 90-100% ages it about 1.2 to 1.3 times faster than at 50%.
- Charging speed has no measurable effect at 25 °C, but in the cold (10 °C or below) each doubling
  of the charging rate multiplies the wear by 1.42.
- Leaving each test condition out in turn, the model predicts it with R² 0.79 (log scale) and a
  median error of 15%.
- Charge levels below 90% were not tested, so no extra benefit is assumed below 90%; the session
  wear still falls because less energy is charged.

**Example** (driver 17, SparkEV recommends 84% at Station 2; waiting times depend on the hour
of the day the congestion forecast is made for):

| Choice | Effect |
|---|---|
| Station 1 instead of Station 2 | about half an hour longer in total (slower 7 kW charger, higher waiting risk), same battery wear |
| 100% instead of 84% | about half an hour longer, 20 km more range, 2.8 times the battery wear (3.6 instead of 1.3 capacity points per 100 such charges) |
| 80% instead of 84% | 7 min shorter, 5 km less range, 15% less battery wear |
| Smoother and slightly slower driving (style 0.7, speed 0.9) | driving stress 62.6 to 40.5, energy use 0.198 to 0.190 kWh/km |

**Limitation.** These are model estimates of what changes when one input changes, not measured
outcomes. The models were trained to predict, not to establish cause and effect, so the
simulator is meant for comparing options, not for exact values.

## Explainable recommendations (SHAP)

**Goal.** Tell the driver *why* a prediction was made and what they can change, using the
prediction for their own data rather than fixed rules.

**Method.** For every new result, the server (`backend/explain.py`) computes SHAP values for the
latest prediction of each module. A SHAP value is how much one factor moved the prediction away
from the average case; the values add up exactly to the prediction.

| Module | Prediction explained | Model | SHAP method |
|---|---|---|---|
| 1 | Battery Stress Index | XGBoost | TreeSHAP (exact) |
| 2 | Charge needed before the next charge | Federated MLP (FedAvg) | Kernel SHAP, 40 background sessions, 500 samples (same settings as the Module 2 notebook) |
| 3 | SOH change over the next 10 cycles | Federated MLP (FedAvg) | Kernel SHAP over 14 factor groups (for example charging temperature, charging current, state-of-charge window, fast or slow charging), 40 background cycles |
| 4 | Energy use of the trip | Elastic Net | Linear SHAP (exact) |

For Modules 2 to 4 the server explains its current global federated model, which is the model
the car installs after each federated round. Related inputs are reported as one factor (for
example the one-hot route columns, or the 357 Module 3 inputs grouped into readable factors).

**From explanation to advice.** A factor becomes a recommendation only if both are true:

1. the driver can change it (for example acceleration, braking, speed, heating, charging
   temperature, charging current, charge window), and
2. SHAP shows that, for this driver now, it makes the prediction worse (raises stress or energy
   use, or lowers the SOH forecast).

For Module 4, whose model is correlational, advice is also given only when the direction agrees
with physics (for example, more heating power than average). Factors the driver cannot change,
such as outside temperature or battery age, are shown as reasons, not as advice. The charging
station choice comes from the Module 2 decision layer, and its explanation is the time breakdown
(travel, expected waiting and charging time) of every station. The dashboard lists the insights
as "Things to try" (SHAP found a factor to change) or "Good to know" (information), in plain
language and without model names, and shows the SHAP contributions of each prediction as bars.

**Example (driver 17).** Stress 62.6 against 11.8 for the average case. The largest controllable
contributions are acceleration (+21.8 points in total) and speed (+19.1), so the advice is
"accelerate more gently and evenly" and "lower your speed". The predicted charge need is 69%
against 59% on average, mostly because of the driver's own usual need (+25.1 points). For the
battery, the charge window (how full or empty the battery is charged) lowers the SOH forecast by 0.05 points, which leads to the
advice to keep the charge between about 20 and 80%. Heating adds 5 Wh/km to the trip's energy
use.

**Cost.** About 30 to 70 ms on the server for all four modules. Results are cached until a
federated round changes a model.

---

## Federated learning

**Goal.** Check whether SparkEV can learn from many vehicles without collecting their raw data,
and whether a personal layer per vehicle helps.

**Method.**
- **FedAvg:** each client trains locally, and the server averages the full models.
- **FedPer:** the server averages only the shared base layers. Each client keeps its own personal
  layer, which never leaves the client.
- **FedProx** (Modules 2 and 3): FedAvg with an extra term that keeps each client's model close
  to the global model.
- **FedAdam** (Module 2): the server uses the averaged client update as a step for an Adam
  optimiser, which trains faster and more stably when users' data differ a lot.
- Tree models cannot be averaged, so the federated stress model (Module 1) and the federated SOH
  model (Module 3) and the charging-recommendation models (Module 2) are small neural networks. For the linear arrival SOC model (Module 4), the
  personal part is a per-client offset.
- The Munich data comes from one vehicle, so each trip acts as a separate client (51 clients for
  Module 1, 5 condition-based clients for Module 4). In Module 3 each battery is a client
  (105 clients with different chemistries), and in Module 2 each charging user is a client
  (188 to 209 clients). Both are real non-identical client settings.

**Results** (R²):

| Task | Centralized | FedAvg | FedPer |
|---|---|---|---|
| Module 1: battery stress | 0.737 | 0.541 | 0.452 (new vehicles) |
| Module 2: charge needed before next charge | 0.249 (same network) | 0.236 | 0.174 |
| Module 3: SOH 10 cycles ahead | 0.938 (same network) | 0.940 | 0.938 |
| Module 4: energy consumption | 0.837 | 0.740 | about 0.75 (client-local) |
| Module 4: arrival SOC (50% of trip) | 0.976 | 0.973 | 0.965 (client-local) |

- For SOH forecasting, FedAvg matches the same network trained centrally (0.940 against 0.938),
  and is much better than each battery training alone (0.923). FedProx gives 0.939. FedPer brings
  no extra gain on this data, and centralized XGBoost stays best on unseen batteries.
- For charging recommendation, the first federated setup trained for too few rounds and scored
  below simple baselines (charge need R² -0.20). Training longer, a FedAdam server optimiser and
  predicting the difference from the user's own average need fixed this: FedAvg now reaches
  R² 0.236 against 0.249 for the same network trained centrally, and 0.247 for new users. For
  urgency, FedPer and FedAvg reach macro-F1 0.633 to 0.636 against 0.660 centrally. Centralized
  gradient-boosted trees stay slightly better (R² 0.281, urgency F1 0.681).
- For arrival SOC, federated learning almost matches centralized training.
- For battery stress, most of the gap comes from the smaller model type: the same neural network
  trained centrally reaches 0.644, so FedAvg recovers 84% of what it can achieve.
- FedPer gives a lower error than FedAvg for 32 of 51 established clients, but new vehicles start
  with lower accuracy until their personal layer has learned.

**Notebooks:** `notebooks/module1_battery_stress/03_centralized_vs_federated_learning.ipynb`,
`notebooks/module2_charging_recommendation/02_charging_recommendation_federated.ipynb`,
`notebooks/module3_battery_health/02_soh_centralized_vs_federated.ipynb`,
`notebooks/module4_energy_and_soc/02_energy_centralized_vs_federated.ipynb`,
`notebooks/module4_energy_and_soc/03_arrival_soc_forecasting.ipynb`

### Federated learning on the device

The experiments above simulate every client on one computer. To show that the method also works
in a real car, the ESP32 runs the client side itself. While the car is parked or charging, it
downloads the global model of each federated model, trains it on its own data with the same
procedure as the notebook, uploads only the weights, and installs the new global model that
the server returns. The server aggregates the car's update with simulated cars (the other
clients of each notebook, 10 per round), using FedAdam for Module 2 as in its notebook. FedPer
heads stay on the car.

| Model | Strategy | Training on the ESP32 | Weights sent per round |
|---|---|---|---|
| Module 1 stress MLP | FedPer | 188 ms | 5.2 KB |
| Module 2 charge need MLP | FedAvg, FedAdam server | 58 ms | 25.5 KB |
| Module 2 urgency MLP | FedPer, FedAdam server | 53 ms | 19.4 KB |
| Module 3 SOH MLP (first layer frozen) | FedAvg | 45 ms | 8.3 KB |
| Module 4 energy Elastic Net | FedAvg | 0.3 ms | 96 B |

- The training on the ESP32 matches the Python reference to within 6 × 10⁻⁶ (relative), and
  the reference matches PyTorch and the Module 4 notebook.
- A complete session for all five models, including Wi-Fi transfer and aggregation, takes about
  2 seconds; the training needs at most 78 KB of RAM.
- Module 3's first layer (357 × 64 weights) is frozen on the device, because training it would
  need more RAM than the ESP32 has.
- Module 3 now uses the FedAvg network for its predictions on the device and the server, since
  it is more accurate on the chronological test set than XGBoost (R² 0.940 against 0.923) and
  needs no raw battery data to be shared.
- The losses after a round are measured on the car's own data. With the notebooks' local
  learning rates, the local loss falls in every round for all models except one case: for
  driver 24's charge-need model, which is already well fitted to that user, the first Adam
  steps raise the local loss (0.0103 to 0.0143).

Code: `backend/federated.py` (server and reference), `esp32/firmware/SparkEV_Integrated/fl_runtime.h`
and `fl_tasks.h` (device), `esp32/tools/export_fl_setup.py` (set-up from the notebooks).
Measurements: [ESP32 deployment](esp32_deployment.md#on-device-federated-learning).
