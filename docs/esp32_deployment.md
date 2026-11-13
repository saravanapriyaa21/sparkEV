# ESP32 Deployment

This document explains how SparkEV's models run, and are trained, on an ESP32 microcontroller.
It covers how the models are converted to C, how the input data is prepared, how the firmware
works, the on-device federated learning, the measured performance, and how to update the
firmware after retraining.

## Why conversion is needed

The ESP32 runs at 240 MHz and has 320 KB of RAM. Machine learning libraries such as XGBoost
or PyTorch cannot run on a device of this size. The trained models, however, are only numbers
and simple rules:

- an **XGBoost** model is a set of decision trees ("if feature 3 is less than 0.52, go left");
- a **neural network** (MLP) is a few weight matrices, multiplied and passed through ReLU;
- an **Elastic Net** is one weighted sum.

All of these can be stored as plain C arrays and computed with short loops, so the ESP32 needs
no machine learning library at all.

## Pipeline

```
 Computer                                                              ESP32
 --------                                                              -----
 models/*  --convert_model*_to_esp32.py-->  model*_esp32_generated.h  ---+
                                                                         |
 datasets  --export_model*_*.py-->  vectors/*.json                       +--> SparkEV_Integrated.ino
           --convert_model*_to_esp32.py-->  model*_driver_vectors.h  ----+
                                                                         |
 notebooks --export_fl_setup.py-->  models/fl/*  (also used by server)   |
           --convert_fl_to_esp32.py-->  fl_esp32_generated.h  -----------+
```

1. **Model conversion.** The trained models are written into C header files.
2. **Input preparation.** For each demonstration driver (17 and 24), the input of each module is
   prepared exactly as during training (feature engineering, encoding and scaling) and stored as
   a list of numbers, the input vector. The prediction made on the computer is saved as well,
   as a reference.
3. **Federated learning set-up.** `export_fl_setup.py` re-runs the federated training of each
   module's notebook with the same code and seeds, and saves the global models, each
   demonstration car's local training data and the simulated fleet for the server.
4. **Firmware.** The sketch includes all headers, runs the four prediction modules, measures the
   time with the on-chip microsecond timer, shows the results on the OLED display, sends them to
   the server and takes part in federated learning.

## The models on the device

| Module | Model | Header | Input | Output |
|---|---|---|---|---|
| 1: battery stress | XGBoost, 300 trees | `model1_esp32_generated.h` | 24 features | Battery Stress Index (0 to 100) |
| 2: charging recommendation | 2 federated MLPs (68-64-32-1, 44-64-32-3) | `model2_esp32_generated.h` | latest charging session | charge need (85th percentile, %), urgency class |
| 3: battery health | federated MLP (357-64-32-1) | `fl_esp32_generated.h` | 357 features of the newest cycle | SOH 10 cycles ahead |
| 4: energy and arrival SOC | 2 Elastic Net models | `model4_esp32_generated.h` | 23 and 85 trip features | energy use (kWh/km), arrival SOC (%) |

The Module 2 urgency model uses the driver's own FedPer personal head (`model2_driver_vectors.h`).
The Module 3 model predicts the change of SOH over the next 10 cycles; the forecast is the SOH
now plus that change.

All model data is declared `const`, so the compiler places it in flash memory instead of RAM.
The complete program uses 1.47 MB of flash (47% of the 3 MB available with the **Huge APP**
partition scheme) and 51 KB of RAM for global variables (16%). Because the program is larger
than the default 1.2 MB application space, the **Huge APP** partition scheme must be selected
when uploading.

### How a prediction is computed

- **XGBoost (Module 1):** for each tree, start at the first node; if it is a leaf, return its
  value, otherwise compare the input feature with the node's threshold and go to the left or
  right child. The prediction is the base score plus the sum of all trees.
- **MLP (Modules 2 and 3):** the input is scaled as in training, then each layer computes
  `y = x W + b`, with ReLU between layers. Module 2 adds the user's own average need back to the
  charge-need output (the network predicts the difference from it) and uses a softmax for the
  urgency classes.
- **Elastic Net (Module 4):** a weighted sum of the scaled features plus the intercept.

## On-device federated learning

In a real car, prediction and training happen at different times. While the car is driving it
only predicts with the models it has. While it is parked or charging, it can improve those
models with its own recent data and share what it learned, without sharing the data itself.
The ESP32 does exactly this: after every second display cycle it runs one **federated learning
session**, one round for each federated model.

```
 ESP32 (one car)                                   FastAPI server (backend/federated.py)
 ---------------                                   -------------------------------------
 1. GET  /fl/global/<task>  ---------------------> current global model
 2. train on the car's own data (fl_runtime.h)
 3. POST /fl/update/<task>  ---------------------> 10 simulated cars train the same way,
    (weights only)                                 all updates are aggregated
    <--------------------------------------------- new global model
 4. install it; predictions now use it
 5. POST /fl/report         ---------------------> shown in the dashboard's Technical panel
```

| Task | Model | Strategy | Car's local data | Weights exchanged |
|---|---|---|---|---|
| `m1` | Module 1 stress MLP (24-32-16-1) | FedPer | 86 windows of a held-out trip | shared base, 5.2 KB |
| `m2c` | Module 2 charge need MLP | FedAvg, FedAdam server | 27 charging sessions of the demo user | full network, 25.5 KB |
| `m2u` | Module 2 urgency MLP | FedPer, FedAdam server | 31 charging sessions | shared base, 19.4 KB |
| `m3` | Module 3 SOH MLP | FedAvg | 62 cycles of an independent test battery | layers 2 and 3, 8.3 KB |
| `m4` | Module 4 energy Elastic Net | FedAvg | 11 trips | coefficients and intercept, 96 B |

(Local data for driver 17; driver 24 has 73, 194, 197, 172 and 11 samples.)

- **Same procedure as the notebooks.** Local training uses each notebook's settings: Adam with
  the notebook's learning rate, epochs and batch size (state reset every round), the same loss
  (mean squared error, pinball loss at the 85th percentile, or class-weighted cross-entropy),
  and proximal gradient descent for the Elastic Net. `backend/federated.py` contains the
  reference implementation. It gives the same result as PyTorch (to 1 × 10⁻⁷) and as the
  Module 4 notebook (to 4 × 10⁻⁹).
- **FedPer heads stay on the car.** For `m1` and `m2u` only the shared base is uploaded; the car
  keeps training its own head.
- **Module 1 predictions stay with XGBoost**, which is more accurate for this task (held-out
  R² 0.737 against 0.541 for FedAvg). Gradient-boosted trees cannot be trained by
  averaging weights, so the federated stress model is the MLP of the federated notebook.
- **Module 3 first layer is frozen on the car.** Training its 357 × 64 weights with Adam would
  need about 366 KB of RAM, more than the ESP32 has. The car trains and exchanges layers 2 and 3.
- **Memory.** The weights being trained are kept in RAM, allocated the first time a model is
  trained. Gradients and Adam moments are allocated only during training (at most 78 KB, for the
  charge-need model) and freed afterwards. The lowest free RAM during a session was 50 KB.
- **The rest of the fleet is simulated.** There is one board, so the server simulates the other
  cars with the other clients of each module's notebook (10 per round; Module 4 has 3). Without
  Wi-Fi, the car still trains locally and keeps its updated model.

## Firmware overview

`esp32/firmware/SparkEV_Integrated/SparkEV_Integrated.ino`

| Part | What it does |
|---|---|
| Configuration | Includes `secrets.h` (Wi-Fi and server URL), selects the driver with `TEST_DRIVER` and the benchmark with `BENCHMARK_MODE` |
| `run_model1()` to `run_model4()` | Run each module on the stored input and measure the time with `micros()` |
| Display functions | Show each module's result, the total time and the federated learning summary on the SH1106 OLED |
| `setup()` | Starts the display, connects to Wi-Fi, runs the four modules once and sends the results as JSON to `POST /esp32/result` |
| `loop()` | Shows the results of Modules 1 to 4 one after another, then the timing summary, and sends the results to the server; every `FL_EVERY_N_CYCLES` (2) cycles it runs a federated learning session |
| `run_fl_session()` | One round for each federated model, then new predictions with the updated models and a report to `POST /fl/report` |
| `fl_runtime.h` | On-device training: MLP with mini-batch Adam, Elastic Net with proximal gradient descent |
| `fl_tasks.h` | The five federated tasks: weight blocks, local data, frozen layer, switching predictions to the updated weights |

See the [installation guide](installation.md#7-set-up-the-esp32) for wiring and upload steps, and
the [API reference](api.md) for the JSON format and the federated learning endpoints.

## Measured performance

Measured on an ESP32 (ESP32-D0WD-V3, 240 MHz, 4 MB flash) with `BENCHMARK_MODE 1`, driver 17, over a mobile hotspot.

**Prediction (100 runs each)**

| Module | Sample output | Prediction time |
|---|---|---|
| 1: battery stress | Stress score 62.38, MODERATE | 3.37 ms |
| 2: charging recommendation | Charge need 69.4%, target 84%, Time-critical | 2.18 ms |
| 3: battery health | SOH 88.30% in 10 cycles (87.56% now), risk LOW | 3.93 ms |
| 4: energy and arrival SOC | 0.204 kWh/km, 55.4% on arrival | 0.12 ms |
| **All four, one after another** | | **9.64 ms** |

The outputs on the ESP32 match the computer: the largest difference is 8 × 10⁻⁶ (Module 2
charge need, in %).

**On-device training (20 runs each, from the deployed model)**

| Model | Samples | Training time | RAM for training |
|---|---|---|---|
| Module 1 stress MLP | 86 | 188 ms (10 epochs, full batch) | 16 KB |
| Module 2 charge need MLP | 27 | 58 ms (2 epochs) | 77 KB |
| Module 2 urgency MLP | 31 | 53 ms (2 epochs) | 59 KB |
| Module 3 SOH MLP | 62 | 45 ms (2 epochs), plus 220 ms for the frozen first layer | 41 KB |
| Module 4 energy Elastic Net | 11 | 0.3 ms (10 epochs) | 0.1 KB |

After training, the local losses on the ESP32 match the Python reference to within 6 × 10⁻⁶
(relative).

**Federated rounds over Wi-Fi (3 sessions, 15 model rounds, all completed)**

| | Median |
|---|---|
| Download of the global model | 57 ms |
| Upload, aggregation with 10 simulated cars, and new global model back | 137 ms |
| Server time for one round | 4 to 34 ms |
| Whole session, all five models | about 2.1 s |

**Communication of predictions:** over 100 sends, all results arrived. The round trip took
248 ms on average (median 228 ms, 95th percentile 587 ms), and the server needed 49 ms on
average to rank the stations and generate the recommendations. One display cycle lasts
15.4 s, and the prediction work takes less than 0.1% of it.

## Updating the models after retraining

Run these commands from the project folder. All output goes directly into
`esp32/firmware/SparkEV_Integrated/`.

**1. Module 1 (XGBoost)**

```bash
python esp32/tools/convert_model1_to_esp32.py     # reads models/model1_battery_stress_xgb.json
```

**2. Module 2 (federated MLPs) and its demo users**

```bash
python esp32/tools/export_model2_demo_users.py    # demo users' inputs and reference predictions
python esp32/tools/convert_model2_to_esp32.py     # reads models/model2_charge_level_fedavg.*, model2_urgency_fedper.*
```

**3. Module 4 (Elastic Net) and its demo trips**

```bash
python esp32/tools/export_model4_models.py --trips "/path/to/TUM trip data"
python esp32/tools/convert_model4_to_esp32.py
```

**4. Federated learning and Module 3**

```bash
python esp32/tools/export_fl_setup.py --module all \
    --trips "/path/to/TUM trip data" --m3data "/path/to/folder with data_train and data_test"
python esp32/tools/convert_fl_to_esp32.py
```

`export_fl_setup.py` needs PyTorch and XGBoost (see `requirements-notebooks.txt`). It writes
`models/fl/`, which the server also uses, and checks that each notebook's federated result is
reproduced.

**5. Upload the sketch to the ESP32 again** (Huge APP partition scheme).

**Benchmark.** Set `BENCHMARK_MODE 1`, upload, save the Serial Monitor output (115200 baud) and
the server output, then run:

```bash
python esp32/tools/parse_benchmark_log.py --serial serial.txt --server-log server.log --out results
```

It writes the summary, the prediction parity with the computer, the on-device training times and
the federated rounds.

## Limitations

- The ESP32 uses prepared input vectors and local data for two demonstration drivers instead of
  live vehicle signals, and feature preparation runs on the computer.
- The rest of the fleet is simulated on the server; only one car is real. There is no secure
  aggregation or differential privacy.
- The server keeps the global models in memory; they restart from the deployed models when the
  server restarts.
- The Module 3 first layer is not trained on the device because of the RAM limit.

These are addressed in Phase 6 of the project roadmap.
