# ESP32 Deployment

This document explains how SparkEV's trained XGBoost models run on an ESP32 microcontroller: how
they are converted to C, how the input data is prepared, how the firmware works, and how to
update it after retraining.

## Why conversion is needed

The ESP32 runs at 240 MHz with about 520 KB of RAM. The XGBoost library cannot run on a device of
this size. A trained XGBoost model, however, is only a set of decision trees. Each tree is a list
of simple rules such as "if feature 3 is less than 0.52, go left, otherwise go right", ending in
a number at each leaf. These rules can be stored as plain C arrays and evaluated with a short
loop. The ESP32 needs no machine learning library at all.

## Pipeline

```
 Computer                                                            ESP32
 --------                                                            -----
 models/*.json, *.pkl  --convert_model*_to_esp32.py-->  model*_esp32_generated.h  --+
                                                                                   +--> SparkEV_Integrated.ino
 data/raw/*.xlsx  --export_model*_driver*.py-->  vectors/*.txt                     |
                  --convert_model*_driver_vectors.py-->  model*_driver_vectors.h --+
```

1. **Model conversion.** Each trained model is read and its trees are written into a C header file
   (`model1_esp32_generated.h`, `model2_esp32_generated.h`, `model3_esp32_generated.h`).
2. **Input preparation.** For each demonstration driver, a row of data is taken from the dataset
   and prepared exactly as during training (feature engineering, encoding and scaling). The
   result is a list of numbers, the input vector, saved to `esp32/tools/vectors/`. The model's
   prediction for this row is also computed on the computer and saved as a reference.
3. **Vector headers.** The input vectors are written into C headers
   (`model2_driver_vectors.h`, `model3_driver_vectors.h`). The Module 1 vectors are stored
   directly in the sketch.
4. **Firmware.** The sketch includes all headers, runs the three models, measures the time with
   the on-chip microsecond timer, shows the results on the OLED display and sends them to the
   server.

## The models on the device

| Model | Header | Trees | Input features | Output |
|---|---|---|---|---|
| Module 1: battery stress | `model1_esp32_generated.h` | 300 | 24 | Battery Stress Index (0 to 100) |
| Module 2: charging anomaly | `model2_esp32_generated.h` | 300 | 57 | Probability that the session is faulty |
| Module 3: battery health | `model3_esp32_generated.h` | 300 (100 per output) | 199 | SOH, capacity, corrected capacity |

All model data is declared `const`, so the compiler places it in flash memory instead of RAM.
The complete program, with all three models, uses about 2.0 MB of flash (63% of the 3 MB
available with the **Huge APP** partition scheme) and 50 KB of RAM for global variables (15%).
Because the program is larger than the default 1.2 MB application space, the **Huge APP**
partition scheme must be selected when uploading.

### How a prediction is computed

For each tree, the code starts at the first node and repeats:

- If the node is a leaf, return its value.
- Otherwise, compare the input feature with the node's threshold. If the feature is smaller, go to
  the left child, otherwise go to the right child.

The prediction is the model's base score plus the sum of the values returned by all trees. For
Module 2, this sum is passed through a sigmoid function to give a probability. Module 3 adds up
each output's 100 trees separately.

Module 3 needs two extra rules that XGBoost uses:

- **Missing values.** Each node stores which child to follow when the input value is missing
  (`NAN`), matching XGBoost's behaviour.
- **Categorical features.** Some Module 3 features are categories rather than numbers. Nodes that
  split on them are marked and follow XGBoost's category rule.

## Firmware overview

`esp32/firmware/SparkEV_Integrated/SparkEV_Integrated.ino`

| Part | What it does |
|---|---|
| Configuration | Includes `secrets.h` (Wi-Fi and server URL) and selects the driver with `TEST_DRIVER` |
| `run_model1()`, `run_model2()`, `run_model3()` | Run each model on the stored input vector and measure the time with `micros()` |
| Display functions | Show each model's result and the total time on the SH1106 OLED |
| `setup()` | Starts the display, connects to Wi-Fi, runs the three models once and sends the results as JSON to `POST /esp32/result` |
| `loop()` | Shows the results of Modules 1, 2 and 3 one after another, then the timing summary, and sends the results to the server again |

See the [installation guide](installation.md#7-set-up-the-esp32) for wiring and upload steps, and
the [API reference](api.md#post-esp32result) for the JSON format.

## Measured performance

| Model | Sample output | Prediction time |
|---|---|---|
| Module 1: battery stress | Stress score 62.39, MODERATE | 3.35 ms |
| Module 2: charging anomaly | Probability of fault 1.000, Faulty | 1.92 ms |
| Module 3: battery health | SOH 91.76%, risk LOW | 3.38 ms |
| **Total, run one after another** | | **8.65 ms** |

Prediction time is measured on the chip and does not include communication. Over 20 transmissions,
the server took 17.5 ms on average to process a result, and the network round trip over a mobile
hotspot ranged from 52 to 589 ms (median 188 ms).

## Updating the models after retraining

Run these commands from the project folder. All output goes directly into
`esp32/firmware/SparkEV_Integrated/`.

**1. Convert the models**

```bash
python esp32/tools/convert_model1_to_esp32.py     # Module 1, reads models/model1_battery_stress_xgb.json
python esp32/tools/convert_model2_to_esp32.py     # Module 2, reads models/model2_charging_fault_xgb.pkl
python esp32/tools/convert_model3_to_esp32.py     # Module 3, reads models/model3_battery_degradation_xgb.pkl
```

**2. Check the Module 3 conversion**

```bash
python esp32/tools/verify_model3_converter.py
```

This walks through every tree of the model in Python using the same rules as the C code
(including missing values and categories) and checks that the result matches XGBoost's own
prediction.

**3. Regenerate the input vectors for drivers 17 and 24**

```bash
python esp32/tools/export_model2_driver17.py
python esp32/tools/export_model2_driver24.py
python esp32/tools/export_model3_driver17.py
python esp32/tools/export_model3_driver24.py
python esp32/tools/convert_model2_driver_vectors.py
python esp32/tools/convert_model3_driver_vectors.py
```

Each `export_*` script also saves the reference prediction computed on the computer in
`esp32/tools/vectors/`. Compare it with the value shown on the ESP32 to confirm that the device
gives the same result.

**4. Upload the sketch to the ESP32 again.**

## Limitations

- The ESP32 uses prepared input vectors for two demonstration drivers instead of live vehicle
  signals, and feature preparation runs on the computer.
- Module 4 uses a linear model, which is much lighter than the tree models, and is not yet part
  of the device test.
- Only predictions run on the device. Training the FedPer personal layer on the device has not
  been tested.

These are addressed in Phase 6 of the project roadmap.
