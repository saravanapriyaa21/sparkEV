import os
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
VECTORS_DIR = os.path.join(PROJECT_ROOT, "esp32", "tools", "vectors")
FIRMWARE_DIR = os.path.join(
    PROJECT_ROOT, "esp32", "firmware", "SparkEV_Integrated"
)

DATA_PATH = os.path.join(
    DATA_DIR, "processed_data_longer_than_30.xlsx"
)

TRANSACTION_ID = (
    "44010000100000052203030401089310"
)

OUTPUT_PATH = os.path.join(
    VECTORS_DIR, "model2_driver17_vector.txt"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("SparkEV Model 2 - Test Feature Export")
print("=" * 70)

print("\nLoading:")
print(DATA_PATH)

df = pd.read_excel(DATA_PATH)

df["begin_time"] = pd.to_datetime(
    df["begin_time"]
)

df["end_time"] = pd.to_datetime(
    df["end_time"]
)

df = df.sort_values(
    ["transaction_id", "end_time"]
).reset_index(drop=True)


# ============================================================
# SELECT REAL TRANSACTION
# ============================================================

transaction_series = df["transaction_id"].astype(str)

session = df[
    transaction_series == TRANSACTION_ID
].copy()

if len(session) == 0:
    raise ValueError(
        "Transaction not found: "
        + TRANSACTION_ID
    )

print(
    "\nTransaction found:",
    TRANSACTION_ID
)

print(
    "Rows:",
    len(session)
)


# ============================================================
# HELPERS
# ============================================================

def first_value(x):
    return x.iloc[0]


def last_value(x):
    return x.iloc[-1]


# ============================================================
# SESSION FEATURES
# ============================================================

row = {}

row["soc_start"] = first_value(
    session["current_soc"]
)

row["soc_end"] = last_value(
    session["current_soc"]
)

row["soc_min"] = session[
    "current_soc"
].min()

row["soc_max"] = session[
    "current_soc"
].max()

row["soc_mean"] = session[
    "current_soc"
].mean()

row["soc_std"] = session[
    "current_soc"
].std()

row["voltage_start"] = first_value(
    session["chargingv"]
)

row["voltage_end"] = last_value(
    session["chargingv"]
)

row["voltage_min"] = session[
    "chargingv"
].min()

row["voltage_max"] = session[
    "chargingv"
].max()

row["voltage_mean"] = session[
    "chargingv"
].mean()

row["voltage_std"] = session[
    "chargingv"
].std()

row["current_start"] = first_value(
    session["charginga"]
)

row["current_end"] = last_value(
    session["charginga"]
)

row["current_min"] = session[
    "charginga"
].min()

row["current_max"] = session[
    "charginga"
].max()

row["current_mean"] = session[
    "charginga"
].mean()

row["current_std"] = session[
    "charginga"
].std()

row["power_start"] = first_value(
    session["out_power"]
)

row["power_end"] = last_value(
    session["out_power"]
)

row["power_min"] = session[
    "out_power"
].min()

row["power_max"] = session[
    "out_power"
].max()

row["power_mean"] = session[
    "out_power"
].mean()

row["power_std"] = session[
    "out_power"
].std()

row["temp1_start"] = first_value(
    session["charging_gun_temperature1"]
)

row["temp1_end"] = last_value(
    session["charging_gun_temperature1"]
)

row["temp1_min"] = session[
    "charging_gun_temperature1"
].min()

row["temp1_max"] = session[
    "charging_gun_temperature1"
].max()

row["temp1_mean"] = session[
    "charging_gun_temperature1"
].mean()

row["temp1_std"] = session[
    "charging_gun_temperature1"
].std()

row["temp2_start"] = first_value(
    session["charging_gun_temperature2"]
)

row["temp2_end"] = last_value(
    session["charging_gun_temperature2"]
)

row["temp2_min"] = session[
    "charging_gun_temperature2"
].min()

row["temp2_max"] = session[
    "charging_gun_temperature2"
].max()

row["temp2_mean"] = session[
    "charging_gun_temperature2"
].mean()

row["temp2_std"] = session[
    "charging_gun_temperature2"
].std()

row["energy_start"] = first_value(
    session["current_energy_meter_value"]
)

row["energy_end"] = last_value(
    session["current_energy_meter_value"]
)

row["energy_max"] = session[
    "current_energy_meter_value"
].max()

row["duration_max"] = session[
    "total_charging_min"
].max()

row["num_points"] = len(session)


# ============================================================
# DERIVED FEATURES
# ============================================================

row["soc_gain"] = (
    row["soc_end"]
    - row["soc_start"]
)

row["voltage_change"] = (
    row["voltage_end"]
    - row["voltage_start"]
)

row["current_change"] = (
    row["current_end"]
    - row["current_start"]
)

row["power_change"] = (
    row["power_end"]
    - row["power_start"]
)

row["temp1_rise"] = (
    row["temp1_end"]
    - row["temp1_start"]
)

row["temp2_rise"] = (
    row["temp2_end"]
    - row["temp2_start"]
)

row["temp_sensor_difference"] = (
    row["temp1_max"]
    - row["temp2_max"]
)

row["energy_delivered"] = (
    row["energy_end"]
    - row["energy_start"]
)

if row["soc_gain"] > 0:

    row["energy_per_soc"] = (
        row["energy_delivered"]
        /
        row["soc_gain"]
    )

else:

    row["energy_per_soc"] = 0


# ============================================================
# SOC FLAGS
# ============================================================

row["high_soc_start"] = int(
    row["soc_start"] >= 80
)

row["high_soc_session"] = int(
    row["soc_max"] >= 90
)

row["very_high_soc_session"] = int(
    row["soc_max"] >= 95
)


# ============================================================
# BATTERY TYPE
# ============================================================

battery_type = str(
    session["types"].iloc[0]
)

for battery in [
    3,
    4,
    5,
    6
]:

    row[
        f"battery_type_{battery}"
    ] = int(
        battery_type == str(battery)
    )


# ============================================================
# EXACT MODEL FEATURE ORDER
# ============================================================

FEATURE_NAMES = [
    "soc_start",
    "soc_end",
    "soc_min",
    "soc_max",
    "soc_mean",
    "soc_std",
    "voltage_start",
    "voltage_end",
    "voltage_min",
    "voltage_max",
    "voltage_mean",
    "voltage_std",
    "current_start",
    "current_end",
    "current_min",
    "current_max",
    "current_mean",
    "current_std",
    "power_start",
    "power_end",
    "power_min",
    "power_max",
    "power_mean",
    "power_std",
    "temp1_start",
    "temp1_end",
    "temp1_min",
    "temp1_max",
    "temp1_mean",
    "temp1_std",
    "temp2_start",
    "temp2_end",
    "temp2_min",
    "temp2_max",
    "temp2_mean",
    "temp2_std",
    "energy_start",
    "energy_end",
    "energy_max",
    "duration_max",
    "num_points",
    "soc_gain",
    "voltage_change",
    "current_change",
    "power_change",
    "temp1_rise",
    "temp2_rise",
    "temp_sensor_difference",
    "energy_delivered",
    "energy_per_soc",
    "high_soc_start",
    "high_soc_session",
    "very_high_soc_session",
    "battery_type_3",
    "battery_type_4",
    "battery_type_5",
    "battery_type_6",
]


# ============================================================
# CLEAN VALUES
# ============================================================

values = []

for name in FEATURE_NAMES:

    value = row[name]

    if pd.isna(value):
        value = 0.0

    values.append(
        float(value)
    )


# ============================================================
# PRINT
# ============================================================

print("\n57 FEATURE VALUES")
print("-" * 70)

for i, (name, value) in enumerate(
    zip(FEATURE_NAMES, values)
):

    print(
        f"{i:2d}: "
        f"{name:28s} = "
        f"{value:.10f}"
    )


# ============================================================
# EXPORT C++ ARRAY
# ============================================================

with open(
    OUTPUT_PATH,
    "w"
) as f:

    f.write(
        "float model2_features[MODEL2_NUM_FEATURES] = {\n"
    )

    for i, value in enumerate(values):

        comma = "," if i < 56 else ""

        f.write(
            f"    {value:.10f}f{comma}"
            f"  // {i} {FEATURE_NAMES[i]}\n"
        )

    f.write("};\n")


print("\nSaved:")
print(OUTPUT_PATH)

print("\nExport complete.")