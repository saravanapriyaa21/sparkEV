import time
import pickle
import os
import numpy as np
import pandas as pd


# ============================================================
# MODEL 2 - CHARGING BEHAVIOUR / ABNORMAL SESSION DETECTION
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT, "models", "model2_charging_fault_xgb.pkl"
)

DATA_PATH = os.path.join(
    PROJECT_ROOT, "data", "raw", "processed_data_longer_than_30.xlsx"
)


# ============================================================
# EXACT 57 MODEL FEATURES
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
# LOAD TRAINED MODEL ONCE
# ============================================================

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

print("Model 2 loaded successfully")
print("Model type:", type(model).__name__)


# ============================================================
# VERIFY MODEL
# ============================================================

if hasattr(model, "n_features_in_"):

    if model.n_features_in_ != 57:

        raise ValueError(
            f"Model expects {model.n_features_in_} features, "
            f"but Model 2 requires 57."
        )


# ============================================================
# LOAD REAL MODEL 2 DATASET
# ============================================================

print("\nLoading Model 2 source dataset...")

df = pd.read_excel(
    DATA_PATH
)

print(
    "Raw dataset shape:",
    df.shape
)

print(
    "Number of charging sessions:",
    df["transaction_id"].nunique()
)


# ============================================================
# VERIFY REQUIRED COLUMNS
# ============================================================

REQUIRED_COLUMNS = [

    "transaction_id",
    "begin_time",
    "end_time",

    "current_soc",
    "chargingv",
    "charginga",
    "out_power",

    "charging_gun_temperature1",
    "charging_gun_temperature2",

    "current_energy_meter_value",

    "total_charging_min",

    "types",
    "label",
]


missing_columns = [
    col
    for col in REQUIRED_COLUMNS
    if col not in df.columns
]

if missing_columns:

    raise ValueError(
        "Missing required Model 2 columns: "
        + ", ".join(missing_columns)
    )


# ============================================================
# SAME PREPROCESSING AS TRAINING
# ============================================================

df["begin_time"] = pd.to_datetime(
    df["begin_time"]
)

df["end_time"] = pd.to_datetime(
    df["end_time"]
)

df = df.sort_values(
    ["transaction_id", "end_time"]
).reset_index(
    drop=True
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def first_value(x):
    return x.iloc[0]


def last_value(x):
    return x.iloc[-1]


# ============================================================
# CREATE SESSION-LEVEL FEATURES
# ============================================================

print("\nCreating session-level features...")


session_features = (

    df.groupby("transaction_id")

    .agg(

        label=("label", "first"),

        battery_type=("types", "first"),


        # ---------------- SOC ----------------

        soc_start=("current_soc", first_value),
        soc_end=("current_soc", last_value),
        soc_min=("current_soc", "min"),
        soc_max=("current_soc", "max"),
        soc_mean=("current_soc", "mean"),
        soc_std=("current_soc", "std"),


        # ---------------- VOLTAGE ----------------

        voltage_start=("chargingv", first_value),
        voltage_end=("chargingv", last_value),
        voltage_min=("chargingv", "min"),
        voltage_max=("chargingv", "max"),
        voltage_mean=("chargingv", "mean"),
        voltage_std=("chargingv", "std"),


        # ---------------- CURRENT ----------------

        current_start=("charginga", first_value),
        current_end=("charginga", last_value),
        current_min=("charginga", "min"),
        current_max=("charginga", "max"),
        current_mean=("charginga", "mean"),
        current_std=("charginga", "std"),


        # ---------------- POWER ----------------

        power_start=("out_power", first_value),
        power_end=("out_power", last_value),
        power_min=("out_power", "min"),
        power_max=("out_power", "max"),
        power_mean=("out_power", "mean"),
        power_std=("out_power", "std"),


        # ---------------- TEMPERATURE 1 ----------------

        temp1_start=(
            "charging_gun_temperature1",
            first_value
        ),

        temp1_end=(
            "charging_gun_temperature1",
            last_value
        ),

        temp1_min=(
            "charging_gun_temperature1",
            "min"
        ),

        temp1_max=(
            "charging_gun_temperature1",
            "max"
        ),

        temp1_mean=(
            "charging_gun_temperature1",
            "mean"
        ),

        temp1_std=(
            "charging_gun_temperature1",
            "std"
        ),


        # ---------------- TEMPERATURE 2 ----------------

        temp2_start=(
            "charging_gun_temperature2",
            first_value
        ),

        temp2_end=(
            "charging_gun_temperature2",
            last_value
        ),

        temp2_min=(
            "charging_gun_temperature2",
            "min"
        ),

        temp2_max=(
            "charging_gun_temperature2",
            "max"
        ),

        temp2_mean=(
            "charging_gun_temperature2",
            "mean"
        ),

        temp2_std=(
            "charging_gun_temperature2",
            "std"
        ),


        # ---------------- ENERGY ----------------

        energy_start=(
            "current_energy_meter_value",
            first_value
        ),

        energy_end=(
            "current_energy_meter_value",
            last_value
        ),

        energy_max=(
            "current_energy_meter_value",
            "max"
        ),


        # ---------------- DURATION ----------------

        duration_max=(
            "total_charging_min",
            "max"
        ),


        # ---------------- NUMBER OF POINTS ----------------

        num_points=(
            "transaction_id",
            "count"
        )
    )
)


# ============================================================
# DERIVED FEATURES
# ============================================================

session_features["soc_gain"] = (
    session_features["soc_end"]
    -
    session_features["soc_start"]
)

session_features["voltage_change"] = (
    session_features["voltage_end"]
    -
    session_features["voltage_start"]
)

session_features["current_change"] = (
    session_features["current_end"]
    -
    session_features["current_start"]
)

session_features["power_change"] = (
    session_features["power_end"]
    -
    session_features["power_start"]
)

session_features["temp1_rise"] = (
    session_features["temp1_end"]
    -
    session_features["temp1_start"]
)

session_features["temp2_rise"] = (
    session_features["temp2_end"]
    -
    session_features["temp2_start"]
)

session_features["temp_sensor_difference"] = (
    session_features["temp1_max"]
    -
    session_features["temp2_max"]
)

session_features["energy_delivered"] = (
    session_features["energy_end"]
    -
    session_features["energy_start"]
)


# ============================================================
# ENERGY PER SOC
# ============================================================

session_features["energy_per_soc"] = np.where(

    session_features["soc_gain"] > 0,

    session_features["energy_delivered"]
    /
    session_features["soc_gain"],

    0
)


# ============================================================
# SOC BEHAVIOURAL FEATURES
# ============================================================

session_features["high_soc_start"] = (
    session_features["soc_start"] >= 80
).astype(int)

session_features["high_soc_session"] = (
    session_features["soc_max"] >= 90
).astype(int)

session_features["very_high_soc_session"] = (
    session_features["soc_max"] >= 95
).astype(int)


# ============================================================
# HANDLE MISSING / INVALID VALUES
# ============================================================

session_features = (

    session_features

    .replace(
        [np.inf, -np.inf],
        np.nan
    )

    .fillna(0)
)


# ============================================================
# BATTERY TYPE ONE-HOT ENCODING
# ============================================================

session_features = pd.get_dummies(
    session_features,
    columns=["battery_type"],
    prefix="battery_type"
)


# ============================================================
# ENSURE ALL FOUR BATTERY TYPE FEATURES EXIST
# ============================================================

for battery_type in [3, 4, 5, 6]:

    column = f"battery_type_{battery_type}"

    if column not in session_features.columns:

        session_features[column] = 0


# ============================================================
# CONVERT BOOLEAN TO INTEGER
# ============================================================

for col in session_features.columns:

    if session_features[col].dtype == bool:

        session_features[col] = (
            session_features[col]
            .astype(int)
        )


# ============================================================
# VERIFY REQUIRED FEATURES
# ============================================================

missing_features = [

    feature
    for feature in FEATURE_NAMES
    if feature not in session_features.columns

]

if missing_features:

    raise ValueError(
        "Missing Model 2 features: "
        + ", ".join(missing_features)
    )


# ============================================================
# CREATE FINAL 57-FEATURE MATRIX
# ============================================================

X_sessions = session_features[
    FEATURE_NAMES
].copy()


print("\n")
print("=" * 65)
print("MODEL 2 DATA PREPARATION COMPLETE")
print("=" * 65)

print(
    "Feature matrix shape:",
    X_sessions.shape
)

print(
    "Number of features:",
    X_sessions.shape[1]
)

print(
    "Number of sessions:",
    X_sessions.shape[0]
)


if X_sessions.shape[1] != 57:

    raise ValueError(
        "Final Model 2 input does not contain exactly 57 features."
    )


# ============================================================
# SESSION IDS
# ============================================================

SESSION_IDS = X_sessions.index.tolist()


# ============================================================
# MODEL 2 INFERENCE
# ============================================================

def run_model2(transaction_id=None):

    """
    Run Model 2 using an actual charging session.

    transaction_id:
        The real transaction ID from the dataset.

    If transaction_id is None:
        The latest available session is used.
    """

    # --------------------------------------------------------
    # Select transaction
    # --------------------------------------------------------

    if transaction_id is None:

        transaction_id = SESSION_IDS[-1]

    if transaction_id not in X_sessions.index:

        raise ValueError(
            f"Transaction ID '{transaction_id}' "
            f"not found in Model 2 dataset."
        )


    # --------------------------------------------------------
    # Extract real 57-feature input
    # --------------------------------------------------------

    X_input = X_sessions.loc[
        [transaction_id]
    ].copy()


    # --------------------------------------------------------
    # Model inference
    # --------------------------------------------------------

    start_time = time.perf_counter()

    probability = float(
        model.predict_proba(
            X_input
        )[0][1]
    )

    prediction = (
        1
        if probability >= 0.5
        else 0
    )

    latency_ms = (
        time.perf_counter()
        -
        start_time
    ) * 1000


    # --------------------------------------------------------
    # Interpretation
    # --------------------------------------------------------

    if prediction == 1:

        status = (
            "Abnormal charging session"
        )

    else:

        status = (
            "Normal charging session"
        )


    return {

        "transaction_id": transaction_id,

        "prediction": prediction,

        "fault_probability": probability,

        "status": status,

        "latency_ms": latency_ms,
    }


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 65)
    print("SPARKEV - MODEL 2 REAL SESSION TEST")
    print("=" * 65)


    # --------------------------------------------------------
    # Test several actual transactions
    # --------------------------------------------------------

    test_sessions = SESSION_IDS[:10]


    for transaction_id in test_sessions:

        result = run_model2(
            transaction_id
        )

        print("\n" + "-" * 50)

        print(
            "Transaction ID   :",
            result["transaction_id"]
        )

        print(
            "Fault probability:",
            f"{result['fault_probability']:.6f}"
        )

        print(
            "Prediction       :",
            result["prediction"]
        )

        print(
            "Status           :",
            result["status"]
        )

        print(
            "Model 2 latency  :",
            f"{result['latency_ms']:.3f} ms"
        )