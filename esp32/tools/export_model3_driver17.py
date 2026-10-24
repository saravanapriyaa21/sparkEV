import os
import pickle
import pandas as pd
import numpy as np


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

MODEL_PATH = os.path.join(MODELS_DIR, "model3_battery_degradation_xgb.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "model3_scaler.pkl")
FEATURES_PATH = os.path.join(MODELS_DIR, "model3_numerical_features.pkl")
TARGET_COLUMNS_PATH = os.path.join(MODELS_DIR, "model3_target_columns.pkl")

DATA_PATH = os.path.join(DATA_DIR, "LFP01_fast_feature.xlsx")

OUTPUT_PATH = os.path.join(VECTORS_DIR, "model3_driver17_vector.txt")


# ============================================================
# LOAD MODEL + PREPROCESSING ARTIFACTS
# ============================================================

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

with open(FEATURES_PATH, "rb") as f:
    numerical_features = pickle.load(f)

with open(TARGET_COLUMNS_PATH, "rb") as f:
    target_columns = pickle.load(f)


booster = model.get_booster()

model_features = booster.feature_names


print("=" * 70)
print("SPARKEV MODEL 3 TEST VECTOR EXPORT")
print("=" * 70)

print()
print("Model type        :", type(model))
print("Model n_estimators:", model.n_estimators)
print("Actual trees      :", len(booster.get_dump()))
print("Model features    :", len(model_features))
print("Numerical features:", len(numerical_features))
print("Scaler features   :", scaler.n_features_in_)


# ============================================================
# LOAD DATA
# ============================================================

raw_data = pd.read_excel(
    DATA_PATH
)

print()
print(
    "Raw dataset shape:",
    raw_data.shape
)


# ============================================================
# EXACT MODEL 3 PREPROCESSING
# ============================================================

def preprocess_data(df):

    df = df.copy()

    # --------------------------------------------------------
    # Time features
    # --------------------------------------------------------

    if "t_start" in df.columns:

        df["t_start"] = pd.to_datetime(
            df["t_start"]
        )

    if "t_end" in df.columns:

        df["t_end"] = pd.to_datetime(
            df["t_end"]
        )

    if (
        "t_start" in df.columns
        and
        "t_end" in df.columns
    ):

        df["cycle_duration_hours"] = (
            df["t_end"] - df["t_start"]
        ).dt.total_seconds() / 3600

        df["start_month"] = (
            df["t_start"].dt.month
        )

        df["start_day_of_week"] = (
            df["t_start"].dt.dayofweek
        )

        df["start_hour"] = (
            df["t_start"].dt.hour
        )

        df.drop(
            columns=[
                "t_start",
                "t_end"
            ],
            inplace=True
        )


    # --------------------------------------------------------
    # Remove target columns
    # --------------------------------------------------------

    for col in [
        "cap",
        "cap_corr",
        "soh"
    ]:

        if col in df.columns:

            df.drop(
                columns=[col],
                inplace=True
            )


    # --------------------------------------------------------
    # Car identifier
    # --------------------------------------------------------

    if "car_identifier" in df.columns:

        df["car_identifier"] = (
            df["car_identifier"]
            .fillna("Unknown")
            .astype(str)
        )

        df = pd.get_dummies(
            df,
            columns=["car_identifier"],
            prefix="car"
        )


    # --------------------------------------------------------
    # Add missing numerical features
    # --------------------------------------------------------

    missing_numerical = [
        col
        for col in numerical_features
        if col not in df.columns
    ]

    if missing_numerical:

        print()
        print(
            "Adding missing numerical features:",
            len(missing_numerical)
        )

        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    0.0,
                    index=df.index,
                    columns=missing_numerical
                )
            ],
            axis=1
        )


    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Do NOT median-impute numerical values.
    #
    # The verified Model 3 pipeline preserves NaN values
    # because XGBoost performs native missing-value routing.
    # --------------------------------------------------------

    df[numerical_features] = (
        scaler.transform(
            df[numerical_features]
        )
    )


    # --------------------------------------------------------
    # Add missing model features
    # --------------------------------------------------------

    missing_model_features = [
        col
        for col in model_features
        if col not in df.columns
    ]

    if missing_model_features:

        print()
        print(
            "Adding missing model features:",
            len(missing_model_features)
        )

        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    0,
                    index=df.index,
                    columns=missing_model_features
                )
            ],
            axis=1
        )


    # --------------------------------------------------------
    # Exact model feature order
    # --------------------------------------------------------

    df = df[
        model_features
    ]

    return df


# ============================================================
# PREPROCESS
# ============================================================

X = preprocess_data(
    raw_data
)

print()
print(
    "Prepared matrix shape:",
    X.shape
)


# ============================================================
# VALIDATE FEATURE COUNT
# ============================================================

if X.shape[1] != 199:

    raise RuntimeError(
        f"Expected 199 model features, "
        f"got {X.shape[1]}"
    )


# ============================================================
# CHECK MISSING VALUES
# ============================================================

nan_count = int(
    X.iloc[-1].isna().sum()
)

print(
    "NaN values in final row:",
    nan_count
)


# ============================================================
# CPU PREDICTION
# ============================================================

prediction = model.predict(
    X
)


# ============================================================
# IMPORTANT:
#
# Use FINAL row.
#
# This matches the verified Model 3 pipeline:
#
# latest_prediction = prediction_df.iloc[-1]
# ============================================================

final_row_index = len(X) - 1

final_prediction = prediction[
    final_row_index
]


cap = float(
    final_prediction[0]
)

cap_corr = float(
    final_prediction[1]
)

soh = float(
    final_prediction[2]
)


# ============================================================
# GOLDEN RESULT
# ============================================================

print()
print("=" * 70)
print("CURRENT MODEL 3 CPU REFERENCE")
print("=" * 70)

print(
    f"CAP      = {cap:.10f}"
)

print(
    f"CAP_CORR = {cap_corr:.10f}"
)

print(
    f"SOH      = {soh:.10f}"
)

print(
    f"SOH_PCT  = {soh * 100:.6f}"
)


# ============================================================
# CAR FEATURES
# ============================================================

car_features = [
    col
    for col in model_features
    if col.startswith("car_")
]


print()
print("=" * 70)
print("ACTIVE CAR FEATURES IN FINAL ROW")
print("=" * 70)

for col in car_features:

    value = float(
        X.iloc[
            final_row_index
        ][col]
    )

    if value != 0:

        print(
            f"{col} = {value}"
        )


# ============================================================
# SAVE CURRENT ESP32 TEST VECTOR
# ============================================================

with open(
    OUTPUT_PATH,
    "w"
) as f:

    f.write(
        "SPARKEV MODEL 3 ESP32 TEST VECTOR\n"
    )

    f.write(
        "=================================\n\n"
    )

    f.write(
        "Model: XGBRegressor\n"
    )

    f.write(
        "XGBoost actual trees: 300\n"
    )

    f.write(
        "Features: 199\n"
    )

    f.write(
        "Vector row: FINAL ROW\n\n"
    )


    # --------------------------------------------------------
    # CPU reference
    # --------------------------------------------------------

    f.write(
        "CPU REFERENCE\n"
    )

    f.write(
        f"CAP      = {cap:.10f}\n"
    )

    f.write(
        f"CAP_CORR = {cap_corr:.10f}\n"
    )

    f.write(
        f"SOH      = {soh:.10f}\n"
    )

    f.write(
        f"SOH_PCT  = {soh * 100:.6f}\n\n"
    )


    # --------------------------------------------------------
    # Feature vector
    # --------------------------------------------------------

    f.write(
        "199 FEATURES\n"
    )

    f.write(
        "============\n"
    )

    for i, feature in enumerate(
        model_features
    ):

        value = X.iloc[
            final_row_index
        ][feature]


        # Preserve NaN exactly.
        if pd.isna(value):

            value_text = "NAN"

        elif np.isposinf(value):

            value_text = "INFINITY"

        elif np.isneginf(value):

            value_text = "-INFINITY"

        else:

            value_text = (
                f"{float(value):.10g}f"
            )


        f.write(
            f"{i:3d}: "
            f"{feature:40s} = "
            f"{value_text}\n"
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("CURRENT MODEL 3 TEST VECTOR CREATED")
print("=" * 70)

print(
    "Output:",
    OUTPUT_PATH
)

print(
    "Features:",
    X.shape[1]
)

print(
    "Selected row:",
    final_row_index
)

print(
    "NaN values:",
    nan_count
)

print()
print("Golden CPU reference:")

print(
    f"CAP      = {cap:.8f}"
)

print(
    f"CAP_CORR = {cap_corr:.8f}"
)

print(
    f"SOH      = {soh:.8f}"
)

print(
    f"SOH %    = {soh * 100:.4f}%"
)

print("=" * 70)