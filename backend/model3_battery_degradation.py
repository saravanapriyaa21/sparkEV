import os
import pickle
import time
import pandas as pd


# ============================================================
# LOAD MODEL 3 ARTIFACTS ONCE
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

with open(os.path.join(MODELS_DIR, "model3_battery_degradation_xgb.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(MODELS_DIR, "model3_scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

with open(os.path.join(MODELS_DIR, "model3_numerical_features.pkl"), "rb") as f:
    numerical_features = pickle.load(f)

with open(os.path.join(MODELS_DIR, "model3_target_columns.pkl"), "rb") as f:
    target_columns = pickle.load(f)


print("Model 3 loaded successfully")
print("Model features:", model.n_features_in_)
print("Numerical features:", len(numerical_features))
print("Targets:", target_columns)


# ============================================================
# MODEL 3 FEATURE INFORMATION
# ============================================================

model_features = model.get_booster().feature_names


# ============================================================
# MODEL 3 PREPROCESSING
# ============================================================

def preprocess_data(df):

    df = df.copy()

    # --------------------------------------------------------
    # Temporal features
    # --------------------------------------------------------

    if "t_start" in df.columns:
        df["t_start"] = pd.to_datetime(df["t_start"])

    if "t_end" in df.columns:
        df["t_end"] = pd.to_datetime(df["t_end"])

    if "t_start" in df.columns and "t_end" in df.columns:

        df["cycle_duration_hours"] = (
            df["t_end"] - df["t_start"]
        ).dt.total_seconds() / 3600

        df["start_month"] = df["t_start"].dt.month
        df["start_day_of_week"] = df["t_start"].dt.dayofweek
        df["start_hour"] = df["t_start"].dt.hour

        df.drop(
            columns=["t_start", "t_end"],
            inplace=True
        )

    # --------------------------------------------------------
    # Remove target columns if present
    # --------------------------------------------------------

    for col in ["cap", "cap_corr", "soh"]:

        if col in df.columns:

            df.drop(
                columns=[col],
                inplace=True
            )

    # --------------------------------------------------------
    # Handle car identifier
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
    # Scale numerical features
    # --------------------------------------------------------

    df[numerical_features] = scaler.transform(
        df[numerical_features]
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
    # Exact XGBoost feature order
    # --------------------------------------------------------

    df = df[model_features]

    return df


# ============================================================
# RISK CLASSIFICATION
# ============================================================

def get_risk_level(soh):

    if soh >= 0.95:

        return "Low"

    elif soh >= 0.90:

        return "Medium"

    else:

        return "High"


# ============================================================
# PREPROCESS DATA
# ============================================================

def prepare_model3_data(raw_data):

    prepared_data = preprocess_data(
        raw_data
    )

    if prepared_data.shape[1] != model.n_features_in_:

        raise ValueError(

            f"Expected {model.n_features_in_} features, "

            f"but received {prepared_data.shape[1]}"

        )

    return prepared_data


# ============================================================
# MODEL 3 INFERENCE
# ============================================================

def run_model3(
    raw_data=None,
    preprocessed_data=None
):

    # --------------------------------------------------------
    # Select input
    # --------------------------------------------------------

    if preprocessed_data is not None:

        X = preprocessed_data

    elif raw_data is not None:

        X = preprocess_data(
            raw_data
        )

    else:

        raise ValueError(
            "No Model 3 input data available."
        )

    # --------------------------------------------------------
    # Feature validation
    # --------------------------------------------------------

    if X.shape[1] != model.n_features_in_:

        raise ValueError(

            f"Expected {model.n_features_in_} features, "

            f"but received {X.shape[1]}"

        )

    # --------------------------------------------------------
    # Model inference
    # --------------------------------------------------------

    start_time = time.perf_counter()

    prediction = model.predict(X)

    end_time = time.perf_counter()

    inference_latency_ms = (

        end_time - start_time

    ) * 1000

    # --------------------------------------------------------
    # Convert all predictions to DataFrame
    # --------------------------------------------------------

    prediction_df = pd.DataFrame(

        prediction,

        columns=target_columns

    )

    # --------------------------------------------------------
    # Select latest available record
    #
    # The test file contains multiple battery records.
    # We use the final record as the current battery state.
    # --------------------------------------------------------

    latest_prediction = prediction_df.iloc[-1]

    cap = float(
        latest_prediction["cap"]
    )

    cap_corr = float(
        latest_prediction["cap_corr"]
    )

    soh = float(
        latest_prediction["soh"]
    )

    # --------------------------------------------------------
    # Risk
    # --------------------------------------------------------

    risk = get_risk_level(
        soh
    )

    # --------------------------------------------------------
    # Return
    # --------------------------------------------------------

    return {

        "cap": round(
            cap,
            4
        ),

        "cap_corr": round(
            cap_corr,
            4
        ),

        "soh": round(
            soh,
            4
        ),

        "soh_percent": round(
            soh * 100,
            2
        ),

        "battery_risk": risk,

        "model_3_latency_ms": round(
            inference_latency_ms,
            3
        )
    }


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":

    file_path = os.path.join(
        PROJECT_ROOT, "data", "raw", "NMC01_Car02_fast_feature.xlsx"
    )

    print()

    print(
        "Loading test data:",
        file_path
    )

    raw_data = pd.read_excel(
        file_path
    )

    print(
        "Input rows:",
        len(raw_data)
    )

    print(
        "Input columns:",
        len(raw_data.columns)
    )

    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    print(
        "Preprocessing Model 3 data..."
    )

    prepared_data = prepare_model3_data(
        raw_data
    )

    print(
        "Model 3 preprocessing completed."
    )

    print(
        "Prepared shape:",
        prepared_data.shape
    )

    # --------------------------------------------------------
    # Run inference
    # --------------------------------------------------------

    result = run_model3(
        preprocessed_data=prepared_data
    )

    # --------------------------------------------------------
    # Display result
    # --------------------------------------------------------

    print()
    print("=" * 55)
    print("MODEL 3 RESULT")
    print("=" * 55)

    print(
        f"CAP          : "
        f"{result['cap']}"
    )

    print(
        f"CAP_CORR     : "
        f"{result['cap_corr']}"
    )

    print(
        f"SOH          : "
        f"{result['soh']}"
    )

    print(
        f"SOH (%)      : "
        f"{result['soh_percent']}%"
    )

    print(
        f"Battery Risk : "
        f"{result['battery_risk']}"
    )

    print(
        f"Model 3 Latency : "
        f"{result['model_3_latency_ms']} ms"
    )

    print("=" * 55)