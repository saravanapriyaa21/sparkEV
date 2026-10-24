from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import time
import os
import pandas as pd
import xgboost as xgb

from model2_charging_fault import run_model2

from model3_battery_degradation import (
    run_model3,
    prepare_model3_data
)


# ==================================================
# FASTAPI APP
# ==================================================

app = FastAPI()


# ==================================================
# CORS
# ==================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================================================
# LOAD MODEL 1
#
# Used only by OLD /predict testing endpoint.
# Final ESP32 architecture uses ESP32 Model 1.
# ==================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT, "models", "model1_battery_stress_xgb.json"
)

model1 = xgb.XGBRegressor()

model1.load_model(MODEL_PATH)

print("Model 1 loaded successfully")


# ==================================================
# MODEL 2
#
# Used only by OLD /predict testing endpoint.
# Final ESP32 architecture uses ESP32 Model 2.
# ==================================================

print("Model 2 loaded through model2_charging_fault.py")


# ==================================================
# MODEL 2 DEMO SESSION ASSIGNMENT
# ==================================================

MODEL2_DRIVER_TRANSACTIONS = {

    17: 44010000100000052203030401089310,

    24: 44020100400000012010151800001210,

}


# ==================================================
# LOAD MODEL 3 DRIVER-SPECIFIC DATA
#
# Used only by OLD /predict testing endpoint.
# Final ESP32 architecture uses ESP32 Model 3.
# ==================================================

MODEL3_DATA_DIR = os.path.join(
    PROJECT_ROOT, "data", "raw"
)


MODEL3_DRIVER_FILES = {

    17: "LFP01_fast_feature.xlsx",

    18: "NMC02_fast_feature.xlsx",

    19: "Lab_fast_feature.xlsx",

    20: "NMC01_Car01_fast_feature.xlsx",

    21: "NMC01_slow_feature.xlsx",

    24: "NMC01_Car02_fast_feature.xlsx",

}


# ==================================================
# MODEL 3 PREPROCESSED DATA CACHE
# ==================================================

model3_driver_data = {}


print(
    "\nLoading Model 3 driver-specific datasets..."
)


for driver_id, filename in MODEL3_DRIVER_FILES.items():

    file_path = os.path.join(
        MODEL3_DATA_DIR,
        filename
    )

    if not os.path.exists(file_path):

        raise FileNotFoundError(
            f"Model 3 dataset not found for "
            f"Driver {driver_id}: {file_path}"
        )

    raw_data = pd.read_excel(
        file_path
    )

    print(
        f"Driver {driver_id}: "
        f"{filename} "
        f"({len(raw_data)} rows)"
    )

    print(
        f"Preprocessing Model 3 data "
        f"for Driver {driver_id}..."
    )

    prepared_data = prepare_model3_data(
        raw_data
    )

    model3_driver_data[
        driver_id
    ] = prepared_data

    print(
        f"Driver {driver_id} preprocessing "
        f"completed. "
        f"Shape: {prepared_data.shape}"
    )


print(
    "All Model 3 driver datasets "
    "loaded and cached."
)


# ==================================================
# OLD REQUEST FORMAT
#
# Used only by /predict.
# ==================================================

class DriverRequest(BaseModel):

    driver_id: int


# ==================================================
# ESP32 EDGE RESULT FORMAT
#
# ESP32 performs ALL THREE ML INFERENCES.
#
# FastAPI receives:
#
#   Model 1 prediction
#   Model 2 prediction
#   Model 3 prediction
#
# AND their actual ESP32-side latency.
#
# FastAPI does NOT run XGBoost here.
#
# FastAPI only:
#
#   1. Receives edge predictions
#   2. Generates recommendations
#   3. Stores result
#   4. Serves /latest
# ==================================================

class ESP32Result(BaseModel):

    # ----------------------------------------------
    # Driver
    # ----------------------------------------------

    driver_id: int


    # ----------------------------------------------
    # Model 1
    # ----------------------------------------------

    stress: float

    stress_status: str

    model_1_latency_ms: float


    # ----------------------------------------------
    # Model 2
    # ----------------------------------------------

    charging: str

    model_2_prediction: int

    model_2_probability: float

    model_2_status: str

    model_2_latency_ms: float


    # ----------------------------------------------
    # Model 3
    # ----------------------------------------------

    soh: float

    battery_risk: str

    cap: float

    cap_corr: float

    model_3_latency_ms: float


    # ----------------------------------------------
    # TOTAL ESP32 EDGE INFERENCE LATENCY
    #
    # This is measured on ESP32.
    #
    # It represents:
    #
    # Model 1
    # + Model 2
    # + Model 3
    #
    # It does NOT include:
    #
    # Wi-Fi
    # FastAPI
    # React
    # Recommendations
    # ----------------------------------------------

    edge_inference_latency_ms: float


    # ----------------------------------------------
    # Recommendations
    #
    # ESP32 does not generate these.
    # FastAPI generates them.
    # ----------------------------------------------

    recommendations: list[str] = []


# ==================================================
# MODEL 1 FEATURE ORDER
# ==================================================

FEATURE_NAMES = [

    "avg_speed_kmh",
    "speed_std_kmh",
    "max_speed_kmh",

    "accel_mean",
    "accel_std",
    "accel_max",
    "accel_min",

    "jerk_std",

    "hard_accel_count",
    "hard_brake_count",

    "idle_time_pct",
    "regen_braking_pct",

    "driver_aggressiveness_score",

    "ambient_temp_mean_c",

    "route_Highway",
    "route_Munich East",
    "route_Munich North",
    "route_Munich South",
    "route_Other",

    "weather_cloudy",
    "weather_dark",
    "weather_slightly cloudy",
    "weather_sunny",
    "weather_sunrise",

]


# ==================================================
# MODEL 1 DEMO FEATURE DATA
#
# Used by OLD /predict endpoint and
# personalized recommendation engine.
# ==================================================

driver_features = {

    17: {

        "avg_speed_kmh": 68.0,
        "speed_std_kmh": 21.0,
        "max_speed_kmh": 118.0,

        "accel_mean": 0.42,
        "accel_std": 1.15,
        "accel_max": 3.20,
        "accel_min": -3.00,

        "jerk_std": 1.35,

        "hard_accel_count": 8,
        "hard_brake_count": 6,

        "idle_time_pct": 8.0,
        "regen_braking_pct": 18.0,

        "driver_aggressiveness_score": 72.0,

        "ambient_temp_mean_c": 29.0,

        "route_Highway": 1,
        "route_Munich East": 0,
        "route_Munich North": 0,
        "route_Munich South": 0,
        "route_Other": 0,

        "weather_cloudy": 0,
        "weather_dark": 0,
        "weather_slightly cloudy": 0,
        "weather_sunny": 1,
        "weather_sunrise": 0,

    },


    24: {

        "avg_speed_kmh": 45.0,
        "speed_std_kmh": 12.0,
        "max_speed_kmh": 82.0,

        "accel_mean": 0.18,
        "accel_std": 0.55,
        "accel_max": 1.60,
        "accel_min": -1.40,

        "jerk_std": 0.65,

        "hard_accel_count": 2,
        "hard_brake_count": 1,

        "idle_time_pct": 5.0,
        "regen_braking_pct": 24.0,

        "driver_aggressiveness_score": 45.0,

        "ambient_temp_mean_c": 27.0,

        "route_Highway": 0,
        "route_Munich East": 1,
        "route_Munich North": 0,
        "route_Munich South": 0,
        "route_Other": 0,

        "weather_cloudy": 0,
        "weather_dark": 0,
        "weather_slightly cloudy": 1,
        "weather_sunny": 0,
        "weather_sunrise": 0,

    }

}


# ==================================================
# MODEL 1 STATUS
# ==================================================

def get_stress_status(stress):

    if stress < 40:

        return "Low"

    elif stress < 70:

        return "Moderate"

    else:

        return "High"


# ==================================================
# MODEL 1 RECOMMENDATIONS
# ==================================================

def get_driving_recommendations(
    features,
    stress
):

    recommendations = []


    # ----------------------------------------------
    # Hard acceleration
    # ----------------------------------------------

    if features["hard_accel_count"] >= 5:

        recommendations.append(
            "Frequent hard acceleration detected; "
            "use smoother acceleration."
        )


    # ----------------------------------------------
    # Hard braking
    # ----------------------------------------------

    if features["hard_brake_count"] >= 5:

        recommendations.append(
            "Frequent hard braking detected; "
            "brake more smoothly when possible."
        )


    # ----------------------------------------------
    # Driving aggressiveness
    # ----------------------------------------------

    if features[
        "driver_aggressiveness_score"
    ] >= 70:

        recommendations.append(
            "High driving aggressiveness detected; "
            "adopt smoother driving behaviour."
        )


    # ----------------------------------------------
    # Regenerative braking
    # ----------------------------------------------

    if features["regen_braking_pct"] < 15:

        recommendations.append(
            "Low regenerative braking usage; "
            "use regenerative braking more consistently."
        )


    # ----------------------------------------------
    # High battery stress
    # ----------------------------------------------

    if stress >= 70:

        recommendations.append(
            "High battery stress detected; "
            "adopt smoother driving behaviour."
        )


    # ----------------------------------------------
    # No warnings
    # ----------------------------------------------

    if not recommendations:

        recommendations.append(
            "Driving behaviour is relatively smooth."
        )


    return recommendations


# ==================================================
# MODEL 2 RECOMMENDATIONS
# ==================================================

def get_charging_recommendations(
    prediction,
    probability
):

    recommendations = []


    if prediction == 1:

        recommendations.append(
            "Abnormal charging behaviour detected; "
            "review the charging session before continued use."
        )

        recommendations.append(
            "Avoid prolonged high-stress charging sessions "
            "until the charging behaviour is verified."
        )

    else:

        recommendations.append(
            "Charging behaviour is within the "
            "normal operating pattern."
        )

        recommendations.append(
            "Continue following recommended charging practices."
        )


    return recommendations


# ==================================================
# MODEL 3 RECOMMENDATIONS
# ==================================================

def get_battery_recommendations(
    soh,
    battery_risk
):

    recommendations = []


    if battery_risk == "High":

        recommendations.append(
            f"Battery health is reduced ({soh:.2f}% SOH); "
            "avoid unnecessary charging and thermal stress."
        )

        recommendations.append(
            "Prefer moderate charging conditions and "
            "avoid prolonged high-stress operation."
        )


    elif battery_risk == "Medium":

        recommendations.append(
            f"Battery health is moderately reduced ({soh:.2f}% SOH); "
            "follow consistent charging practices."
        )


    elif battery_risk == "Low":

        recommendations.append(
            f"Battery health is currently good ({soh:.2f}% SOH)."
        )


    else:

        recommendations.append(
            f"Battery health status is {battery_risk} "
            f"({soh:.2f}% SOH); monitor battery behaviour."
        )


    return recommendations


# ==================================================
# CROSS-MODEL PERSONALIZED RECOMMENDATIONS
# ==================================================

def get_combined_recommendations(
    driving_recommendations,
    charging_recommendations,
    battery_recommendations,
    stress,
    model2_prediction,
    battery_risk
):

    recommendations = []


    # ----------------------------------------------
    # Driving
    # ----------------------------------------------

    recommendations.extend(
        driving_recommendations
    )


    # ----------------------------------------------
    # Charging
    # ----------------------------------------------

    recommendations.extend(
        charging_recommendations
    )


    # ----------------------------------------------
    # Battery
    # ----------------------------------------------

    recommendations.extend(
        battery_recommendations
    )


    # ----------------------------------------------
    # Abnormal charging + high battery risk
    # ----------------------------------------------

    if (
        model2_prediction == 1
        and battery_risk == "High"
    ):

        recommendations.append(
            "Abnormal charging behaviour and high "
            "battery health risk were detected together; "
            "minimize charging and driving stress until "
            "the battery and charging system are checked."
        )


    # ----------------------------------------------
    # High driving stress + high battery risk
    # ----------------------------------------------

    elif (
        stress >= 70
        and battery_risk == "High"
    ):

        recommendations.append(
            "High driving stress combined with reduced "
            "battery health; use smoother driving "
            "and avoid unnecessary high-load operation."
        )


    return recommendations


# ==================================================
# LATEST RESULT
# ==================================================

latest_result = None


# ==================================================
# PREDICTION COUNTER
# ==================================================

prediction_counter = 0


# ==================================================
# ESP32 EDGE RESULT ENDPOINT
#
# FINAL ARCHITECTURE:
#
# ESP32
#   |
#   +-- Model 1
#   +-- Model 2
#   +-- Model 3
#   |
#   +-- Edge latency measurements
#   |
#   v
# Wi-Fi
#   |
#   v
# FastAPI /esp32/result
#   |
#   +-- Personalized recommendation engine
#   |
#   v
# /latest
#   |
#   v
# React Dashboard
# ==================================================

@app.post("/esp32/result")
def receive_esp32_result(
    data: ESP32Result
):

    global latest_result
    global prediction_counter


    # ==================================================
    # SERVER RECEIVE TIMER
    # ==================================================

    receive_start = time.perf_counter()


    # ==================================================
    # VALIDATE DRIVER
    # ==================================================

    if data.driver_id not in driver_features:

        raise HTTPException(
            status_code=400,
            detail=(
                f"Driver {data.driver_id} is not "
                "configured for ESP32 demo."
            )
        )


    # ==================================================
    # GET DRIVER PROFILE
    #
    # Used to personalize driving recommendations.
    # ==================================================

    features = driver_features[
        data.driver_id
    ]


    # ==================================================
    # NORMALIZE BATTERY RISK
    #
    # ESP32:
    #
    #   LOW
    #   MODERATE
    #   HIGH
    #   CRITICAL
    #
    # Backend:
    #
    #   Low
    #   Medium
    #   High
    # ==================================================

    raw_battery_risk = str(
        data.battery_risk
    ).strip().upper()


    if raw_battery_risk == "LOW":

        battery_risk = "Low"


    elif raw_battery_risk in [
        "MODERATE",
        "MEDIUM"
    ]:

        battery_risk = "Medium"


    elif raw_battery_risk == "HIGH":

        battery_risk = "High"


    elif raw_battery_risk == "CRITICAL":

        battery_risk = "High"


    else:

        battery_risk = data.battery_risk


    # ==================================================
    # MODEL 1 PERSONALIZED RECOMMENDATIONS
    # ==================================================

    driving_recommendations = (
        get_driving_recommendations(

            features,

            data.stress

        )
    )


    # ==================================================
    # MODEL 2 PERSONALIZED RECOMMENDATIONS
    # ==================================================

    charging_recommendations = (
        get_charging_recommendations(

            data.model_2_prediction,

            data.model_2_probability

        )
    )


    # ==================================================
    # MODEL 3 PERSONALIZED RECOMMENDATIONS
    # ==================================================

    battery_recommendations = (
        get_battery_recommendations(

            data.soh,

            battery_risk

        )
    )


    # ==================================================
    # COMBINED PERSONALIZED INSIGHTS
    # ==================================================

    recommendations = (
        get_combined_recommendations(

            driving_recommendations,

            charging_recommendations,

            battery_recommendations,

            data.stress,

            data.model_2_prediction,

            battery_risk

        )
    )


    # ==================================================
    # PREDICTION ID
    # ==================================================

    prediction_counter += 1


    # ==================================================
    # SERVER PROCESSING TIME
    #
    # This is NOT ESP32 latency.
    #
    # It only measures FastAPI-side processing.
    # ==================================================

    receive_end = time.perf_counter()


    server_latency_ms = (

        receive_end
        - receive_start

    ) * 1000


    # ==================================================
    # STORE FINAL RESULT
    # ==================================================

    latest_result = {

        # ----------------------------------------------
        # Driver
        # ----------------------------------------------

        "driver_id":
            data.driver_id,


        # ----------------------------------------------
        # Model 1
        # ----------------------------------------------

        "stress":
            round(
                data.stress,
                2
            ),

        "stress_status":
            data.stress_status,

        "model_1_latency_ms":
            round(
                data.model_1_latency_ms,
                3
            ),


        # ----------------------------------------------
        # Model 2
        # ----------------------------------------------

        "charging":
            data.charging,

        "model_2_prediction":
            data.model_2_prediction,

        "model_2_probability":
            round(
                data.model_2_probability,
                6
            ),

        "model_2_status":
            data.model_2_status,

        "model_2_latency_ms":
            round(
                data.model_2_latency_ms,
                3
            ),


        # ----------------------------------------------
        # Model 3
        # ----------------------------------------------

        "soh":
            round(
                data.soh,
                2
            ),

        "battery_risk":
            battery_risk,

        "cap":
            round(
                data.cap,
                4
            ),

        "cap_corr":
            round(
                data.cap_corr,
                4
            ),

        "model_3_latency_ms":
            round(
                data.model_3_latency_ms,
                3
            ),


        # ----------------------------------------------
        # TOTAL ESP32 EDGE INFERENCE LATENCY
        #
        # Directly received from ESP32.
        # ----------------------------------------------

        "edge_inference_latency_ms":
            round(
                data.edge_inference_latency_ms,
                3
            ),


        # ----------------------------------------------
        # PERSONALIZED INSIGHTS
        # ----------------------------------------------

        "recommendations":
            recommendations,


        # ----------------------------------------------
        # SYSTEM
        # ----------------------------------------------

        "prediction_id":
            prediction_counter,

        "server_latency_ms":
            round(
                server_latency_ms,
                3
            )

    }


    # ==================================================
    # TERMINAL LOG
    # ==================================================

    print()

    print(
        "========================================"
    )

    print(
        "ESP32 EDGE RESULT RECEIVED"
    )

    print(
        "========================================"
    )


    print(
        "Driver       :",
        data.driver_id
    )


    print(
        "Stress       :",
        data.stress
    )


    print(
        "Stress Status:",
        data.stress_status
    )


    print(
        "Charging     :",
        data.charging
    )


    print(
        "Charging Prob:",
        data.model_2_probability
    )


    print(
        "SOH          :",
        data.soh
    )


    print(
        "Battery Risk :",
        battery_risk
    )


    print()

    print(
        "ESP32 INFERENCE LATENCY:"
    )


    print(
        "Model 1      :",
        data.model_1_latency_ms,
        "ms"
    )


    print(
        "Model 2      :",
        data.model_2_latency_ms,
        "ms"
    )


    print(
        "Model 3      :",
        data.model_3_latency_ms,
        "ms"
    )


    print(
        "TOTAL EDGE   :",
        data.edge_inference_latency_ms,
        "ms"
    )


    print()

    print(
        "PERSONALIZED INSIGHTS:"
    )


    for index, recommendation in enumerate(
        recommendations,
        start=1
    ):

        print(
            f"{index}. {recommendation}"
        )


    print()

    print(
        "FastAPI time :",
        round(
            server_latency_ms,
            3
        ),
        "ms"
    )


    print(
        "Prediction ID:",
        prediction_counter
    )


    print(
        "========================================"
    )


    # ==================================================
    # RESPONSE TO ESP32
    # ==================================================

    return {

        "status":
            "received",

        "driver_id":
            data.driver_id,

        "prediction_id":
            prediction_counter,

        "recommendations":
            recommendations

    }


# ==================================================
# OLD PYTHON PREDICTION ENDPOINT
#
# BACKUP / TESTING ONLY.
#
# This endpoint runs models on the computer.
#
# FINAL EDGE ARCHITECTURE DOES NOT USE THIS.
# ==================================================

@app.post("/predict")
def predict(data: DriverRequest):

    global latest_result
    global prediction_counter


    server_start = time.perf_counter()


    driver_id = data.driver_id


    # ==================================================
    # VALIDATE DRIVER
    # ==================================================

    if driver_id not in driver_features:

        raise HTTPException(
            status_code=400,
            detail=(
                f"Driver {driver_id} is not configured. "
                f"Available drivers: "
                f"{sorted(driver_features.keys())}"
            )
        )


    if driver_id not in model3_driver_data:

        raise HTTPException(
            status_code=400,
            detail=(
                f"Model 3 data is not configured for "
                f"Driver {driver_id}."
            )
        )


    if driver_id not in MODEL2_DRIVER_TRANSACTIONS:

        raise HTTPException(
            status_code=400,
            detail=(
                f"Model 2 charging session is not "
                f"configured for Driver {driver_id}."
            )
        )


    # ==================================================
    # MODEL 1
    # ==================================================

    features = driver_features[
        driver_id
    ]


    feature_values = [

        features[name]

        for name in FEATURE_NAMES

    ]


    X1 = pd.DataFrame(

        [feature_values],

        columns=FEATURE_NAMES

    )


    model1_start = time.perf_counter()


    prediction1 = model1.predict(
        X1
    )


    model1_end = time.perf_counter()


    model1_latency_ms = (

        model1_end
        - model1_start

    ) * 1000


    stress = float(
        prediction1[0]
    )


    stress = max(
        0.0,
        min(
            100.0,
            stress
        )
    )


    stress = round(
        stress,
        2
    )


    stress_status = get_stress_status(
        stress
    )


    driving_recommendations = (
        get_driving_recommendations(

            features,

            stress

        )
    )


    # ==================================================
    # MODEL 2
    # ==================================================

    transaction_id = (
        MODEL2_DRIVER_TRANSACTIONS[
            driver_id
        ]
    )


    model2_result = run_model2(
        transaction_id=transaction_id
    )


    model2_prediction = int(
        model2_result["prediction"]
    )


    model2_probability = float(
        model2_result["fault_probability"]
    )


    model2_status = (
        model2_result["status"]
    )


    model2_latency_ms = float(
        model2_result["latency_ms"]
    )


    if model2_prediction == 1:

        charging = "Faulty"

    else:

        charging = "Healthy"


    charging_recommendations = (
        get_charging_recommendations(

            model2_prediction,

            model2_probability

        )
    )


    # ==================================================
    # MODEL 3
    # ==================================================

    model3_result = run_model3(

        preprocessed_data=
        model3_driver_data[
            driver_id
        ]

    )


    model3_latency_ms = float(
        model3_result[
            "model_3_latency_ms"
        ]
    )


    soh = float(
        model3_result[
            "soh_percent"
        ]
    )


    battery_risk = (
        model3_result[
            "battery_risk"
        ]
    )


    battery_recommendations = (
        get_battery_recommendations(

            soh,

            battery_risk

        )
    )


    # ==================================================
    # COMBINED RECOMMENDATIONS
    # ==================================================

    recommendations = (
        get_combined_recommendations(

            driving_recommendations,

            charging_recommendations,

            battery_recommendations,

            stress,

            model2_prediction,

            battery_risk

        )
    )


    # ==================================================
    # SERVER LATENCY
    # ==================================================

    server_end = time.perf_counter()


    server_latency_ms = (

        server_end
        - server_start

    ) * 1000


    # ==================================================
    # PREDICTION ID
    # ==================================================

    prediction_counter += 1


    # ==================================================
    # RESPONSE
    # ==================================================

    response = {

        "driver_id":
            driver_id,

        "stress":
            stress,

        "stress_status":
            stress_status,

        "model_1_latency_ms":
            round(
                model1_latency_ms,
                3
            ),

        "charging":
            charging,

        "model_2_transaction_id":
            transaction_id,

        "model_2_prediction":
            model2_prediction,

        "model_2_probability":
            round(
                model2_probability,
                6
            ),

        "model_2_status":
            model2_status,

        "model_2_latency_ms":
            round(
                model2_latency_ms,
                3
            ),

        "soh":
            round(
                soh,
                2
            ),

        "battery_risk":
            battery_risk,

        "cap":
            model3_result[
                "cap"
            ],

        "cap_corr":
            model3_result[
                "cap_corr"
            ],

        "model_3_latency_ms":
            round(
                model3_latency_ms,
                3
            ),

        "recommendations":
            recommendations,

        "prediction_id":
            prediction_counter,

        "server_latency_ms":
            round(
                server_latency_ms,
                3
            )

    }


    latest_result = response


    return response


# ==================================================
# LATEST RESULT
#
# React dashboard continuously reads this endpoint.
# ==================================================

@app.get("/latest")
def get_latest():

    if latest_result is None:

        return {

            "status":
                "waiting",

            "message":
                "Waiting for ESP32 data..."

        }


    return latest_result


# ==================================================
# HEALTH CHECK
# ==================================================

@app.get("/")
def root():

    return {

        "project":
            "SparkEV",

        "status":
            "running",

        "model_1":
            "loaded",

        "model_2":
            "loaded",

        "model_3":
            "loaded",

        "edge_endpoint":
            "/esp32/result",

        "dashboard_endpoint":
            "/latest"

    }