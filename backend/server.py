from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import time
import os
import pandas as pd
import xgboost as xgb

import json

from model2_charging_recommendation import (
    URGENCY_NAMES,
    predict_charge_need,
    predict_urgency,
    recommend_charging
)

from model2_charging_recommendation import STATIONS

from whatif import run_whatif

from model3_soh_federated import (
    run_model3,
    DRIVER_BATTERIES as MODEL3_DRIVER_BATTERIES
)
import federated


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
# MODEL 2 - CHARGING RECOMMENDATION
#
# The ESP32 runs the charge-level and urgency
# networks. The server always runs the site
# congestion model and the station ranking
# (model2_charging_recommendation.py).
# ==================================================

print("Model 2 loaded through model2_charging_recommendation.py")


# ==================================================
# MODEL 2 DEMO USERS
#
# One real charging user per demo driver, chosen by
# esp32/tools/export_model2_demo_users.py.
# Used only by OLD /predict testing endpoint.
# ==================================================

MODEL2_DEMO_USERS = {
    int(driver): user
    for driver, user in json.load(open(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "esp32", "tools", "vectors", "model2_demo_users.json"
    ))).items()
}


# ==================================================
# MODEL 4 - ENERGY CONSUMPTION AND ARRIVAL SOC
#
# Used only by OLD /predict testing endpoint.
# Final ESP32 architecture uses ESP32 Model 4.
# ==================================================

from model4_energy import run_model4, demo_trips as MODEL4_DEMO_TRIPS
from model2_charging_recommendation import charge_level_model
from explain import explain_all


# ==================================================
# MODEL 3 - FEDERATED SOH FORECASTING
#
# Used by the /predict testing endpoint. The ESP32 runs
# the same federated model on the car. Demo batteries:
# MODEL3_DRIVER_BATTERIES (model3_soh_federated.py).
# ==================================================


# ==================================================
# OLD REQUEST FORMAT
#
# Used only by /predict.
# ==================================================

class DriverRequest(BaseModel):

    driver_id: int


# ==================================================
# WHAT-IF SIMULATOR REQUEST
#
# The driver's alternative decision. Factors are
# relative to the driver's current behaviour
# (1.0 = unchanged).
# ==================================================

class WhatIfRequest(BaseModel):

    station: str

    target_soc: float

    power_kw: float | None = None

    driving_style: float = 1.0

    speed: float = 1.0

    climate: float = 1.0


# ==================================================
# ESP32 EDGE RESULT FORMAT
#
# ESP32 performs the ML INFERENCES of all four
# modules.
#
# FastAPI receives:
#
#   Model 1 prediction
#   Model 2 charge level and urgency
#   Model 3 prediction
#   Model 4 energy and arrival SOC
#
# AND their actual ESP32-side latency.
#
# FastAPI does NOT run the edge models here.
#
# FastAPI only:
#
#   1. Receives edge predictions
#   2. Ranks the charging stations (Model 2:
#      site congestion + decision layer)
#   3. Generates recommendations
#   4. Stores result
#   5. Serves /latest
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
    #
    # charge_need_pct: 85th-percentile battery need
    # until the next charge (%); target_soc adds the
    # 15 % reserve.
    # ----------------------------------------------

    model_2_charge_need_pct: float

    model_2_target_soc: float

    model_2_urgency: str

    model_2_urgency_class: int

    model_2_battery_kwh: float

    model_2_latency_ms: float


    # ----------------------------------------------
    # Model 3
    # ----------------------------------------------

    soh: float

    soh_now: float

    battery_risk: str

    model_3_latency_ms: float


    # ----------------------------------------------
    # Model 4
    # ----------------------------------------------

    model_4_energy_kwh_per_km: float

    model_4_arrival_soc_pct: float

    model_4_latency_ms: float


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
    # + Model 4
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
# MODEL 2 RECOMMENDATIONS
#
# recommendation: output of recommend_charging()
# (target SOC, ranked stations, message).
# ==================================================

def get_charging_recommendations(
    recommendation
):

    recommendations = []

    stations = recommendation["stations"]


    if not stations:

        recommendations.append(
            recommendation["message"]
        )

        return recommendations


    best = stations[0]

    recommendations.append(
        f"Charge to {recommendation['target_soc']:.0f}% at "
        f"{best['name']}. It takes about {best['total_min']:.0f} min "
        f"in total, including the drive there and any waiting."
    )


    # ----------------------------------------------
    # Why not the nearest station?
    # ----------------------------------------------

    nearest = min(
        stations,
        key=lambda s: s["distance_km"]
    )

    if nearest["name"] != best["name"]:

        recommendations.append(
            f"{nearest['name']} is closer, but you may have to "
            f"wait about {nearest['wait_min']:.0f} min there, "
            f"so {best['name']} is faster overall."
        )


    return recommendations


# ==================================================
# SHAP-BASED RECOMMENDATIONS
#
# Every recommendation comes from the SHAP explanation
# of the driver's own prediction (explain.py): a factor
# becomes advice only if the driver can change it and
# SHAP shows that it makes the prediction worse now.
# The charging station choice comes from the Module 2
# decision layer (its time breakdown is its explanation).
# ==================================================

def explain_driver(driver_id, charging_recommendation, soh, soh_now, battery_risk):

    demo_user = MODEL2_DEMO_USERS[driver_id]

    m2_x = charge_level_model.transform(
        demo_user["charge_level_features"],
        demo_user["location"]
    )

    return explain_all(
        driver_id,
        driver_features[driver_id],
        m2_x,
        demo_user["charge_level_features"],
        charging_recommendation["target_soc"],
        soh_now,
        soh,
        battery_risk,
        MODEL4_DEMO_TRIPS[str(driver_id)]["energy_features"]
    )


def build_recommendations(explanations, charging_recommendation,
                          energy_kwh_per_km, arrival_soc_pct):
    """Returns the insights (module, text, kind) and the same texts as a list.
    kind: "action" = SHAP shows a controllable factor making things worse,
    "info" = explanation or decision without a needed change."""

    insights = []

    # topic: customer-facing label shown on the dashboard
    topics = {"Module 1": "Driving", "Module 2": "Charging",
              "Module 3": "Battery", "Module 4": "Trip"}

    def add(module, items):
        for item in items:
            insights.append(dict(module=module, topic=topics[module], **item))

    # Module 1: driving (SHAP of the stress prediction)
    add("Module 1", explanations["stress"]["advice"])

    # Module 2: station and charge level (decision layer)
    # and why this charge level (SHAP of the charge need)
    add("Module 2", [dict(text=t, kind="info")
                     for t in get_charging_recommendations(charging_recommendation)])
    add("Module 2", explanations["charging"]["advice"])

    # Module 3: battery health (SHAP of the SOH forecast)
    add("Module 3", explanations["battery"]["advice"])

    # Module 4: energy (SHAP of the energy prediction)
    add("Module 4", [dict(
        text=f"This trip should use about {1000 * energy_kwh_per_km:.0f} Wh/km. "
             f"You should arrive with about {arrival_soc_pct:.0f}% battery.",
        kind="info")])
    add("Module 4", explanations["energy"]["advice"])

    return insights, [i["text"] for i in insights]


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

    


    # ==================================================
    # MODEL 2 CHARGING RECOMMENDATION
    #
    # The ESP32 sends the predicted charge need and
    # urgency. The server adds the site congestion
    # forecast and ranks the stations. The charge on
    # arrival from Model 4 is the starting SOC.
    # ==================================================

    charging_recommendation = recommend_charging(

        current_soc=data.model_4_arrival_soc_pct,

        battery_kwh=data.model_2_battery_kwh,

        charge_need_pct=data.model_2_charge_need_pct,

        urgency_class=data.model_2_urgency_class

    )

    


    # ==================================================
    # MODEL 4 PERSONALIZED RECOMMENDATIONS
    # ==================================================

    


    # ==================================================
    # MODEL 3 PERSONALIZED RECOMMENDATIONS
    # ==================================================

    


    # ==================================================
    # COMBINED PERSONALIZED INSIGHTS
    # ==================================================

    explanations = explain_driver(
        data.driver_id,
        charging_recommendation,
        data.soh,
        data.soh_now,
        battery_risk
    )

    insights, recommendations = build_recommendations(
        explanations,
        charging_recommendation,
        data.model_4_energy_kwh_per_km,
        data.model_4_arrival_soc_pct
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

        "model_2_charge_need_pct":
            round(
                data.model_2_charge_need_pct,
                2
            ),

        "model_2_target_soc":
            round(
                data.model_2_target_soc,
                1
            ),

        "model_2_urgency":
            data.model_2_urgency,

        "model_2_urgency_class":
            data.model_2_urgency_class,

        "model_2_battery_kwh":
            data.model_2_battery_kwh,

        "model_2_latency_ms":
            round(
                data.model_2_latency_ms,
                3
            ),

        "charging_recommendation":
            charging_recommendation,


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

        "soh_now":
            round(
                data.soh_now,
                2
            ),

        "model_3_latency_ms":
            round(
                data.model_3_latency_ms,
                3
            ),


        # ----------------------------------------------
        # Model 4
        # ----------------------------------------------

        "model_4_energy_kwh_per_km":
            round(
                data.model_4_energy_kwh_per_km,
                4
            ),

        "model_4_arrival_soc_pct":
            round(
                data.model_4_arrival_soc_pct,
                1
            ),

        "model_4_latency_ms":
            round(
                data.model_4_latency_ms,
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

        "explanations":
            explanations,

        "insights":
            insights,


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
        charging_recommendation["message"]
    )


    print(
        "Urgency      :",
        data.model_2_urgency
    )


    print(
        "Energy       :",
        data.model_4_energy_kwh_per_km,
        "kWh/km, arrival SOC",
        data.model_4_arrival_soc_pct,
        "%"
    )


    print(
        "SOH (+10 cyc):",
        data.soh,
        "  now:",
        data.soh_now
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
        "Model 4      :",
        data.model_4_latency_ms,
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


    if driver_id not in MODEL3_DRIVER_BATTERIES:

        raise HTTPException(
            status_code=400,
            detail=(
                f"Model 3 data is not configured for "
                f"Driver {driver_id}."
            )
        )


    if driver_id not in MODEL2_DEMO_USERS:

        raise HTTPException(
            status_code=400,
            detail=(
                f"Model 2 charging user is not "
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


    


    # ==================================================
    # MODEL 4
    #
    # Runs before Model 2: the predicted charge on
    # arrival is the starting SOC of the charging
    # recommendation.
    # ==================================================

    model4_start = time.perf_counter()

    model4_result = run_model4(
        driver_id
    )

    model4_latency_ms = (
        time.perf_counter()
        - model4_start
    ) * 1000


    


    # ==================================================
    # MODEL 2
    # ==================================================

    demo_user = MODEL2_DEMO_USERS[
        driver_id
    ]


    model2_start = time.perf_counter()


    charge_need_pct = predict_charge_need(

        demo_user["charge_level_features"],

        demo_user["location"]

    )


    urgency_class, urgency_probs = predict_urgency(

        demo_user["urgency_features"],

        demo_user["location"],

        demo_user["user_id"]

    )


    model2_latency_ms = (
        time.perf_counter()
        - model2_start
    ) * 1000


    urgency = URGENCY_NAMES[
        urgency_class
    ]


    charging_recommendation = recommend_charging(

        current_soc=model4_result["arrival_soc_pct"],

        battery_kwh=demo_user["battery_capacity_kwh"],

        charge_need_pct=charge_need_pct,

        urgency_class=urgency_class

    )


    


    # ==================================================
    # MODEL 3
    # ==================================================

    model3_result = run_model3(
        driver_id
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


    


    # ==================================================
    # COMBINED RECOMMENDATIONS
    # ==================================================

    explanations = explain_driver(
        driver_id,
        charging_recommendation,
        soh,
        model3_result["soh_now_percent"],
        battery_risk
    )

    insights, recommendations = build_recommendations(
        explanations,
        charging_recommendation,
        model4_result["energy_kwh_per_km"],
        model4_result["arrival_soc_pct"]
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

        "model_2_user_id":
            demo_user["user_id"],

        "model_2_charge_need_pct":
            round(
                charge_need_pct,
                2
            ),

        "model_2_target_soc":
            charging_recommendation["target_soc"],

        "model_2_urgency":
            urgency,

        "model_2_urgency_class":
            urgency_class,

        "model_2_battery_kwh":
            demo_user["battery_capacity_kwh"],

        "model_2_urgency_probs":
            [round(p, 4) for p in urgency_probs],

        "model_2_latency_ms":
            round(
                model2_latency_ms,
                3
            ),

        "charging_recommendation":
            charging_recommendation,

        "soh":
            round(
                soh,
                2
            ),

        "battery_risk":
            battery_risk,

        "soh_now":
            model3_result[
                "soh_now_percent"
            ],

        "model_3_battery":
            model3_result[
                "battery"
            ],

        "model_3_latency_ms":
            round(
                model3_latency_ms,
                3
            ),

        "model_4_trip":
            model4_result["trip"],

        "model_4_energy_kwh_per_km":
            model4_result["energy_kwh_per_km"],

        "model_4_arrival_soc_pct":
            model4_result["arrival_soc_pct"],

        "model_4_latency_ms":
            round(
                model4_latency_ms,
                3
            ),

        "recommendations":
            recommendations,

        "explanations":
            explanations,

        "insights":
            insights,

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
# WHAT-IF SIMULATOR
#
# Compares the recommended charging option with the
# driver's own scenario, for the latest result.
# Runs on the server: the dashboard calls it
# whenever the driver changes a setting.
# ==================================================

@app.post("/whatif")
def what_if(request: WhatIfRequest):

    if latest_result is None:

        raise HTTPException(
            status_code=400,
            detail="No result yet. Send an ESP32 result or call /predict first."
        )


    station_names = [
        station["name"]
        for station in STATIONS
    ]

    if request.station not in station_names:

        raise HTTPException(
            status_code=400,
            detail=f"Unknown station. Available: {station_names}"
        )


    if not 30 <= request.target_soc <= 100:

        raise HTTPException(
            status_code=400,
            detail="target_soc must be between 30 and 100."
        )


    for name in ["driving_style", "speed", "climate"]:

        if not 0.5 <= getattr(request, name) <= 1.5:

            raise HTTPException(
                status_code=400,
                detail=f"{name} must be between 0.5 and 1.5."
            )


    recommendation = latest_result["charging_recommendation"]

    driver_id = latest_result["driver_id"]


    # ----------------------------------------------
    # Starting point: the recommended option. When no
    # charging is needed, the reference is "no charge"
    # at the nearest station.
    # ----------------------------------------------

    context = {

        "driver_id": driver_id,

        "current_soc": recommendation["current_soc"],

        "battery_kwh": latest_result["model_2_battery_kwh"],

        "recommended_station":
            recommendation["recommended_station"] or station_names[0],

        "target_soc":
            recommendation["target_soc"]
            if recommendation["recommended_station"]
            else recommendation["current_soc"],

    }


    return run_whatif(

        context,

        model1,

        FEATURE_NAMES,

        driver_features[driver_id],

        request.model_dump()

    )


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
# ON-DEVICE FEDERATED LEARNING
#
# The ESP32 (one real car) downloads the global model
# of each task, trains it on its own data and uploads
# only the weights. The server aggregates them with the
# simulated fleet and returns the new global model.
# Weights travel as little-endian float32 arrays.
# See federated.py.
# ==================================================

def get_fl_task(task):

    if task not in federated.TASKS:

        raise HTTPException(
            status_code=404,
            detail=(
                f"Unknown federated task '{task}'. "
                f"Available: {sorted(federated.TASKS)}"
            )
        )

    return federated.TASKS[task]


@app.get("/fl/global/{task}")
def fl_global(task: str):

    state = get_fl_task(task)

    return Response(
        content=state.payload(),
        media_type="application/octet-stream",
        headers={
            "X-FL-Round": str(state.round),
            "X-FL-Floats": str(state.n_floats),
        },
    )


@app.post("/fl/update/{task}")
async def fl_update(task: str, request: Request, client: str = "esp32", n: int = 1):

    state = get_fl_task(task)

    if n < 1:

        raise HTTPException(
            status_code=400,
            detail="n (number of local samples) must be at least 1."
        )

    body = await request.body()

    try:

        info = state.run_round(body, n, client)

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    print(
        f"FL {task} round {info['round']}: {client} + "
        f"{info['fleet_clients']} simulated cars, "
        f"server {info['server_ms']} ms"
    )

    return Response(
        content=state.payload(),
        media_type="application/octet-stream",
        headers={
            "X-FL-Round": str(info["round"]),
            "X-FL-Participants": str(info["participants"]),
            "X-FL-Server-Ms": str(info["server_ms"]),
        },
    )


@app.post("/fl/report")
def fl_report(report: dict):

    if not isinstance(report.get("tasks"), list):

        raise HTTPException(
            status_code=400,
            detail="The report needs a 'tasks' list."
        )

    report["received_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    federated.latest_fl_report = report

    return {"status": "received"}


@app.get("/fl/status")
def fl_status():

    return federated.fl_status()


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

        "model_4":
            "loaded",

        "federated_learning":
            sorted(federated.TASKS),

        "edge_endpoint":
            "/esp32/result",

        "dashboard_endpoint":
            "/latest"

    }