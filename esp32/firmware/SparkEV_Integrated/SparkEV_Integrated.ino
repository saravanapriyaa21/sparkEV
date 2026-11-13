#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SH110X.h>

#include <WiFi.h>
#include <HTTPClient.h>

#include "model1_esp32_generated.h"
#include "model2_esp32_generated.h"
#include "model2_driver_vectors.h"
#include "model4_esp32_generated.h"
#include "model4_driver_vectors.h"

// On-device federated learning (all four modules) and the
// Module 3 federated SOH forecasting model
#include "fl_esp32_generated.h"
#include "fl_runtime.h"
#include "fl_tasks.h"


// Model 2 inputs of one driver (declared before any function, so the
// Arduino-generated prototypes can use it)
struct Model2Input
{
    const float* charge_features;
    const float* urgency_features;
    int location;
    float battery_kwh;
    const float* head_w;
    const float* head_b;
};


// Outcome of one on-device federated learning round of one model
struct FLResult
{
    bool trained;
    bool online;            // global model downloaded and update accepted
    int samples;
    float loss_before;      // car data, model at the start of the round
    float loss_local;       // after local training
    float loss_global;      // after installing the new global model
    float download_ms;      // GET global model
    float train_ms;         // local training on the ESP32
    float upload_ms;        // POST update + server aggregation + new global model back
    int bytes_up;
    int bytes_down;
    int server_round;
    int participants;
    uint32_t scratch_bytes; // RAM used by training (gradients, Adam moments, inputs)
};


// ============================================================
// WIFI CONFIGURATION
// ============================================================

// WIFI_SSID, WIFI_PASSWORD and SERVER_URL live in secrets.h
// (not committed). Copy secrets.example.h to secrets.h to set them.
#include "secrets.h"


// ============================================================
// DRIVER SELECTION
// ============================================================
//

#define TEST_DRIVER 17


// ============================================================
// BENCHMARK MODE
// ============================================================
//
// 0 = normal firmware (default, unchanged behaviour)
// 1 = run the IoT benchmark once at the end of setup(),
//     print the results on Serial, then continue with loop()
// 2 = weak Wi-Fi run: only the 100-send communication test
//

#define BENCHMARK_MODE 0


#if BENCHMARK_MODE
// Benchmark types are declared here, before any function, because
// the Arduino build adds function prototypes above the first function.
struct BenchStats
{
    float mean;
    float std;
    float min;
    float max;
};

struct BenchSend
{
    int code;
    uint32_t rtt_ms;
    int32_t rssi_dbm;
    int payload_bytes;
    long prediction_id;
};
#endif


// ============================================================
// OLED
// ============================================================

#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET -1
#define OLED_ADDRESS 0x3C

Adafruit_SH1106G display(
    SCREEN_WIDTH,
    SCREEN_HEIGHT,
    &Wire,
    OLED_RESET
);


// ============================================================
// MODEL 1 - EXACT VERIFIED TEST VECTORS
// ============================================================

static const float MODEL1_DRIVER17_VECTOR[MODEL1_NUM_FEATURES] = {

    68.0f,
    21.0f,
    118.0f,

    0.42f,
    1.15f,
    3.20f,
    -3.00f,

    1.35f,

    8.0f,
    6.0f,

    8.0f,
    18.0f,

    72.0f,
    29.0f,

    1.0f,
    0.0f,
    0.0f,
    0.0f,
    0.0f,

    0.0f,
    0.0f,
    1.0f,
    0.0f
};


static const float MODEL1_DRIVER24_VECTOR[MODEL1_NUM_FEATURES] = {

    45.0f,
    12.0f,
    82.0f,

    0.18f,
    0.55f,
    1.60f,
    -1.40f,

    0.65f,

    2.0f,
    1.0f,

    5.0f,
    24.0f,

    45.0f,
    27.0f,

    0.0f,
    1.0f,
    0.0f,
    0.0f,
    0.0f,

    0.0f,
    0.0f,
    1.0f,
    0.0f
};


const float* get_model1_input()
{
    if (TEST_DRIVER == 17)
        return MODEL1_DRIVER17_VECTOR;

    return MODEL1_DRIVER24_VECTOR;
}


// ============================================================
// MODEL 2 INPUT
// ============================================================
//
// Charge-level and urgency features of the driver's latest
// charging session, the charging site, the battery capacity
// and the driver's own FedPer urgency head.
//

Model2Input get_model2_input()
{
    if (TEST_DRIVER == 17)
        return {
            MODEL2_DRIVER17_CHARGE_INPUT,
            MODEL2_DRIVER17_URGENCY_INPUT,
            MODEL2_DRIVER17_LOCATION,
            MODEL2_DRIVER17_BATTERY_KWH,
            MODEL2_DRIVER17_HEAD_W,
            MODEL2_DRIVER17_HEAD_B
        };

    return {
        MODEL2_DRIVER24_CHARGE_INPUT,
        MODEL2_DRIVER24_URGENCY_INPUT,
        MODEL2_DRIVER24_LOCATION,
        MODEL2_DRIVER24_BATTERY_KWH,
        MODEL2_DRIVER24_HEAD_W,
        MODEL2_DRIVER24_HEAD_B
    };
}


// ============================================================
// MODEL 4 INPUT
// ============================================================

const float* get_model4_energy_input()
{
    if (TEST_DRIVER == 17)
        return MODEL4_DRIVER17_ENERGY_INPUT;

    return MODEL4_DRIVER24_ENERGY_INPUT;
}


const float* get_model4_arrival_input()
{
    if (TEST_DRIVER == 17)
        return MODEL4_DRIVER17_ARRIVAL_INPUT;

    return MODEL4_DRIVER24_ARRIVAL_INPUT;
}


// ============================================================
// MODEL 3 INPUT
// ============================================================
//
// Processed features of the battery's newest cycle (357 values,
// see notebooks/module3_battery_health/02_...) and its SOH now.
//

const float* get_model3_input()
{
    if (TEST_DRIVER == 17)
        return fl_m3_car17_now;

    return fl_m3_car24_now;
}


float get_model3_soh_now()
{
    if (TEST_DRIVER == 17)
        return FL_M3_CAR17_SOH_NOW;

    return FL_M3_CAR24_SOH_NOW;
}


// ============================================================
// RESULTS
// ============================================================

// ------------------------------------------------------------
// Model 1
// ------------------------------------------------------------

float model1_result =
    0.0f;

float model1_time_ms =
    0.0f;


// ------------------------------------------------------------
// Model 2
// ------------------------------------------------------------

const char* URGENCY_NAMES[3] = {
    "Time-critical",
    "Moderate",
    "Flexible"
};

// 85th-percentile battery need (%) until the next charge
float model2_need_pct =
    0.0f;

// Recommended charge level: need + 15 % reserve, 30..95 %
float model2_target_soc =
    0.0f;

int model2_urgency_class =
    0;

float model2_urgency_probs[3] = {

    0.0f,
    0.0f,
    0.0f

};

float model2_time_ms =
    0.0f;


// ------------------------------------------------------------
// Model 3
// ------------------------------------------------------------

// SOH now and forecast SOH 10 cycles ahead (fractions)
float model3_soh_now =
    0.0f;

float model3_soh_future =
    0.0f;

float model3_time_ms =
    0.0f;


// ------------------------------------------------------------
// Model 4
// ------------------------------------------------------------

float model4_energy_kwh_per_km =
    0.0f;

float model4_arrival_pct =
    0.0f;

float model4_time_ms =
    0.0f;


// ------------------------------------------------------------
// TOTAL ESP32 EDGE INFERENCE LATENCY
// ------------------------------------------------------------
//
// This is:
//
// Model 1 inference
// +
// Model 2 inference
// +
// Model 3 inference
// +
// Model 4 inference
//
// Measured entirely on ESP32.
//

float total_edge_inference_time_ms =
    0.0f;


// ============================================================
// WIFI STATUS
// ============================================================

bool wifi_connected =
    false;


// ============================================================
// MODEL 1 RUN
// ============================================================

void run_model1()
{
    uint32_t startMicros =
        micros();


    model1_result =
        model1_predict(
            get_model1_input()
        );


    uint32_t endMicros =
        micros();


    model1_time_ms =
        (endMicros - startMicros)
        / 1000.0f;
}


// ============================================================
// MODEL 2 RUN
// ============================================================

void run_model2()
{
    uint32_t startMicros =
        micros();


    Model2Input input =
        get_model2_input();


    model2_need_pct =
        model2_charge_need(
            input.charge_features,
            input.location
        );


    model2_target_soc =
        model2_need_pct + 15.0f;

    if (model2_target_soc < 30.0f)
        model2_target_soc = 30.0f;

    if (model2_target_soc > 95.0f)
        model2_target_soc = 95.0f;


    model2_urgency_class =
        model2_urgency(
            input.urgency_features,
            input.location,
            fl_m2u_head_w(input.head_w),
            fl_m2u_head_b(input.head_b),
            model2_urgency_probs
        );


    uint32_t endMicros =
        micros();


    model2_time_ms =
        (endMicros - startMicros)
        / 1000.0f;
}


// ============================================================
// MODEL 3 RUN
// ============================================================

void run_model3()
{
    uint32_t startMicros =
        micros();


    model3_soh_now =
        get_model3_soh_now();

    model3_soh_future =
        model3_soh_forecast(
            get_model3_input(),
            model3_soh_now
        );


    uint32_t endMicros =
        micros();


    model3_time_ms =
        (endMicros - startMicros)
        / 1000.0f;
}


// ============================================================
// MODEL 4 RUN
// ============================================================

void run_model4()
{
    uint32_t startMicros =
        micros();


    model4_energy_kwh_per_km =
        model4_energy(
            get_model4_energy_input()
        );


    model4_arrival_pct =
        model4_arrival_soc_pct(
            get_model4_arrival_input()
        );


    uint32_t endMicros =
        micros();


    model4_time_ms =
        (endMicros - startMicros)
        / 1000.0f;
}


// ============================================================
// CALCULATE TOTAL EDGE INFERENCE LATENCY
// ============================================================
//
// IMPORTANT:
// This contains ONLY ESP32 model inference time.
//
// It does NOT include:
// - Wi-Fi
// - HTTP
// - FastAPI
// - React
// - Recommendation processing
//

void calculate_total_edge_latency()
{
    total_edge_inference_time_ms =
        model1_time_ms
        +
        model2_time_ms
        +
        model3_time_ms
        +
        model4_time_ms;
}


// ============================================================
// MODEL 1 RISK LABEL
// ============================================================

const char* get_stress_label(
    float score
)
{
    if (score < 40.0f)
        return "LOW";


    if (score < 70.0f)
        return "MODERATE";


    return "HIGH";
}


// ============================================================
// MODEL 3 SOH RISK
// ============================================================

const char* get_soh_risk(
    float soh
)
{
    float sohPercent =
        soh * 100.0f;


    if (sohPercent >= 85.0f)
        return "LOW";


    if (sohPercent >= 80.0f)
        return "MODERATE";


    if (sohPercent >= 70.0f)
        return "HIGH";


    return "CRITICAL";
}


// ============================================================
// WIFI CONNECTION
// ============================================================

void connect_wifi()
{
    Serial.println();
    Serial.println(
        "========================================"
    );

    Serial.println(
        "Connecting to Wi-Fi..."
    );

    Serial.println(
        "========================================"
    );


    display.clearDisplay();

    display.setTextColor(
        SH110X_WHITE
    );

    display.setTextSize(1);


    display.setCursor(
        0,
        0
    );

    display.println(
        "SparkEV"
    );


    display.setCursor(
        0,
        12
    );

    display.println(
        "Connecting Wi-Fi..."
    );


    display.display();


    WiFi.mode(
        WIFI_STA
    );


    WiFi.begin(
        WIFI_SSID,
        WIFI_PASSWORD
    );


    int attempts =
        0;


    while (
        WiFi.status() != WL_CONNECTED
        &&
        attempts < 30
    )
    {
        delay(500);

        Serial.print(".");


        display.setCursor(
            0,
            24
        );

        display.print(
            "Attempt: "
        );

        display.println(
            attempts + 1
        );

        display.display();


        attempts++;
    }


    Serial.println();


    if (
        WiFi.status()
        == WL_CONNECTED
    )
    {
        wifi_connected =
            true;


        Serial.println(
            "Wi-Fi connected!"
        );


        Serial.print(
            "SSID: "
        );

        Serial.println(
            WiFi.SSID()
        );


        Serial.print(
            "ESP32 IP: "
        );

        Serial.println(
            WiFi.localIP()
        );


        Serial.print(
            "Signal: "
        );

        Serial.print(
            WiFi.RSSI()
        );

        Serial.println(
            " dBm"
        );


        display.clearDisplay();


        display.setCursor(
            0,
            0
        );

        display.println(
            "Wi-Fi Connected"
        );


        display.setCursor(
            0,
            12
        );

        display.print(
            "IP: "
        );

        display.println(
            WiFi.localIP()
        );


        display.display();


        delay(1500);
    }
    else
    {
        wifi_connected =
            false;


        Serial.println(
            "Wi-Fi connection FAILED"
        );


        display.clearDisplay();


        display.setCursor(
            0,
            0
        );

        display.println(
            "Wi-Fi FAILED"
        );


        display.setCursor(
            0,
            15
        );

        display.println(
            "Models still run"
        );


        display.setCursor(
            0,
            30
        );

        display.println(
            "Dashboard offline"
        );


        display.display();


        delay(1500);
    }
}


// ============================================================
// BUILD RESULT JSON
// ============================================================
//
// Payload sent to FastAPI. Shared by the normal send and the
// benchmark, so both send exactly the same bytes.
//

String build_results_json()
{
    // --------------------------------------------------------
    // Prepare values
    // --------------------------------------------------------

    float sohPercent =
        model3_soh_future
        * 100.0f;


    const char* stressStatus =
        get_stress_label(
            model1_result
        );


    const char* batteryRisk =
        get_soh_risk(
            model3_soh_future
        );


    // --------------------------------------------------------
    // JSON
    // --------------------------------------------------------

    String json = "{";


    json += "\"driver_id\":";
    json += String(
        TEST_DRIVER
    );


    // --------------------------------------------------------
    // Model 1
    // --------------------------------------------------------

    json += ",";

    json += "\"stress\":";
    json += String(
        model1_result,
        2
    );


    json += ",";

    json += "\"stress_status\":\"";
    json += stressStatus;
    json += "\"";


    json += ",";

    json += "\"model_1_latency_ms\":";
    json += String(
        model1_time_ms,
        3
    );


    // --------------------------------------------------------
    // Model 2
    // --------------------------------------------------------

    json += ",";

    json += "\"model_2_charge_need_pct\":";
    json += String(
        model2_need_pct,
        2
    );


    json += ",";

    json += "\"model_2_target_soc\":";
    json += String(
        model2_target_soc,
        1
    );


    json += ",";

    json += "\"model_2_urgency\":\"";
    json += URGENCY_NAMES[model2_urgency_class];
    json += "\"";


    json += ",";

    json += "\"model_2_urgency_class\":";
    json += String(
        model2_urgency_class
    );


    json += ",";

    json += "\"model_2_battery_kwh\":";
    json += String(
        get_model2_input().battery_kwh,
        2
    );


    json += ",";

    json += "\"model_2_latency_ms\":";
    json += String(
        model2_time_ms,
        3
    );


    // --------------------------------------------------------
    // Model 3
    // --------------------------------------------------------

    json += ",";

    json += "\"soh\":";
    json += String(
        sohPercent,
        2
    );


    json += ",";

    json += "\"battery_risk\":\"";
    json += batteryRisk;
    json += "\"";


    json += ",";

    json += "\"soh_now\":";
    json += String(
        model3_soh_now * 100.0f,
        2
    );


    json += ",";

    json += "\"model_3_latency_ms\":";
    json += String(
        model3_time_ms,
        3
    );


    // --------------------------------------------------------
    // Model 4
    // --------------------------------------------------------

    json += ",";

    json += "\"model_4_energy_kwh_per_km\":";
    json += String(
        model4_energy_kwh_per_km,
        4
    );


    json += ",";

    json += "\"model_4_arrival_soc_pct\":";
    json += String(
        model4_arrival_pct,
        1
    );


    json += ",";

    json += "\"model_4_latency_ms\":";
    json += String(
        model4_time_ms,
        3
    );


    // --------------------------------------------------------
    // TOTAL ESP32 EDGE LATENCY
    // --------------------------------------------------------

    json += ",";

    json += "\"edge_inference_latency_ms\":";
    json += String(
        total_edge_inference_time_ms,
        3
    );


    // --------------------------------------------------------
    // Recommendations
    //
    // FastAPI generates these from the
    // edge predictions.
    // --------------------------------------------------------

    json += ",";

    json += "\"recommendations\":[]";


    json += "}";


    return json;
}


// ============================================================
// SEND RESULTS TO FASTAPI
// ============================================================

void send_results_to_fastapi()
{
    // --------------------------------------------------------
    // Check Wi-Fi
    // --------------------------------------------------------

    if (
        WiFi.status()
        != WL_CONNECTED
    )
    {
        wifi_connected =
            false;


        Serial.println();
        Serial.println(
            "Wi-Fi disconnected."
        );


        Serial.println(
            "Attempting reconnect..."
        );


        connect_wifi();


        if (!wifi_connected)
        {
            Serial.println(
                "Could not reconnect."
            );

            return;
        }
    }


    wifi_connected =
        true;


    // --------------------------------------------------------
    // HTTP client
    // --------------------------------------------------------

    HTTPClient http;


    Serial.println();

    Serial.println(
        "Sending results to FastAPI..."
    );


    if (
        !http.begin(
            SERVER_URL
        )
    )
    {
        Serial.println(
            "HTTP connection failed."
        );

        return;
    }


    http.addHeader(
        "Content-Type",
        "application/json"
    );


    http.setTimeout(
        5000
    );


    String json =
        build_results_json();


    // --------------------------------------------------------
    // Print payload
    // --------------------------------------------------------

    Serial.println();

    Serial.println(
        "JSON payload:"
    );

    Serial.println(
        json
    );


    // --------------------------------------------------------
    // POST
    // --------------------------------------------------------

    int httpResponseCode =
        http.POST(
            json
        );


    Serial.print(
        "HTTP response code: "
    );

    Serial.println(
        httpResponseCode
    );


    if (
        httpResponseCode > 0
    )
    {
        String response =
            http.getString();


        Serial.print(
            "FastAPI response: "
        );

        Serial.println(
            response
        );
    }
    else
    {
        Serial.print(
            "HTTP POST failed: "
        );

        Serial.println(
            http.errorToString(
                httpResponseCode
            )
        );
    }


    http.end();


    Serial.println(
        "Result transmission complete."
    );
}


// ============================================================
// OLED - MODEL 1
// ============================================================

void display_model1()
{
    display.clearDisplay();

    display.setTextColor(
        SH110X_WHITE
    );

    display.setTextSize(1);


    display.setCursor(
        0,
        0
    );

    display.println(
        "SparkEV - Model 1"
    );


    display.setCursor(
        0,
        12
    );

    display.print(
        "Driver: "
    );

    display.println(
        TEST_DRIVER
    );


    display.setCursor(
        0,
        24
    );

    display.println(
        "Driving Stress"
    );


    display.setCursor(
        0,
        36
    );

    display.print(
        "Score: "
    );

    display.println(
        model1_result,
        2
    );


    display.setCursor(
        0,
        48
    );

    display.print(
        model1_time_ms,
        2
    );

    display.println(
        " ms"
    );


    display.display();
}


// ============================================================
// OLED - MODEL 2
// ============================================================

void display_model2()
{
    display.clearDisplay();

    display.setTextColor(
        SH110X_WHITE
    );

    display.setTextSize(1);


    display.setCursor(
        0,
        0
    );

    display.println(
        "SparkEV - Model 2"
    );


    display.setCursor(
        0,
        12
    );

    display.print(
        "Driver: "
    );

    display.println(
        TEST_DRIVER
    );


    display.setCursor(
        0,
        24
    );

    display.print(
        "Charge to: "
    );

    display.print(
        model2_target_soc,
        0
    );

    display.println(
        "%"
    );


    display.setCursor(
        0,
        36
    );

    display.println(
        URGENCY_NAMES[model2_urgency_class]
    );


    display.setCursor(
        0,
        48
    );

    display.print(
        "Time: "
    );

    display.print(
        model2_time_ms,
        2
    );

    display.println(
        " ms"
    );


    display.display();
}


// ============================================================
// OLED - MODEL 4
// ============================================================

void display_model4()
{
    display.clearDisplay();

    display.setTextColor(
        SH110X_WHITE
    );

    display.setTextSize(1);


    display.setCursor(
        0,
        0
    );

    display.println(
        "SparkEV - Model 4"
    );


    display.setCursor(
        0,
        12
    );

    display.print(
        "Driver: "
    );

    display.println(
        TEST_DRIVER
    );


    display.setCursor(
        0,
        24
    );

    display.print(
        "Energy: "
    );

    display.print(
        model4_energy_kwh_per_km,
        3
    );

    display.println(
        " kWh/km"
    );


    display.setCursor(
        0,
        36
    );

    display.print(
        "Arrival SOC: "
    );

    display.print(
        model4_arrival_pct,
        0
    );

    display.println(
        "%"
    );


    display.setCursor(
        0,
        48
    );

    display.print(
        "Time: "
    );

    display.print(
        model4_time_ms,
        3
    );

    display.println(
        " ms"
    );


    display.display();
}


// ============================================================
// OLED - MODEL 3
// ============================================================

void display_model3()
{
    display.clearDisplay();

    display.setTextColor(
        SH110X_WHITE
    );

    display.setTextSize(1);


    display.setCursor(
        0,
        0
    );

    display.println(
        "SparkEV - Model 3"
    );


    display.setCursor(
        0,
        12
    );

    display.print(
        "Driver: "
    );

    display.println(
        TEST_DRIVER
    );


    display.setCursor(
        0,
        24
    );

    display.print(
        "SOH now: "
    );

    display.print(
        model3_soh_now * 100.0f,
        1
    );

    display.println(
        "%"
    );


    display.setCursor(
        0,
        34
    );

    display.print(
        "SOH +10cyc: "
    );

    display.print(
        model3_soh_future
        * 100.0f,
        1
    );

    display.println(
        "%"
    );


    display.setCursor(
        0,
        44
    );

    display.print(
        "Risk: "
    );

    display.println(
        get_soh_risk(
            model3_soh_future
        )
    );


    display.setCursor(
        0,
        55
    );

    display.print(
        model3_time_ms,
        2
    );

    display.println(
        " ms"
    );


    display.display();
}


// ============================================================
// OLED - EDGE LATENCY
// ============================================================

void display_edge_latency()
{
    display.clearDisplay();

    display.setTextColor(
        SH110X_WHITE
    );

    display.setTextSize(1);


    display.setCursor(
        0,
        0
    );

    display.println(
        "ESP32 Edge AI"
    );


    display.setCursor(
        0,
        11
    );

    display.print(
        "M1: "
    );

    display.print(
        model1_time_ms,
        2
    );

    display.println(
        " ms"
    );


    display.setCursor(
        0,
        21
    );

    display.print(
        "M2: "
    );

    display.print(
        model2_time_ms,
        2
    );

    display.println(
        " ms"
    );


    display.setCursor(
        0,
        31
    );

    display.print(
        "M3: "
    );

    display.print(
        model3_time_ms,
        2
    );

    display.println(
        " ms"
    );


    display.setCursor(
        0,
        41
    );

    display.print(
        "M4: "
    );

    display.print(
        model4_time_ms,
        2
    );

    display.println(
        " ms"
    );


    display.setCursor(
        0,
        53
    );

    display.print(
        "TOTAL: "
    );

    display.print(
        total_edge_inference_time_ms,
        2
    );

    display.println(
        " ms"
    );


    display.display();
}


// ============================================================
// OLED - WIFI / TRANSMISSION
// ============================================================

void display_transmission_status(
    bool success
)
{
    display.clearDisplay();

    display.setTextColor(
        SH110X_WHITE
    );

    display.setTextSize(1);


    display.setCursor(
        0,
        0
    );

    display.println(
        "SparkEV Edge AI"
    );


    display.setCursor(
        0,
        14
    );

    display.print(
        "Driver: "
    );

    display.println(
        TEST_DRIVER
    );


    display.setCursor(
        0,
        28
    );

    if (success)
    {
        display.println(
            "Data Sent"
        );
    }
    else
    {
        display.println(
            "Send Failed"
        );
    }


    display.setCursor(
        0,
        42
    );

    if (wifi_connected)
    {
        display.println(
            "Wi-Fi Connected"
        );
    }
    else
    {
        display.println(
            "Wi-Fi Offline"
        );
    }


    display.setCursor(
        0,
        55
    );

    display.println(
        "FastAPI -> Dashboard"
    );


    display.display();
}


// ============================================================
// SERIAL SUMMARY
// ============================================================

void print_results()
{
    Serial.println();

    Serial.println(
        "========================================"
    );

    Serial.println(
        "        SparkEV INTEGRATED EDGE AI"
    );

    Serial.println(
        "========================================"
    );


    Serial.print(
        "Driver: "
    );

    Serial.println(
        TEST_DRIVER
    );


    Serial.println();


    // ========================================================
    // MODEL 1
    // ========================================================

    Serial.println(
        "MODEL 1 - DRIVING"
    );


    Serial.print(
        "Stress Score : "
    );

    Serial.println(
        model1_result,
        4
    );


    Serial.print(
        "Risk         : "
    );

    Serial.println(
        get_stress_label(
            model1_result
        )
    );


    Serial.print(
        "Inference    : "
    );

    Serial.print(
        model1_time_ms,
        3
    );

    Serial.println(
        " ms"
    );


    Serial.println();


    // ========================================================
    // MODEL 2
    // ========================================================

    Serial.println(
        "MODEL 2 - CHARGING RECOMMENDATION"
    );

    Serial.print(
        "Charge need  : "
    );

    Serial.print(
        model2_need_pct,
        4
    );

    Serial.println(
        "%"
    );

    Serial.print(
        "Target SOC   : "
    );

    Serial.print(
        model2_target_soc,
        1
    );

    Serial.println(
        "%"
    );

    Serial.print(
        "Urgency      : "
    );

    Serial.println(
        URGENCY_NAMES[model2_urgency_class]
    );

    Serial.print(
        "Inference    : "
    );

    Serial.print(
        model2_time_ms,
        3
    );

    Serial.println(
        " ms"
    );

    Serial.println();


    // ========================================================
    // MODEL 3
    // ========================================================

    Serial.println(
        "MODEL 3 - BATTERY"
    );


    Serial.print(
        "SOH now      : "
    );

    Serial.print(
        model3_soh_now * 100.0f,
        4
    );

    Serial.println(
        "%"
    );


    Serial.print(
        "SOH +10 cyc. : "
    );

    Serial.print(
        model3_soh_future
        * 100.0f,
        4
    );

    Serial.println(
        "%"
    );


    Serial.print(
        "Risk         : "
    );

    Serial.println(
        get_soh_risk(
            model3_soh_future
        )
    );


    Serial.print(
        "Inference    : "
    );

    Serial.print(
        model3_time_ms,
        3
    );

    Serial.println(
        " ms"
    );


    Serial.println();


    // ========================================================
    // MODEL 4
    // ========================================================

    Serial.println(
        "MODEL 4 - ENERGY"
    );

    Serial.print(
        "Energy       : "
    );

    Serial.print(
        model4_energy_kwh_per_km,
        4
    );

    Serial.println(
        " kWh/km"
    );

    Serial.print(
        "Arrival SOC  : "
    );

    Serial.print(
        model4_arrival_pct,
        2
    );

    Serial.println(
        "%"
    );

    Serial.print(
        "Inference    : "
    );

    Serial.print(
        model4_time_ms,
        3
    );

    Serial.println(
        " ms"
    );

    Serial.println();


    // ========================================================
    // TOTAL EDGE INFERENCE
    // ========================================================

    Serial.println(
        "========================================"
    );

    Serial.println(
        "ESP32 EDGE INFERENCE PERFORMANCE"
    );

    Serial.println(
        "========================================"
    );


    Serial.print(
        "Model 1      : "
    );

    Serial.print(
        model1_time_ms,
        3
    );

    Serial.println(
        " ms"
    );


    Serial.print(
        "Model 2      : "
    );

    Serial.print(
        model2_time_ms,
        3
    );

    Serial.println(
        " ms"
    );


    Serial.print(
        "Model 3      : "
    );

    Serial.print(
        model3_time_ms,
        3
    );

    Serial.println(
        " ms"
    );


    Serial.print(
        "Model 4      : "
    );

    Serial.print(
        model4_time_ms,
        3
    );

    Serial.println(
        " ms"
    );


    Serial.println(
        "----------------------------------------"
    );


    Serial.print(
        "TOTAL EDGE   : "
    );

    Serial.print(
        total_edge_inference_time_ms,
        3
    );

    Serial.println(
        " ms"
    );


    Serial.println(
        "========================================"
    );


    // ========================================================
    // SYSTEM
    // ========================================================

    Serial.print(
        "Wi-Fi        : "
    );


    if (wifi_connected)
    {
        Serial.println(
            "Connected"
        );
    }
    else
    {
        Serial.println(
            "Disconnected"
        );
    }


    if (wifi_connected)
    {
        Serial.print(
            "ESP32 IP     : "
        );

        Serial.println(
            WiFi.localIP()
        );
    }


    Serial.print(
        "CPU          : "
    );

    Serial.print(
        ESP.getCpuFreqMHz()
    );

    Serial.println(
        " MHz"
    );


    Serial.print(
        "Free Heap    : "
    );

    Serial.print(
        ESP.getFreeHeap()
    );

    Serial.println(
        " bytes"
    );


    Serial.println(
        "========================================"
    );
}


// ============================================================
// ON-DEVICE FEDERATED LEARNING
// ============================================================
//
// While the car is parked or charging it takes part in one
// federated round for each model (all four modules):
//
//   1. download the current global model      GET  /fl/global/<task>
//   2. train it on the car's own data         (fl_runtime.h)
//   3. upload only the weights                POST /fl/update/<task>
//      -> the server aggregates them with the other cars and
//         returns the new global model, which is installed
//   4. report losses, times and sizes         POST /fl/report
//
// Raw data never leaves the car. FedPer heads stay on the car.
// The Module 2, 3 and 4 predictions then use the updated models.
// Without Wi-Fi the car still trains locally and keeps its model.
//

#define FL_EVERY_N_CYCLES 2
#define FL_HTTP_TIMEOUT_MS 10000

FLResult fl_results[FL_NUM_TASKS];

int fl_session_count =
    0;

float fl_session_ms =
    0.0f;

uint32_t fl_free_heap_min =
    0;

int fl_loop_cycles =
    0;

static const char* FL_HEADER_KEYS[2] = {"X-FL-Round", "X-FL-Participants"};


// http://host:port/esp32/result -> http://host:port
String fl_server_base()
{
    String url = SERVER_URL;
    int slash = url.indexOf('/', 8);
    return slash > 0 ? url.substring(0, slash) : url;
}


// Reads exactly n bytes of the response body
bool fl_read_body(HTTPClient& http, uint8_t* dst, int n)
{
    if (http.getSize() != n)
        return false;

    WiFiClient* stream = http.getStreamPtr();
    int got = 0;
    uint32_t start = millis();

    while (got < n && millis() - start < FL_HTTP_TIMEOUT_MS) {
        int avail = stream->available();
        if (avail > 0)
            got += stream->readBytes(dst + got, min(avail, n - got));
        else
            delay(1);
    }
    return got == n;
}


// Sends (when body != nullptr) and receives one global model; on success
// the received weights replace the task's exchanged weights.
bool fl_exchange(int t, const char* path, const uint8_t* body, FLResult& r, float& ms)
{
    FLTask& k = fl_task[t];
    int nbytes = k.exchange * (int)sizeof(float);
    uint8_t* incoming = (uint8_t*)malloc(nbytes);
    if (!incoming)
        return false;

    HTTPClient http;
    String url = fl_server_base() + path;
    uint32_t start = micros();
    bool ok = false;

    if (http.begin(url)) {
        http.setTimeout(FL_HTTP_TIMEOUT_MS);
        http.collectHeaders(FL_HEADER_KEYS, 2);
        int code;
        if (body) {
            http.addHeader("Content-Type", "application/octet-stream");
            code = http.POST((uint8_t*)body, nbytes);
            r.bytes_up += nbytes;
        } else {
            code = http.GET();
        }
        ok = code == 200 && fl_read_body(http, incoming, nbytes);
        if (ok) {
            r.bytes_down += nbytes;
            r.server_round = http.header("X-FL-Round").toInt();
            if (body)
                r.participants = http.header("X-FL-Participants").toInt();
            memcpy(k.block, incoming, nbytes);
        }
        http.end();
    }
    ms = (micros() - start) / 1000.0f;
    free(incoming);
    return ok;
}


// One federated round of one model
void fl_round(int t, bool online)
{
    FLResult& r = fl_results[t];
    r = FLResult();

    if (!fl_init_task(t, TEST_DRIVER)) {
        Serial.printf("FL %s: not enough RAM for the model\n", FL_TASK_IDS[t]);
        return;
    }
    r.samples = fl_task_samples(t, TEST_DRIVER);

    // 1. current global model
    if (online)
        r.online = fl_exchange(t, (String("/fl/global/") + FL_TASK_IDS[t]).c_str(), nullptr, r, r.download_ms);

    // 2. local training on the car's data
    size_t input_bytes = 0, train_bytes = 0;
    if (!fl_prepare_inputs(t, TEST_DRIVER, &input_bytes)) {
        Serial.printf("FL %s: not enough RAM for the inputs\n", FL_TASK_IDS[t]);
        return;
    }
    r.loss_before = fl_task_loss(t, TEST_DRIVER);

    uint32_t seed = 1000u * TEST_DRIVER + t + 100000u * (fl_session_count - 1);
    uint32_t start = micros();
    r.trained = fl_task_train(t, TEST_DRIVER, seed, &train_bytes);
    r.train_ms = (micros() - start) / 1000.0f;
    r.scratch_bytes = input_bytes + train_bytes;
    r.loss_local = fl_task_loss(t, TEST_DRIVER);
    r.loss_global = r.loss_local;

    // 3. upload the update, install the new global model
    if (r.online && r.trained) {
        String path = String("/fl/update/") + FL_TASK_IDS[t] +
                      "?client=car" + TEST_DRIVER + "&n=" + r.samples;
        r.online = fl_exchange(t, path.c_str(), (const uint8_t*)fl_task[t].block, r, r.upload_ms);
        if (r.online)
            r.loss_global = fl_task_loss(t, TEST_DRIVER);
    }
    fl_release_inputs(t);

    // predictions use the updated model from now on
    fl_activate(t);
}


void send_fl_report()
{
    if (WiFi.status() != WL_CONNECTED)
        return;

    String json = "{\"driver_id\":" + String(TEST_DRIVER) +
                  ",\"session\":" + String(fl_session_count) +
                  ",\"session_ms\":" + String(fl_session_ms, 1) +
                  ",\"free_heap_min\":" + String(fl_free_heap_min) +
                  ",\"tasks\":[";
    for (int t = 0; t < FL_NUM_TASKS; t++) {
        FLResult& r = fl_results[t];
        if (t) json += ",";
        json += "{\"task\":\"" + String(FL_TASK_IDS[t]) + "\"";
        json += ",\"trained\":" + String(r.trained ? "true" : "false");
        json += ",\"online\":" + String(r.online ? "true" : "false");
        json += ",\"samples\":" + String(r.samples);
        json += ",\"loss_before\":" + String(r.loss_before, 9);
        json += ",\"loss_local\":" + String(r.loss_local, 9);
        json += ",\"loss_global\":" + String(r.loss_global, 9);
        json += ",\"download_ms\":" + String(r.download_ms, 1);
        json += ",\"train_ms\":" + String(r.train_ms, 2);
        json += ",\"upload_ms\":" + String(r.upload_ms, 1);
        json += ",\"bytes_up\":" + String(r.bytes_up);
        json += ",\"bytes_down\":" + String(r.bytes_down);
        json += ",\"server_round\":" + String(r.server_round);
        json += ",\"participants\":" + String(r.participants);
        json += ",\"scratch_bytes\":" + String(r.scratch_bytes);
        json += "}";
    }
    json += "]}";

    HTTPClient http;
    if (!http.begin(fl_server_base() + "/fl/report"))
        return;
    http.addHeader("Content-Type", "application/json");
    http.setTimeout(5000);
    int code = http.POST(json);
    http.end();
    Serial.printf("FL report sent: HTTP %d\n", code);
}


// One federated learning session: all five models, then new predictions
void run_fl_session()
{
    fl_session_count++;

    Serial.println();
    Serial.println("============================================================");
    Serial.printf("FEDERATED LEARNING SESSION %d (car parked / charging)\n", fl_session_count);
    Serial.println("============================================================");

    display.clearDisplay();
    display.setTextColor(SH110X_WHITE);
    display.setTextSize(1);
    display.setCursor(0, 0);
    display.println("Federated learning");
    display.setCursor(0, 14);
    display.println("Car parked: training");
    display.setCursor(0, 26);
    display.println("on local data...");
    display.display();

    if (WiFi.status() != WL_CONNECTED)
        connect_wifi();
    bool online = WiFi.status() == WL_CONNECTED;

    uint32_t start = millis();
    for (int t = 0; t < FL_NUM_TASKS; t++) {
        fl_round(t, online);
        FLResult& r = fl_results[t];
        Serial.printf("%-11s n=%3d  loss %.4e -> local %.4e -> global %.4e  train %7.2f ms  "
                      "down %6.1f ms  up+agg %6.1f ms  %5d B up  round %d (%d cars)  %s\n",
                      FL_TASK_NAMES[t], r.samples, r.loss_before, r.loss_local, r.loss_global,
                      r.train_ms, r.download_ms, r.upload_ms, r.bytes_up, r.server_round,
                      r.participants, r.online ? "online" : "offline");
    }
    fl_session_ms = millis() - start;
    fl_free_heap_min = ESP.getMinFreeHeap();
    Serial.printf("Session: %.0f ms, lowest free heap %u B, free heap now %u B\n",
                  fl_session_ms, fl_free_heap_min, ESP.getFreeHeap());

    // new predictions with the updated models
    run_model2();
    run_model3();
    run_model4();
    calculate_total_edge_latency();

    send_fl_report();
}


void display_fl_session()
{
    display.clearDisplay();
    display.setTextColor(SH110X_WHITE);
    display.setTextSize(1);
    display.setCursor(0, 0);
    display.printf("Fed. learning #%d", fl_session_count);

    for (int t = 0; t < FL_NUM_TASKS; t++) {
        FLResult& r = fl_results[t];
        display.setCursor(0, 11 + 10 * t);
        display.printf("%-10s%5.0fms %s", FL_TASK_NAMES[t], r.train_ms,
                       r.online ? "OK" : (r.trained ? "loc" : "--"));
    }
    display.display();
}


// ============================================================
// IOT BENCHMARK (BENCHMARK_MODE 1 or 2)
// ============================================================
//
// Prints clearly marked blocks on Serial (115200 baud):
//
//   BEGIN_SUMMARY_JSON ... END_SUMMARY_JSON
//   BEGIN_SENDS_CSV    ... END_SENDS_CSV
//   BEGIN_PARITY_CSV   ... END_PARITY_CSV   (mode 1 only)
//   BEGIN_CYCLE_JSON   ... END_CYCLE_JSON   (mode 1 only, printed
//                                            from loop() after
//                                            3 normal cycles)
//
// Send status codes: HTTP status (> 0), HTTPClient error (< 0),
// -100 = Wi-Fi not connected, -101 = http.begin() failed.
//

#if BENCHMARK_MODE

#define BENCH_RUNS 100
#define BENCH_WARMUP 5
#define BENCH_SENDS 100
#define BENCH_SEND_INTERVAL_MS 1000
#define BENCH_CYCLES 3

float bench_times_ms[5][BENCH_RUNS];   // m1, m2, m3, m4, full pass
float bench_full_pass_mean_ms = 0.0f;
BenchSend bench_sends[BENCH_SENDS];


BenchStats bench_stats(const float* values, int n)
{
    BenchStats s = {0.0f, 0.0f, values[0], values[0]};

    double sum = 0.0;
    for (int i = 0; i < n; i++)
    {
        sum += values[i];
        if (values[i] < s.min) s.min = values[i];
        if (values[i] > s.max) s.max = values[i];
    }
    s.mean = sum / n;

    double sq = 0.0;
    for (int i = 0; i < n; i++)
    {
        sq += (values[i] - s.mean) * (values[i] - s.mean);
    }
    s.std = n > 1 ? sqrt(sq / (n - 1)) : 0.0f;

    return s;
}


// Nearest-rank percentile of n values (n <= BENCH_SENDS)
float bench_percentile(const float* values, int n, float pct)
{
    if (n == 0) return NAN;

    float sorted[BENCH_SENDS];
    for (int i = 0; i < n; i++) sorted[i] = values[i];

    for (int i = 1; i < n; i++)
    {
        float key = sorted[i];
        int j = i - 1;
        while (j >= 0 && sorted[j] > key)
        {
            sorted[j + 1] = sorted[j];
            j--;
        }
        sorted[j + 1] = key;
    }

    int rank = (int)ceil(pct / 100.0f * n);
    if (rank < 1) rank = 1;
    return sorted[rank - 1];
}


void bench_print_stats(const char* name, BenchStats s, bool last)
{
    Serial.printf(
        "    \"%s\": {\"mean\": %.4f, \"std\": %.4f, \"min\": %.4f, \"max\": %.4f}%s\n",
        name, s.mean, s.std, s.min, s.max, last ? "" : ","
    );
}


// ------------------------------------------------------------
// 1. Inference latency (100 timed runs after warm-up)
// ------------------------------------------------------------

void bench_inference()
{
    Serial.println("Benchmark: inference warm-up...");

    for (int i = 0; i < BENCH_WARMUP; i++)
    {
        run_model1();
        run_model2();
        run_model3();
        run_model4();
    }

    Serial.println("Benchmark: timing 100 inference runs...");

    for (int r = 0; r < BENCH_RUNS; r++)
    {
        run_model1();
        bench_times_ms[0][r] = model1_time_ms;

        run_model2();
        bench_times_ms[1][r] = model2_time_ms;

        run_model3();
        bench_times_ms[2][r] = model3_time_ms;

        run_model4();
        bench_times_ms[3][r] = model4_time_ms;

        uint32_t start = micros();
        run_model1();
        run_model2();
        run_model3();
        run_model4();
        bench_times_ms[4][r] = (micros() - start) / 1000.0f;
    }

    calculate_total_edge_latency();

    bench_full_pass_mean_ms =
        bench_stats(bench_times_ms[4], BENCH_RUNS).mean;
}


// ------------------------------------------------------------
// 8. On-device outputs for drivers 17 and 24 (parity check)
// ------------------------------------------------------------

void bench_parity()
{
    

    Serial.println();
    Serial.println("BEGIN_PARITY_CSV");
    Serial.println("driver,model,output,value");

    Serial.printf("17,m1,stress,%.6f\n", model1_predict(MODEL1_DRIVER17_VECTOR));
    Serial.printf("24,m1,stress,%.6f\n", model1_predict(MODEL1_DRIVER24_VECTOR));

    float m2p[3];
    int m2u;

    Serial.printf("17,m2,charge_need,%.6f\n",
                  model2_charge_need(MODEL2_DRIVER17_CHARGE_INPUT, MODEL2_DRIVER17_LOCATION));
    m2u = model2_urgency(MODEL2_DRIVER17_URGENCY_INPUT, MODEL2_DRIVER17_LOCATION,
                         MODEL2_DRIVER17_HEAD_W, MODEL2_DRIVER17_HEAD_B, m2p);
    Serial.printf("17,m2,urgency_class,%d\n", m2u);
    Serial.printf("17,m2,p_time_critical,%.6f\n", m2p[0]);

    Serial.printf("24,m2,charge_need,%.6f\n",
                  model2_charge_need(MODEL2_DRIVER24_CHARGE_INPUT, MODEL2_DRIVER24_LOCATION));
    m2u = model2_urgency(MODEL2_DRIVER24_URGENCY_INPUT, MODEL2_DRIVER24_LOCATION,
                         MODEL2_DRIVER24_HEAD_W, MODEL2_DRIVER24_HEAD_B, m2p);
    Serial.printf("24,m2,urgency_class,%d\n", m2u);
    Serial.printf("24,m2,p_time_critical,%.6f\n", m2p[0]);

    Serial.printf("17,m4,energy,%.6f\n", model4_energy(MODEL4_DRIVER17_ENERGY_INPUT));
    Serial.printf("17,m4,arrival_soc_pct,%.6f\n", model4_arrival_soc_pct(MODEL4_DRIVER17_ARRIVAL_INPUT));
    Serial.printf("24,m4,energy,%.6f\n", model4_energy(MODEL4_DRIVER24_ENERGY_INPUT));
    Serial.printf("24,m4,arrival_soc_pct,%.6f\n", model4_arrival_soc_pct(MODEL4_DRIVER24_ARRIVAL_INPUT));

    Serial.printf("17,m3,soh_forecast,%.6f\n", model3_soh_forecast(fl_m3_car17_now, FL_M3_CAR17_SOH_NOW));
    Serial.printf("24,m3,soh_forecast,%.6f\n", model3_soh_forecast(fl_m3_car24_now, FL_M3_CAR24_SOH_NOW));

    Serial.println("END_PARITY_CSV");
}


// ------------------------------------------------------------
// 9. On-device federated training: time and RAM per model
//    (FL_BENCH_RUNS runs from the deployed weights, no network;
//    the first run's losses are checked against Python)
// ------------------------------------------------------------

#define FL_BENCH_RUNS 20

BenchStats fl_bench_stats[FL_NUM_TASKS];
float fl_bench_median[FL_NUM_TASKS];
float fl_bench_p95[FL_NUM_TASKS];
float fl_bench_prepare_ms[FL_NUM_TASKS];
float fl_bench_loss[FL_NUM_TASKS][2];
uint32_t fl_bench_scratch[FL_NUM_TASKS];

void bench_fl_training()
{
    static float times[FL_BENCH_RUNS];

    for (int t = 0; t < FL_NUM_TASKS; t++) {
        for (int r = 0; r < FL_BENCH_RUNS; r++) {
            fl_reset_all();
            fl_init_task(t, TEST_DRIVER);

            size_t in_bytes = 0, train_bytes = 0;
            uint32_t p0 = micros();
            fl_prepare_inputs(t, TEST_DRIVER, &in_bytes);
            float prep_ms = (micros() - p0) / 1000.0f;

            if (r == 0) fl_bench_loss[t][0] = fl_task_loss(t, TEST_DRIVER);
            uint32_t start = micros();
            fl_task_train(t, TEST_DRIVER, 1000u * TEST_DRIVER + t, &train_bytes);
            times[r] = (micros() - start) / 1000.0f;
            if (r == 0) {
                fl_bench_loss[t][1] = fl_task_loss(t, TEST_DRIVER);
                fl_bench_prepare_ms[t] = prep_ms;
                fl_bench_scratch[t] = in_bytes + train_bytes;
            }
            fl_release_inputs(t);
        }
        fl_bench_stats[t] = bench_stats(times, FL_BENCH_RUNS);
        fl_bench_median[t] = bench_percentile(times, FL_BENCH_RUNS, 50.0f);
        fl_bench_p95[t] = bench_percentile(times, FL_BENCH_RUNS, 95.0f);
    }
    fl_reset_all();

    Serial.println();
    Serial.println("BEGIN_FL_CSV");
    Serial.println("driver,task,samples,exchange_bytes,scratch_bytes,prepare_ms,train_mean_ms,train_median_ms,"
                   "train_p95_ms,train_min_ms,train_max_ms,loss_before,loss_after");
    for (int t = 0; t < FL_NUM_TASKS; t++) {
        BenchStats& b = fl_bench_stats[t];
        fl_init_task(t, TEST_DRIVER);
        int exch = fl_task[t].exchange * 4;
        Serial.printf("%d,%s,%d,%d,%lu,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.8e,%.8e\n", TEST_DRIVER,
                      FL_TASK_IDS[t], fl_task_samples(t, TEST_DRIVER), exch,
                      (unsigned long)fl_bench_scratch[t], fl_bench_prepare_ms[t], b.mean, fl_bench_median[t],
                      fl_bench_p95[t], b.min, b.max, fl_bench_loss[t][0], fl_bench_loss[t][1]);
    }
    fl_reset_all();
    Serial.println("END_FL_CSV");
}


// ------------------------------------------------------------
// 5/6. Communication: 100 sends, one per second
// ------------------------------------------------------------

void bench_sends_run()
{
    String payload = build_results_json();

    Serial.printf("Benchmark: %d sends to %s (one per second)...\n", BENCH_SENDS, SERVER_URL);

    for (int i = 0; i < BENCH_SENDS; i++)
    {
        uint32_t slot_start = millis();
        BenchSend& s = bench_sends[i];

        s.payload_bytes = payload.length();
        s.rtt_ms = 0;
        s.prediction_id = -1;

        if (WiFi.status() != WL_CONNECTED)
        {
            connect_wifi();
        }

        if (WiFi.status() != WL_CONNECTED)
        {
            s.code = -100;
            s.rssi_dbm = 0;
        }
        else
        {
            s.rssi_dbm = WiFi.RSSI();

            HTTPClient http;

            if (!http.begin(SERVER_URL))
            {
                s.code = -101;
            }
            else
            {
                http.addHeader("Content-Type", "application/json");
                http.setTimeout(5000);

                uint32_t t0 = millis();
                s.code = http.POST(payload);
                s.rtt_ms = millis() - t0;

                if (s.code > 0)
                {
                    String response = http.getString();
                    int at = response.indexOf("\"prediction_id\":");
                    if (at >= 0)
                    {
                        s.prediction_id = response.substring(at + 16).toInt();
                    }
                }

                http.end();
            }
        }

        Serial.print(s.code == 200 ? "." : "x");
        if ((i + 1) % 50 == 0) Serial.println();

        uint32_t elapsed = millis() - slot_start;
        if (elapsed < BENCH_SEND_INTERVAL_MS)
        {
            delay(BENCH_SEND_INTERVAL_MS - elapsed);
        }
    }
}


void bench_print_sends_summary(const char* name, bool last)
{
    float rtt_ok[BENCH_SENDS];
    int ok = 0;
    long rssi_sum = 0;
    int rssi_n = 0;

    for (int i = 0; i < BENCH_SENDS; i++)
    {
        if (bench_sends[i].code == 200) rtt_ok[ok++] = bench_sends[i].rtt_ms;
        if (bench_sends[i].code != -100)
        {
            rssi_sum += bench_sends[i].rssi_dbm;
            rssi_n++;
        }
    }

    float rtt_mean = 0.0f;
    float rtt_max = 0.0f;
    for (int i = 0; i < ok; i++)
    {
        rtt_mean += rtt_ok[i];
        if (rtt_ok[i] > rtt_max) rtt_max = rtt_ok[i];
    }
    if (ok > 0) rtt_mean /= ok;

    float success_pct = 100.0f * ok / BENCH_SENDS;

    Serial.printf("  \"%s\": {\"sends\": %d, \"success_pct\": %.1f, \"loss_pct\": %.1f, ",
                  name, BENCH_SENDS, success_pct, 100.0f - success_pct);
    Serial.printf("\"rtt_ms_mean\": %.1f, \"rtt_ms_median\": %.1f, \"rtt_ms_p95\": %.1f, \"rtt_ms_max\": %.1f, ",
                  rtt_mean, bench_percentile(rtt_ok, ok, 50.0f),
                  bench_percentile(rtt_ok, ok, 95.0f), rtt_max);
    Serial.printf("\"rssi_dbm_mean\": %.1f, \"payload_bytes\": %d}%s\n",
                  rssi_n ? (float)rssi_sum / rssi_n : NAN,
                  bench_sends[0].payload_bytes, last ? "" : ",");
}


void bench_print_sends_csv(const char* condition)
{
    Serial.println();
    Serial.println("BEGIN_SENDS_CSV");
    Serial.println("condition,index,status_code,rtt_ms,rssi_dbm,payload_bytes,prediction_id");

    for (int i = 0; i < BENCH_SENDS; i++)
    {
        const BenchSend& s = bench_sends[i];
        Serial.printf("%s,%d,%d,%lu,%ld,%d,%ld\n", condition, i + 1, s.code,
                      (unsigned long)s.rtt_ms, (long)s.rssi_dbm, s.payload_bytes, s.prediction_id);
    }

    Serial.println("END_SENDS_CSV");
}


void bench_print_device_info()
{
    Serial.printf("  \"board\": \"FILL IN: board name and module as printed on it\",\n");
    Serial.printf("  \"chip_model\": \"%s\", \"chip_revision\": %d, \"chip_cores\": %d,\n",
                  ESP.getChipModel(), ESP.getChipRevision(), ESP.getChipCores());
    Serial.printf("  \"cpu_mhz\": %lu, \"flash_chip_bytes\": %lu, \"sdk\": \"%s\",\n",
                  (unsigned long)ESP.getCpuFreqMHz(), (unsigned long)ESP.getFlashChipSize(),
                  ESP.getSdkVersion());
    Serial.printf("  \"test_driver\": %d,\n", TEST_DRIVER);
}


void run_benchmark()
{
    Serial.println();
    Serial.println("==================== SPARKEV IOT BENCHMARK ====================");

#if BENCHMARK_MODE == 1

    uint32_t bench_start_ms = millis();
    float temp_start = temperatureRead();
    uint32_t heap_before = ESP.getFreeHeap();

    bench_inference();
    bench_parity();
    bench_fl_training();
    bench_sends_run();

    uint32_t heap_after = ESP.getFreeHeap();
    float temp_end = temperatureRead();
    uint32_t bench_duration_ms = millis() - bench_start_ms;

    Serial.println();
    Serial.println("BEGIN_SUMMARY_JSON");
    Serial.println("{");
    bench_print_device_info();
    Serial.printf("  \"runs\": %d, \"warmup_runs\": %d,\n", BENCH_RUNS, BENCH_WARMUP);
    Serial.println("  \"inference_ms\": {");
    bench_print_stats("m1", bench_stats(bench_times_ms[0], BENCH_RUNS), false);
    bench_print_stats("m2", bench_stats(bench_times_ms[1], BENCH_RUNS), false);
    bench_print_stats("m3", bench_stats(bench_times_ms[2], BENCH_RUNS), false);
    bench_print_stats("m4", bench_stats(bench_times_ms[3], BENCH_RUNS), false);
    bench_print_stats("full_pass", bench_stats(bench_times_ms[4], BENCH_RUNS), true);
    Serial.println("  },");
    Serial.printf("  \"sketch_bytes\": %lu, \"free_sketch_bytes\": %lu,\n",
                  (unsigned long)ESP.getSketchSize(), (unsigned long)ESP.getFreeSketchSpace());
    Serial.printf("  \"heap_size\": %lu, \"free_heap_before\": %lu, \"free_heap_after\": %lu, \"min_free_heap\": %lu,\n",
                  (unsigned long)ESP.getHeapSize(), (unsigned long)heap_before,
                  (unsigned long)heap_after, (unsigned long)ESP.getMinFreeHeap());
    Serial.printf("  \"chip_temp_c_start\": %.2f, \"chip_temp_c_end\": %.2f, \"benchmark_duration_s\": %.1f,\n",
                  temp_start, temp_end, bench_duration_ms / 1000.0f);
    bench_print_sends_summary("normal_wifi", true);
    Serial.println("}");
    Serial.println("END_SUMMARY_JSON");

    bench_print_sends_csv("normal_wifi");

    Serial.println();
    Serial.println("Benchmark done. Cycle length follows after 3 normal display cycles.");

#else

    bench_sends_run();

    Serial.println();
    Serial.println("BEGIN_SUMMARY_JSON");
    Serial.println("{");
    bench_print_device_info();
    bench_print_sends_summary("weak_wifi", true);
    Serial.println("}");
    Serial.println("END_SUMMARY_JSON");

    bench_print_sends_csv("weak_wifi");

#endif

    Serial.println("================== END SPARKEV IOT BENCHMARK ==================");
}


#if BENCHMARK_MODE == 1

// ------------------------------------------------------------
// 4. Real cycle length, measured between loop() starts
// ------------------------------------------------------------

void bench_record_cycle()
{
    static uint32_t last_start = 0;
    static uint32_t cycle_ms[BENCH_CYCLES];
    static int cycles = 0;

    uint32_t now = millis();

    if (last_start != 0 && cycles < BENCH_CYCLES)
    {
        cycle_ms[cycles++] = now - last_start;

        if (cycles == BENCH_CYCLES)
        {
            float mean = 0.0f;
            for (int i = 0; i < BENCH_CYCLES; i++) mean += cycle_ms[i];
            mean /= BENCH_CYCLES;

            Serial.println();
            Serial.println("BEGIN_CYCLE_JSON");
            Serial.printf("{\"cycles_measured\": %d, \"cycle_ms\": %.1f, \"cycle_ms_values\": [%lu, %lu, %lu], ",
                          BENCH_CYCLES, mean, (unsigned long)cycle_ms[0],
                          (unsigned long)cycle_ms[1], (unsigned long)cycle_ms[2]);
            Serial.printf("\"full_pass_mean_ms\": %.4f, \"cpu_busy_share_pct\": %.4f}\n",
                          bench_full_pass_mean_ms, 100.0f * bench_full_pass_mean_ms / mean);
            Serial.println("END_CYCLE_JSON");
        }
    }

    last_start = now;
}

#endif

#endif


// ============================================================
// SETUP
// ============================================================

void setup()
{
    Serial.begin(
        115200
    );


    delay(
        1000
    );


    // --------------------------------------------------------
    // OLED
    // --------------------------------------------------------

    Wire.begin(
        21,
        22
    );


    if (
        !display.begin(
            OLED_ADDRESS,
            true
        )
    )
    {
        Serial.println(
            "OLED initialization failed"
        );


        while (true)
        {
            delay(
                1000
            );
        }
    }


    // --------------------------------------------------------
    // Startup screen
    // --------------------------------------------------------

    display.clearDisplay();

    display.setTextColor(
        SH110X_WHITE
    );

    display.setTextSize(1);


    display.setCursor(
        0,
        0
    );

    display.println(
        "SparkEV"
    );


    display.setCursor(
        0,
        12
    );

    display.println(
        "Integrated Edge AI"
    );


    display.setCursor(
        0,
        24
    );

    display.print(
        "Driver: "
    );

    display.println(
        TEST_DRIVER
    );


    display.setCursor(
        0,
        36
    );

    display.println(
        "Loading 4 Models..."
    );


    display.display();


    delay(
        2000
    );


    // --------------------------------------------------------
    // Wi-Fi
    // --------------------------------------------------------

    connect_wifi();


    // --------------------------------------------------------
    // Run all four models
    // --------------------------------------------------------

    run_model1();

    run_model2();

    run_model3();

    run_model4();


    // --------------------------------------------------------
    // Calculate total ESP32 inference latency
    // --------------------------------------------------------

    calculate_total_edge_latency();


    // --------------------------------------------------------
    // Print results
    // --------------------------------------------------------

    print_results();


    // --------------------------------------------------------
    // Show edge latency on OLED
    // --------------------------------------------------------

    display_edge_latency();

    delay(
        2500
    );


    // --------------------------------------------------------
    // Send results to FastAPI
    // --------------------------------------------------------

    send_results_to_fastapi();


#if BENCHMARK_MODE
    // --------------------------------------------------------
    // IoT benchmark (runs once, then normal loop continues)
    // --------------------------------------------------------

    run_benchmark();
#endif
}


// ============================================================
// LOOP
// ============================================================

void loop()
{
#if BENCHMARK_MODE == 1
    // --------------------------------------------------------
    // Benchmark: measure the real cycle length
    // --------------------------------------------------------

    bench_record_cycle();
#endif

    // --------------------------------------------------------
    // Model 1
    // --------------------------------------------------------

    display_model1();

    delay(
        2500
    );


    // --------------------------------------------------------
    // Model 2
    // --------------------------------------------------------

    display_model2();

    delay(
        2500
    );


    // --------------------------------------------------------
    // Model 3
    // --------------------------------------------------------

    display_model3();

    delay(
        2500
    );


    // --------------------------------------------------------
    // Model 4
    // --------------------------------------------------------

    display_model4();

    delay(
        2500
    );


    // --------------------------------------------------------
    // Edge inference performance
    // --------------------------------------------------------

    display_edge_latency();

    delay(
        2500
    );


    // --------------------------------------------------------
    // Send latest results
    // --------------------------------------------------------

    send_results_to_fastapi();

    display_transmission_status(
        wifi_connected
    );

    delay(
        2000
    );


    // --------------------------------------------------------
    // Parked / charging: on-device federated learning
    // --------------------------------------------------------

    fl_loop_cycles++;

    #if BENCHMARK_MODE == 1
    bool fl_due = fl_loop_cycles > BENCH_CYCLES && fl_loop_cycles % FL_EVERY_N_CYCLES == 0;
#else
    bool fl_due = fl_loop_cycles % FL_EVERY_N_CYCLES == 0;
#endif

    if (fl_due)
    {
        run_fl_session();

        display_fl_session();

        delay(
            3000
        );
    }
}