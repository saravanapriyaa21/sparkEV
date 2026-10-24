#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SH110X.h>

#include <WiFi.h>
#include <HTTPClient.h>

#include "model1_esp32_generated.h"
#include "model2_esp32_generated.h"
#include "model2_driver_vectors.h"
#include "model3_esp32_generated.h"
#include "model3_driver_vectors.h"


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

const float* get_model2_input()
{
    if (TEST_DRIVER == 17)
        return MODEL2_DRIVER17_VECTOR;

    return MODEL2_DRIVER24_VECTOR;
}


// ============================================================
// MODEL 3 INPUT
// ============================================================

const float* get_model3_input()
{
    if (TEST_DRIVER == 17)
        return MODEL3_DRIVER17_VECTOR;

    return MODEL3_DRIVER24_VECTOR;
}


// ============================================================
// MODEL 3 INFERENCE
// ============================================================

static inline void model3_predict_all(
    const float* features,
    float* predictions
)
{
    predictions[0] =
        MODEL3_BASE_SCORES[0];

    predictions[1] =
        MODEL3_BASE_SCORES[1];

    predictions[2] =
        MODEL3_BASE_SCORES[2];


    for (
        int tree_id = 0;
        tree_id < MODEL3_NUM_TREES;
        tree_id++
    )
    {
        int32_t node_index =
            MODEL3_TREE_ROOTS[tree_id];


        while (true)
        {
            const Model3Node& node =
                MODEL3_NODES[node_index];


            // ------------------------------------------------
            // Leaf node
            // ------------------------------------------------

            if (
                node.yes_child < 0 &&
                node.no_child < 0
            )
            {
                uint8_t target =
                    MODEL3_TREE_TARGET[tree_id];

                predictions[target] +=
                    node.leaf;

                break;
            }


            float value =
                features[node.feature];


            // ------------------------------------------------
            // XGBoost missing-value routing
            // ------------------------------------------------

            if (isnan(value))
            {
                node_index =
                    node.missing_child;
            }
            else if (value < node.threshold)
            {
                node_index =
                    node.yes_child;
            }
            else
            {
                node_index =
                    node.no_child;
            }
        }
    }
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

float model2_probability =
    0.0f;

int model2_prediction =
    0;

float model2_time_ms =
    0.0f;


// ------------------------------------------------------------
// Model 3
// ------------------------------------------------------------

float model3_predictions[3] = {

    0.0f,
    0.0f,
    0.0f

};

float model3_time_ms =
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


    model2_probability =
        model2_predict(
            get_model2_input()
        );


    model2_prediction =
        model2_probability >= 0.5f
        ? 1
        : 0;


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


    model3_predict_all(
        get_model3_input(),
        model3_predictions
    );


    uint32_t endMicros =
        micros();


    model3_time_ms =
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
        model3_time_ms;
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


    // --------------------------------------------------------
    // Prepare values
    // --------------------------------------------------------

    float sohPercent =
        model3_predictions[2]
        * 100.0f;


    const char* stressStatus =
        get_stress_label(
            model1_result
        );


    const char* batteryRisk =
        get_soh_risk(
            model3_predictions[2]
        );


    const char* chargingStatus;


    if (
        model2_prediction == 1
    )
    {
        chargingStatus =
            "Faulty";
    }
    else
    {
        chargingStatus =
            "Healthy";
    }


    const char* model2Status;


    if (
        model2_prediction == 1
    )
    {
        model2Status =
            "Abnormal charging session";
    }
    else
    {
        model2Status =
            "Normal charging session";
    }


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

    json += "\"charging\":\"";
    json += chargingStatus;
    json += "\"";


    json += ",";

    json += "\"model_2_prediction\":";
    json += String(
        model2_prediction
    );


    json += ",";

    json += "\"model_2_probability\":";
    json += String(
        model2_probability,
        6
    );


    json += ",";

    json += "\"model_2_status\":\"";
    json += model2Status;
    json += "\"";


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

    json += "\"cap\":";
    json += String(
        model3_predictions[0],
        4
    );


    json += ",";

    json += "\"cap_corr\":";
    json += String(
        model3_predictions[1],
        4
    );


    json += ",";

    json += "\"model_3_latency_ms\":";
    json += String(
        model3_time_ms,
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
    // FastAPI generates these from the three
    // edge predictions.
    // --------------------------------------------------------

    json += ",";

    json += "\"recommendations\":[]";


    json += "}";


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


    if (
        model2_prediction == 1
    )
    {
        display.println(
            "Charging Fault"
        );
    }
    else
    {
        display.println(
            "Charging Healthy"
        );
    }


    display.setCursor(
        0,
        36
    );

    display.print(
        "Prob: "
    );

    display.println(
        model2_probability,
        3
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
        "CAP: "
    );

    display.println(
        model3_predictions[0],
        2
    );


    display.setCursor(
        0,
        34
    );

    display.print(
        "SOH: "
    );

    display.print(
        model3_predictions[2]
        * 100.0f,
        2
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
            model3_predictions[2]
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
        14
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
        26
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
        38
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
        52
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
        "MODEL 2 - CHARGING"
    );


    Serial.print(
        "Probability  : "
    );

    Serial.println(
        model2_probability,
        6
    );


    Serial.print(
        "Status       : "
    );


    if (
        model2_prediction == 1
    )
    {
        Serial.println(
            "Faulty"
        );
    }
    else
    {
        Serial.println(
            "Healthy"
        );
    }


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
        "CAP          : "
    );

    Serial.println(
        model3_predictions[0],
        4
    );


    Serial.print(
        "CAP_CORR     : "
    );

    Serial.println(
        model3_predictions[1],
        4
    );


    Serial.print(
        "SOH          : "
    );

    Serial.print(
        model3_predictions[2]
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
            model3_predictions[2]
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
        "Loading 3 Models..."
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
    // Run all three models
    // --------------------------------------------------------

    run_model1();

    run_model2();

    run_model3();


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
}


// ============================================================
// LOOP
// ============================================================

void loop()
{
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
}