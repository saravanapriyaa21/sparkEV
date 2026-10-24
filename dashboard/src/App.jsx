import { useEffect, useState } from "react";
import { AlertTriangle, CheckCircle } from "lucide-react";

import "./App.css";


// ==================================================
// APP
// ==================================================

function App() {

  const [driver, setDriver] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");


  // ==================================================
  // GET LATEST ESP32 RESULT
  //
  // ESP32
  //    ↓
  // FastAPI /esp32/result
  //    ↓
  // latest_result
  //    ↓
  // Dashboard /latest
  // ==================================================

  useEffect(() => {

    const fetchLatestData = async () => {

      try {

        const response = await fetch(
          "http://localhost:8000/latest"
        );


        if (!response.ok) {

          throw new Error(
            `Server returned ${response.status}`
          );

        }


        const data = await response.json();


        console.log(
          "Latest FastAPI data:",
          data
        );


        // ------------------------------------------
        // WAITING FOR ESP32
        // ------------------------------------------

        if (data.status === "waiting") {

          setDriver(null);
          setLoading(false);
          setError("");

          return;

        }


        // ------------------------------------------
        // ESP32 DATA RECEIVED
        // ------------------------------------------

        setDriver(data);
        setLoading(false);
        setError("");

      }


      catch (err) {

        console.error(
          "FastAPI error:",
          err
        );


        setError(
          "Unable to connect to FastAPI. Make sure the server is running."
        );


        setLoading(false);

      }

    };


    // Initial request

    fetchLatestData();


    // Poll for new ESP32 data

    const interval = setInterval(
      fetchLatestData,
      1000
    );


    // Cleanup

    return () => {

      clearInterval(interval);

    };

  }, []);


  // ==================================================
  // WAITING SCREEN
  // ==================================================

  if (loading || !driver) {

    return (

      <div className="app">

        <header className="header">

          <div className="brand">

            <h1>
              SparkEV
            </h1>

            <p>
              Intelligent EV Health &amp; Behaviour Analytics
            </p>

          </div>

        </header>


        <main>

          <section className="panel">

            <div className="panel-header">

              <div>

                <h3>
                  Waiting for ESP32 data
                </h3>

                <p>
                  Start the ESP32 to send driver telemetry
                  and generate personalized insights.
                </p>

              </div>

            </div>

          </section>

        </main>

      </div>

    );

  }


  // ==================================================
  // BATTERY STATUS
  //
  // SOH comes from MODEL 3
  // ==================================================

  const batteryRisk =
    driver.battery_risk || "Unknown";


  const batteryHealthy =
    batteryRisk === "Low";


  // ==================================================
  // MAIN DASHBOARD
  // ==================================================

  return (

    <div className="app">


      {/* ==================================================
          HEADER
      ================================================== */}

      <header className="header">

        <div className="brand">

          <h1>
            SparkEV
          </h1>

          <p>
            Intelligent EV Health &amp; Behaviour Analytics
          </p>

        </div>


        <div className="driver-selector">

          <span>
            Driver
          </span>

          <strong>
            Driver {driver.driver_id}
          </strong>

        </div>

      </header>


      {/* ==================================================
          CONNECTION ERROR
      ================================================== */}

      {error && (

        <main>

          <section className="panel">

            <div className="panel-header">

              <div>

                <h3>
                  Connection Error
                </h3>

                <p>
                  {error}
                </p>

              </div>

            </div>

          </section>

        </main>

      )}


      {/* ==================================================
          DASHBOARD
      ================================================== */}

      {driver && !error && (

        <main>


          {/* ==================================================
              PRIMARY RESULTS
          ================================================== */}

          <section className="stats-grid">


            {/* ==================================================
                MODEL 3 - BATTERY HEALTH
                SOH IS MODEL 3 OUTPUT
            ================================================== */}

            <div className="stat-card">

              <div className="stat-content">

                <p>
                  Battery Health
                </p>


                <h2>
                  {driver.soh}%
                </h2>


                <span
                  className={`status ${
                    batteryHealthy
                      ? "healthy"
                      : "attention"
                  }`}
                >

                  {batteryRisk === "Low"
                    ? "Healthy"
                    : `${batteryRisk} Risk`}

                </span>


                <small>
                  Model 3 · Battery Health
                </small>

              </div>

            </div>


            {/* ==================================================
                MODEL 1 - DRIVING BEHAVIOUR
            ================================================== */}

            <div className="stat-card">

              <div className="stat-content">

                <p>
                  Driving Stress
                </p>


                <h2>

                  {driver.stress}

                  <span className="metric-suffix">
                    /100
                  </span>

                </h2>


                <span
                  className={`status ${
                    driver.stress <= 50
                      ? "healthy"
                      : "moderate"
                  }`}
                >

                  {driver.stress_status}

                </span>


                <small>
                  Model 1 · Driving Behaviour
                </small>

              </div>

            </div>


            {/* ==================================================
                MODEL 2 - CHARGING BEHAVIOUR
            ================================================== */}

            <div className="stat-card">

              <div className="stat-content">

                <p>
                  Charging Behaviour
                </p>


                <h2>
                  {driver.charging}
                </h2>


                <span
                  className={`status ${
                    driver.charging === "Healthy"
                      ? "healthy"
                      : "attention"
                  }`}
                >

                  {driver.charging === "Healthy"
                    ? "Within recommended range"
                    : "Needs attention"}

                </span>


                <small>
                  Model 2 · Charging Behaviour
                </small>

              </div>

            </div>

          </section>


          {/* ==================================================
              PERSONALIZED INSIGHTS
          ================================================== */}

          <section className="panel">

            <div className="panel-header">

              <div>

                <h3>
                  Personalized Insights
                </h3>

                <p>
                  Recommendations generated from the
                  driver's behaviour, charging and battery data
                </p>

              </div>

            </div>


            <div className="recommendation-list">

              {driver.recommendations?.map(
                (recommendation, index) => {

                  const text =
                    recommendation.toLowerCase();


                  const isWarning =
                    text.includes("frequent") ||
                    text.includes("above 90%") ||
                    text.includes("hard acceleration") ||
                    text.includes("hard braking") ||
                    text.includes("attention") ||
                    text.includes("high battery stress") ||
                    text.includes("aggressiveness") ||
                    text.includes("abnormal") ||
                    text.includes("avoid prolonged");


                  return (

                    <div
                      className={`recommendation recommendation--${
                        isWarning
                          ? "warning"
                          : "success"
                      }`}
                      key={index}
                    >

                      {isWarning ? (

                        <AlertTriangle
                          size={18}
                          strokeWidth={2.25}
                        />

                      ) : (

                        <CheckCircle
                          size={18}
                          strokeWidth={2.25}
                        />

                      )}


                      <span>
                        {recommendation}
                      </span>

                    </div>

                  );

                }
              )}

            </div>

          </section>


          {/* ==================================================
              AI INFERENCE PERFORMANCE
          ================================================== */}

          <section className="panel">

            <div className="panel-header">

              <div>

                <h3>
                  AI Inference Performance
                </h3>

                <p>
                  Processing latency measured across the
                  ESP32 edge models and FastAPI backend
                </p>

              </div>

            </div>


            <div className="models-grid">


              {/* ==================================================
                  MODEL 1
              ================================================== */}

              <div className="model-card">

                <div>

                  <strong>
                    Model 1
                  </strong>

                  <span>
                    Driving Behaviour
                  </span>

                </div>


                <b>
                  {driver.model_1_latency_ms} ms
                </b>

              </div>


              {/* ==================================================
                  MODEL 2
              ================================================== */}

              <div className="model-card">

                <div>

                  <strong>
                    Model 2
                  </strong>

                  <span>
                    Charging Behaviour
                  </span>

                </div>


                <b>
                  {driver.model_2_latency_ms} ms
                </b>

              </div>


              {/* ==================================================
                  MODEL 3
              ================================================== */}

              <div className="model-card">

                <div>

                  <strong>
                    Model 3
                  </strong>

                  <span>
                    Battery Health / SOH
                  </span>

                </div>


                <b>
                  {driver.model_3_latency_ms} ms
                </b>

              </div>


              {/* ==================================================
                  TOTAL EDGE INFERENCE
                  ESP32 = MODEL 1 + MODEL 2 + MODEL 3
              ================================================== */}

              <div className="model-card">

                <div>

                  <strong>
                    Total Edge Inference
                  </strong>

                  <span>
                    ESP32 · M1 + M2 + M3
                  </span>

                </div>


                <b>
                  {driver.edge_inference_latency_ms} ms
                </b>

              </div>

            </div>


            {/* ==================================================
                SYSTEM LATENCIES
            ================================================== */}

            <div className="system-status">

              <div className="system-item">

                <span>
                  Total Edge Inference
                </span>

                <strong>
                  {driver.edge_inference_latency_ms} ms
                </strong>

              </div>


              <div className="system-item">

                <span>
                  Total Server Processing
                </span>

                <strong>
                  {driver.server_latency_ms} ms
                </strong>

              </div>

            </div>

          </section>


          {/* ==================================================
              ADDITIONAL BATTERY METRICS
              CAP AND CAP_CORR ONLY
              
              SOH IS NOT REPEATED HERE.
          ================================================== */}

          <section className="panel">

            <div className="panel-header">

              <div>

                <h3>
                  Battery Metrics
                </h3>

                <p>
                  Additional capacity predictions from Model 3
                </p>

              </div>

            </div>


            <div className="stats-grid">


              {/* ==================================================
                  PREDICTED CAPACITY
              ================================================== */}

              <div className="stat-card">

                <div className="stat-content">

                  <p>
                    Predicted Capacity
                  </p>


                  <h2>
                    {driver.cap}
                  </h2>


                  <small>
                    Model 3 · CAP
                  </small>

                </div>

              </div>


              {/* ==================================================
                  CORRECTED CAPACITY
              ================================================== */}

              <div className="stat-card">

                <div className="stat-content">

                  <p>
                    Corrected Capacity
                  </p>


                  <h2>
                    {driver.cap_corr}
                  </h2>


                  <small>
                    Model 3 · CAP_CORR
                  </small>

                </div>

              </div>

            </div>

          </section>


          {/* ==================================================
              SYSTEM STATUS
          ================================================== */}

          <section className="system-status">

            <div className="system-item">

              <span>
                ESP32
              </span>

              <strong>
                Data Received
              </strong>

            </div>


            <div className="system-item">

              <span>
                FastAPI
              </span>

              <strong>
                Connected
              </strong>

            </div>


            <div className="system-item">

              <span>
                Prediction ID
              </span>

              <strong>
                #{driver.prediction_id}
              </strong>

            </div>

          </section>


        </main>

      )}

    </div>

  );

}


export default App;