import { useEffect, useState } from "react";
import {
  BatteryMedium, ChevronRight, Cpu, Gauge, Leaf, Lightbulb, MapPin,
  Route, SlidersHorizontal, Sparkles, X, Zap,
} from "lucide-react";

import "./App.css";


// ==================================================
// WHAT-IF HELPERS
// ==================================================

const STATION_NAMES = ["Station 1", "Station 2", "Station 3"];


// Outcomes where a lower value is better for the driver
const LOWER_IS_BETTER = [
  "total_min", "charge_min", "wait_min", "relative_wear",
  "capacity_loss_per_100", "parked_ageing", "driving_stress", "energy_kwh_per_km",
];

// Outcomes where a higher value is better
const HIGHER_IS_BETTER = ["range_added_km", "range_km"];


function factorLabel(value, lowWord, highWord) {
  const v = Number(value);
  if (Math.abs(v - 1) < 0.001) return "as now";
  const pct = Math.round(Math.abs(v - 1) * 100);
  return `${v < 1 ? lowWord : highWord} (${pct}%)`;
}


// Customer wording for the What-If rows (the API keeps its own labels)
const WHATIF_LABEL = {
  total_min: "Total time (drive, wait and charge)",
  charge_min: "Charging time",
  wait_min: "Waiting time",
  energy_kwh: "Energy added",
  range_added_km: "Extra range",
  relative_wear: "Battery wear from this charge (1x = our suggestion)",
  capacity_loss_per_100: "Battery capacity lost after 100 charges like this",
  parked_ageing: "Battery ageing while parked (1x = parked at 50%)",
  driving_stress: "Battery stress from driving",
  energy_kwh_per_km: "Energy use",
  range_km: "Range after charging",
};


function formatValue(value, unit) {
  if (value === null || value === undefined) return "-";
  if (unit === "x") return `${value}x`;
  if (unit === "/100") return `${value}/100`;
  if (unit === "% points") return `${value}%`;
  if (unit === "kWh/km") return `${Math.round(1000 * value)} Wh/km`;
  return `${value} ${unit}`;
}


function formatChange(row) {
  if (row.difference === null || row.difference === undefined) return "-";
  let d = Number(row.difference);
  if (row.unit === "kWh/km") d *= 1000;          // shown in Wh/km
  if (Math.abs(d) < 1e-9) return "same";
  const digits = row.unit === "kWh/km" ? 0 : row.unit === "x" || row.unit === "% points" ? 2 : 1;
  if (Number(d.toFixed(digits)) === 0) return "same";
  return `${d > 0 ? "+" : ""}${d.toFixed(digits)}`;
}


function changeClass(row) {
  const d = Number(row.difference);
  if (!d) return "";
  if (LOWER_IS_BETTER.includes(row.key)) return d < 0 ? "change-better" : "change-worse";
  if (HIGHER_IS_BETTER.includes(row.key)) return d > 0 ? "change-better" : "change-worse";
  return "";
}


// Battery health: change between SOH now and in 10 cycles, in percentage points
function formatSohChange(future, now) {
  const d = Number(future) - Number(now);
  if (!Number.isFinite(d)) return "–";
  return `${d > 0 ? "+" : ""}${d.toFixed(2)}`;
}


// Federated learning: losses can be very small (energy model), so 3 significant digits
function formatLoss(value) {
  const v = Number(value);
  if (!Number.isFinite(v)) return "–";
  return v < 0.001 ? v.toExponential(2) : v.toPrecision(3);
}


function formatBytes(bytes) {
  const b = Number(bytes);
  if (!b) return "–";
  return b < 1024 ? `${b} B` : `${(b / 1024).toFixed(1)} KB`;
}


// SHAP panels: customer title and unit, and which direction of a
// contribution is bad for the driver
// (+1: a higher prediction is worse, -1: a lower one is worse, 0: neither)
const SHAP_PANELS = [
  { key: "stress", badWhen: 1, title: "Battery stress from your driving", unit: "points" },
  { key: "charging", badWhen: 0, title: "Battery you use before your next charge", unit: "%" },
  { key: "battery", badWhen: -1, title: "Change in battery health over 10 charges", unit: "%" },
  { key: "energy", badWhen: 1, title: "Energy use on this trip", unit: "Wh/km" },
];


// Customer wording for model outputs
const BATTERY_STATUS = {
  Low: { label: "Good", tone: "good" },
  Medium: { label: "Needs care", tone: "warn" },
  High: { label: "Service soon", tone: "bad" },
};

const STRESS_LABEL = { Low: "Gentle", Moderate: "Moderate", High: "Harsh" };

const URGENCY_LABEL = {
  "Time-critical": "Charge soon",
  Moderate: "Charge today",
  Flexible: "No rush",
};


function formatShapValue(value) {
  const v = Number(value);
  if (!Number.isFinite(v)) return "–";
  const a = Math.abs(v);
  return a >= 10 ? v.toFixed(1) : a >= 1 ? v.toFixed(2) : v.toFixed(3);
}


// ==================================================
// APP
// ==================================================

function App() {

  const [driver, setDriver] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");


  // ==================================================
  // WHAT-IF SIMULATOR
  //
  // scenario: the driver's own choice (station, charge
  // level, charger power, driving and climate factors)
  // whatIf:   server comparison with the recommendation
  // ==================================================

  const [scenario, setScenario] = useState(null);
  const [flStatus, setFlStatus] = useState(null);
  const [whatIf, setWhatIf] = useState(null);


  // Technical details (latency, federated learning, model outputs)
  // are hidden in the driver view and opened from the header button
  const [showTech, setShowTech] = useState(false);
  const toggleTech = () => setShowTech((previous) => !previous);


  // App tabs; the last opened tab is remembered per browser
  const [tab, setTabState] = useState(() => {
    try {
      return localStorage.getItem("sparkev-tab") || "overview";
    } catch {
      return "overview";
    }
  });

  const setTab = (next) => {
    setTabState(next);
    try {
      localStorage.setItem("sparkev-tab", next);
    } catch {
      // storage unavailable: the tab still changes for this visit
    }
    window.scrollTo({ top: 0, behavior: "smooth" });
  };


  // Close the technical panel with Escape
  useEffect(() => {
    if (!showTech) return;
    const onKey = (e) => e.key === "Escape" && setShowTech(false);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [showTech]);


  const updateScenario = (key, value) =>
    setScenario((previous) => ({ ...previous, [key]: value }));


  // Start from SparkEV's recommendation for each new driver
  useEffect(() => {

    if (!driver) {
      return;
    }

    setScenario((previous) =>
      previous && previous.driverId === driver.driver_id
        ? previous
        : {
            driverId: driver.driver_id,
            station:
              driver.charging_recommendation?.recommended_station ||
              "Station 1",
            target_soc: Math.round(
              driver.charging_recommendation?.target_soc ?? 80
            ),
            power_kw: "",
            driving_style: 1,
            speed: 1,
            climate: 1,
          }
    );

  }, [driver?.driver_id]);


  // Ask the server for the comparison whenever a setting changes
  useEffect(() => {

    if (!scenario) {
      return;
    }

    const timer = setTimeout(async () => {

      try {

        const response = await fetch(
          "http://localhost:8000/whatif",
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              station: scenario.station,
              target_soc: Number(scenario.target_soc),
              power_kw:
                scenario.power_kw === ""
                  ? null
                  : Number(scenario.power_kw),
              driving_style: Number(scenario.driving_style),
              speed: Number(scenario.speed),
              climate: Number(scenario.climate),
            }),
          }
        );

        if (response.ok) {
          setWhatIf(await response.json());
        }

      }
      catch (err) {
        console.error("What-if error:", err);
      }

    }, 250);

    return () => clearTimeout(timer);

  }, [scenario, driver?.prediction_id]);


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
  // ON-DEVICE FEDERATED LEARNING STATUS
  //
  // The car reports each federated round to FastAPI
  // (/fl/report); /fl/status adds the server side.
  // ==================================================

  useEffect(() => {

    const fetchFlStatus = async () => {
      try {
        const response = await fetch("http://localhost:8000/fl/status");
        if (response.ok) {
          setFlStatus(await response.json());
        }
      } catch (err) {
        // the main panel already reports connection problems
      }
    };

    fetchFlStatus();
    const interval = setInterval(fetchFlStatus, 3000);
    return () => clearInterval(interval);

  }, []);



  // ==================================================
  // TOP BAR (shared by every screen)
  // ==================================================

  const TABS = [
    { id: "overview", label: "Overview" },
    { id: "charging", label: "Charging" },
    { id: "insights", label: "Insights" },
    { id: "simulator", label: "Simulator" },
  ];

  const topbar = (
    <header className="topbar">
      <div className="topbar-inner">
        <div className="logo">
          <span className="logo-mark" aria-hidden="true">
            <Zap size={16} strokeWidth={2.5} />
          </span>
          <span className="logo-word">SparkEV</span>
        </div>

        {driver && (
          <nav className="tabs" aria-label="Sections">
            {TABS.map((t) => (
              <button
                key={t.id}
                type="button"
                className={`tab ${tab === t.id ? "tab--active" : ""}`}
                aria-current={tab === t.id ? "page" : undefined}
                onClick={() => setTab(t.id)}
              >
                {t.label}
              </button>
            ))}
          </nav>
        )}

        <div className="topbar-right">
          <span className={`live ${error ? "live--off" : ""}`}>
            <span className="live-dot" />
            {error ? "Offline" : driver ? "Live" : "Connecting"}
          </span>
          {driver && (
            <span className="driver-chip">
              <span className="avatar">{driver.driver_id}</span>
              <span className="driver-name">Driver {driver.driver_id}</span>
            </span>
          )}
          <button
            type="button"
            className={`icon-btn ${showTech ? "icon-btn--on" : ""}`}
            aria-pressed={showTech}
            onClick={toggleTech}
            title="Technical details"
          >
            <Cpu size={17} strokeWidth={2} />
            <span className="icon-btn-label">Technical</span>
          </button>
        </div>
      </div>
    </header>
  );


  // ==================================================
  // WAITING / ERROR SCREEN
  // ==================================================

  if (loading || !driver || error) {
    return (
      <div className="app">
        {topbar}
        <main className="content">
          <section className="card waiting">
            <CarArt />
            <h1>{error ? "Can't reach SparkEV" : "Connecting to your car"}</h1>
            <p>
              {error
                ? "Please check that the SparkEV server is running."
                : "Start your car to see your battery, charging and driving insights."}
            </p>
          </section>
        </main>
        {showTech && (
          <TechDrawer onClose={toggleTech}>
            <p className="drawer-note">
              {error || "Waiting for the first ESP32 result on /esp32/result."}
            </p>
          </TechDrawer>
        )}
      </div>
    );
  }


  // ==================================================
  // VALUES FOR THE DRIVER VIEW
  // ==================================================

  const batteryRisk = driver.battery_risk || "Unknown";
  const battery = BATTERY_STATUS[batteryRisk] || { label: batteryRisk, tone: "warn" };

  const charging =
    driver.charging_recommendation || { stations: [], message: "" };
  const chargingNeeded = charging.stations.length > 0;
  const best = chargingNeeded ? charging.stations[0] : null;
  const urgent = driver.model_2_urgency === "Time-critical";

  const stress = Number(driver.stress);
  const stressTone = stress < 40 ? "good" : stress < 70 ? "warn" : "bad";

  const arrival = Math.round(Number(driver.model_4_arrival_soc_pct));
  const energyWhKm = Math.round(1000 * Number(driver.model_4_energy_kwh_per_km));
  const energyAvg = driver.explanations?.energy?.base_value;
  const energyVsAvg = energyAvg ? Math.round(100 * (energyWhKm / energyAvg - 1)) : null;
  const stressAvg = driver.explanations?.stress?.base_value;

  // latest federated learning report from the car
  const flReport = flStatus && flStatus.latest_report;

  // tips: things to change first, then information
  const insights = driver.insights ?? [];
  const actions = insights.filter((i) => i.kind === "action");
  const infos = insights.filter((i) => i.kind !== "action");

  const hour = new Date().getHours();
  const greeting = hour < 12 ? "Good morning" : hour < 18 ? "Good afternoon" : "Good evening";

  const maxStationTotal = Math.max(...charging.stations.map((s) => Number(s.total_min)), 1);


  // ==================================================
  // APP
  // ==================================================

  return (

    <div className="app">

      {topbar}

      <main className="content">


        {/* ==================================================
            OVERVIEW
        ================================================== */}

        {tab === "overview" && (

          <div className="view">

            {/* ---------- hero ---------- */}

            <section className="hero card">

              <div className="hero-text">
                <p className="eyebrow">{greeting}, Driver {driver.driver_id}</p>
                <h1 className="hero-title">
                  {chargingNeeded ? (urgent ? "Time to charge" : "Charge when convenient") : "You're good to go"}
                </h1>
                <p className="hero-sub">{charging.message}</p>

                <div className="battery-readout">
                  <div className="battery-big">
                    <span className="num">{arrival}</span>
                    <span className="pct">%</span>
                  </div>
                  <div className="battery-caption">
                    <strong>Battery on arrival</strong>
                    <span>{energyWhKm} Wh/km on this trip</span>
                  </div>
                </div>

                <div className="battery-bar" role="img" aria-label={`Battery ${arrival}% on arrival`}>
                  <span
                    className={`battery-fill ${arrival < 20 ? "battery-fill--low" : ""}`}
                    style={{ width: `${Math.min(100, Math.max(0, arrival))}%` }}
                  />
                  {chargingNeeded && (
                    <span
                      className="battery-target"
                      style={{ left: `${Math.min(100, charging.target_soc)}%` }}
                      title={`Suggested charge: ${Math.round(charging.target_soc)}%`}
                    />
                  )}
                </div>
                <div className="battery-scale">
                  <span>0</span>
                  {chargingNeeded && (
                    <span className="battery-scale-target">
                      Charge to {Math.round(charging.target_soc)}%
                    </span>
                  )}
                  <span>100</span>
                </div>

                {chargingNeeded && (
                  <button type="button" className="cta" onClick={() => setTab("charging")}>
                    <Zap size={16} strokeWidth={2.5} />
                    {best.name} · {Math.round(best.total_min)} min
                    <ChevronRight size={16} strokeWidth={2.5} />
                  </button>
                )}
              </div>

              <div className="hero-art">
                <CarArt charging={chargingNeeded} />
              </div>

            </section>


            {/* ---------- gauges ---------- */}

            <section className="tiles">

              <div className="card tile">
                <div className="tile-head">
                  <span className="tile-icon"><BatteryMedium size={16} /></span>
                  Battery health
                </div>
                <div className="tile-body">
                  <Ring value={Number(driver.soh_now)} max={100} tone={battery.tone}>
                    <span className="ring-num">{Number(driver.soh_now).toFixed(1)}</span>
                    <span className="ring-unit">%</span>
                  </Ring>
                  <div className="tile-info">
                    <span className={`pill pill--${battery.tone}`}>{battery.label}</span>
                    <p>About <strong>{Number(driver.soh).toFixed(1)}%</strong> after your next 10 charges</p>
                  </div>
                </div>
              </div>

              <div className="card tile">
                <div className="tile-head">
                  <span className="tile-icon"><Gauge size={16} /></span>
                  Driving style
                </div>
                <div className="tile-body">
                  <Ring value={stress} max={100} tone={stressTone}>
                    <span className="ring-num">{Math.round(stress)}</span>
                    <span className="ring-unit">/100</span>
                  </Ring>
                  <div className="tile-info">
                    <span className={`pill pill--${stressTone}`}>
                      {STRESS_LABEL[driver.stress_status] || driver.stress_status}
                    </span>
                    <p>
                      Battery stress score. Lower is gentler
                      {stressAvg != null ? ` (average trip: ${Math.round(stressAvg)})` : ""}.
                    </p>
                  </div>
                </div>
              </div>

              <div className="card tile">
                <div className="tile-head">
                  <span className="tile-icon"><Leaf size={16} /></span>
                  Efficiency
                </div>
                <div className="tile-body">
                  <Ring
                    value={energyAvg ? Math.min(energyWhKm, 2 * energyAvg) : energyWhKm}
                    max={energyAvg ? 2 * energyAvg : 300}
                    tone={energyVsAvg != null && energyVsAvg > 10 ? "warn" : "good"}
                  >
                    <span className="ring-num">{energyWhKm}</span>
                    <span className="ring-unit">Wh/km</span>
                  </Ring>
                  <div className="tile-info">
                    <span className={`pill pill--${energyVsAvg != null && energyVsAvg > 10 ? "warn" : "good"}`}>
                      {energyVsAvg == null
                        ? "This trip"
                        : energyVsAvg > 0
                          ? `${energyVsAvg}% above average`
                          : `${Math.abs(energyVsAvg)}% below average`}
                    </span>
                    <p>Expected energy use on this trip</p>
                  </div>
                </div>
              </div>

            </section>


            {/* ---------- top tips ---------- */}

            <section className="card section">
              <div className="section-head">
                <div>
                  <h2>For you</h2>
                  <p>Personal tips based on your own driving and charging</p>
                </div>
                <button type="button" className="link-btn" onClick={() => setTab("insights")}>
                  All insights <ChevronRight size={15} />
                </button>
              </div>
              <div className="tips">
                {(actions.length ? actions : infos).slice(0, 4).map((insight, index) => (
                  <Tip insight={insight} key={index} />
                ))}
              </div>
            </section>

          </div>

        )}


        {/* ==================================================
            CHARGING
        ================================================== */}

        {tab === "charging" && (

          <div className="view">

            <section className="card charge-hero">
              <div>
                <p className="eyebrow">Charging plan</p>
                <h1 className="hero-title">
                  {chargingNeeded ? `Charge to ${Math.round(charging.target_soc)}%` : "No charge needed"}
                </h1>
                <p className="hero-sub">{charging.message}</p>
              </div>
              <div className="charge-chips">
                <span className={`pill pill--${urgent ? "warn" : "good"}`}>
                  {URGENCY_LABEL[driver.model_2_urgency] || driver.model_2_urgency}
                </span>
                {chargingNeeded && (
                  <span className="pill pill--neutral">
                    About {Number(charging.need_kwh).toFixed(1)} kWh
                  </span>
                )}
              </div>
            </section>

            {chargingNeeded && (

              <section className="stations">
                {charging.stations.map((s, index) => (
                  <div className={`card station ${index === 0 ? "station--best" : ""}`} key={s.name}>
                    <div className="station-top">
                      <div>
                        <div className="station-name">
                          <MapPin size={15} />
                          {s.name}
                        </div>
                        <div className="station-meta">
                          {s.distance_km} km away · {s.power_kw} kW charger
                        </div>
                      </div>
                      {index === 0 && <span className="badge">Best choice</span>}
                    </div>

                    <div className="station-total">
                      <span className="num">{Math.round(s.total_min)}</span>
                      <span className="unit">min total</span>
                    </div>

                    <div className="timebar" style={{ width: `${(100 * s.total_min) / maxStationTotal}%` }}>
                      <span className="seg seg--drive" style={{ flex: Number(s.travel_min) || 0.0001 }} />
                      <span className="seg seg--wait" style={{ flex: Number(s.wait_min) || 0.0001 }} />
                      <span className="seg seg--charge" style={{ flex: Number(s.charge_min) || 0.0001 }} />
                    </div>

                    <dl className="station-split">
                      <div><dt><i className="dot dot--drive" />Drive</dt><dd>{Math.round(s.travel_min)} min</dd></div>
                      <div><dt><i className="dot dot--wait" />Wait</dt><dd>{Math.round(s.wait_min)} min</dd></div>
                      <div><dt><i className="dot dot--charge" />Charge</dt><dd>{Math.round(s.charge_min)} min</dd></div>
                    </dl>

                    <div className={`busy busy--${String(s.wait_risk).toLowerCase()}`}>
                      {s.wait_risk} chance of a queue
                    </div>
                  </div>
                ))}
              </section>

            )}

            <section className="card section">
              <div className="section-head">
                <div>
                  <h2>Why this plan</h2>
                  <p>Based on your own charging habits</p>
                </div>
              </div>
              <div className="tips">
                {insights
                  .filter((i) => i.topic === "Charging" && i.text !== charging.message)
                  .map((insight, index) => (
                    <Tip insight={insight} key={index} />
                  ))}
              </div>
            </section>

          </div>

        )}


        {/* ==================================================
            INSIGHTS (tips + SHAP explanations)
        ================================================== */}

        {tab === "insights" && (

          <div className="view">

            <div className="split">

              <section className="card section">
                <div className="section-head">
                  <div>
                    <h2>Things to try</h2>
                    <p>Small changes that help your battery and range</p>
                  </div>
                </div>
                <div className="tips">
                  {actions.length
                    ? actions.map((insight, index) => <Tip insight={insight} key={index} />)
                    : <p className="empty">Nothing to change right now. Keep it up.</p>}
                </div>
              </section>

              <section className="card section">
                <div className="section-head">
                  <div>
                    <h2>Good to know</h2>
                    <p>Your latest results in plain words</p>
                  </div>
                </div>
                <div className="tips">
                  {infos.map((insight, index) => <Tip insight={insight} key={index} />)}
                </div>
              </section>

            </div>

            {driver.explanations && (

              <section className="card section">

                <div className="section-head">
                  <div>
                    <h2>What affects your results</h2>
                    <p>How much each factor raises or lowers your result compared with an average driver</p>
                  </div>
                  <div className="legend">
                    <span className="legend-item legend-item--bad">Worse for you</span>
                    <span className="legend-item legend-item--good">Better for you</span>
                  </div>
                </div>

                <div className="shap-grid">
                  {SHAP_PANELS.map(({ key, badWhen, title, unit }) => {
                    const e = driver.explanations[key];
                    if (!e) return null;
                    const maxAbs = Math.max(...e.contributions.map((c) => Math.abs(c.shap)), 1e-9);
                    return (
                      <div className="shap-card" key={key}>
                        <h3>{title}</h3>
                        <p className="shap-summary">
                          Average {withUnit(formatShapValue(e.base_value), unit)}
                          <span className="arrow">→</span>
                          you <strong>{withUnit(formatShapValue(e.prediction), unit)}</strong>
                        </p>
                        <div className="shap-rows">
                          {e.contributions.map((c) => {
                            const tone = badWhen === 0 ? "neutral"
                              : (c.shap * badWhen > 0 ? "bad" : "good");
                            return (
                              <div className="shap-row" key={c.factor}>
                                <span className="shap-label" title={c.factor}>{c.factor}</span>
                                <span className="shap-track">
                                  <span
                                    className={`shap-bar shap-bar--${tone} ${c.shap < 0 ? "shap-bar--neg" : ""}`}
                                    style={{ width: `${(50 * Math.abs(c.shap)) / maxAbs}%` }}
                                  />
                                </span>
                                <span className={`shap-value shap-value--${tone}`}>
                                  {c.shap > 0 ? "+" : ""}{formatShapValue(c.shap)}
                                </span>
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    );
                  })}
                </div>

              </section>

            )}

          </div>

        )}


        {/* ==================================================
            SIMULATOR (WHAT-IF)
            Try another decision and compare its predicted
            consequences with SparkEV's suggestion
        ================================================== */}

        {tab === "simulator" && scenario && (

          <div className="view">

            <section className="card charge-hero">
              <div>
                <p className="eyebrow">Simulator</p>
                <h1 className="hero-title">What if you did it differently?</h1>
                <p className="hero-sub">
                  Change a setting and see what would happen compared with our suggestion.
                  These are estimates; everything you don't change stays as it is now.
                </p>
              </div>
            </section>

            <div className="sim">

              <section className="card section sim-controls">

                <div className="section-head">
                  <div>
                    <h2><SlidersHorizontal size={17} /> Your choice</h2>
                  </div>
                </div>

                <label className="field">
                  <span className="field-label">Station</span>
                  <select
                    value={scenario.station}
                    onChange={(e) => updateScenario("station", e.target.value)}
                  >
                    {STATION_NAMES.map((name) => (
                      <option key={name} value={name}>
                        {name}
                        {name === charging.recommended_station ? " (suggested)" : ""}
                      </option>
                    ))}
                  </select>
                </label>

                <label className="field">
                  <span className="field-label">Charger power</span>
                  <select
                    value={scenario.power_kw}
                    onChange={(e) => updateScenario("power_kw", e.target.value)}
                  >
                    <option value="">Station default</option>
                    <option value="3.7">3.7 kW (slow)</option>
                    <option value="7">7 kW</option>
                    <option value="11">11 kW</option>
                    <option value="22">22 kW (your car takes up to 11 kW)</option>
                  </select>
                </label>

                <Slider
                  label="Charge to"
                  valueLabel={`${scenario.target_soc}%`}
                  min="30" max="100" step="5"
                  value={scenario.target_soc}
                  onChange={(v) => updateScenario("target_soc", v)}
                />
                <Slider
                  label="Driving style"
                  valueLabel={factorLabel(scenario.driving_style, "Smoother", "More aggressive")}
                  min="0.5" max="1.5" step="0.1"
                  value={scenario.driving_style}
                  onChange={(v) => updateScenario("driving_style", v)}
                />
                <Slider
                  label="Speed"
                  valueLabel={factorLabel(scenario.speed, "Slower", "Faster")}
                  min="0.8" max="1.2" step="0.05"
                  value={scenario.speed}
                  onChange={(v) => updateScenario("speed", v)}
                />
                <Slider
                  label="Heating / AC"
                  valueLabel={factorLabel(scenario.climate, "Less", "More")}
                  min="0.5" max="1.5" step="0.1"
                  value={scenario.climate}
                  onChange={(v) => updateScenario("climate", v)}
                />

              </section>

              <section className="card section">

                <div className="section-head">
                  <div>
                    <h2>What happens</h2>
                    <p>Our suggestion compared with your choice</p>
                  </div>
                </div>

                {whatIf ? (
                  <div className="compare">
                    <div className="compare-row compare-row--head">
                      <span />
                      <span>Suggested</span>
                      <span>Yours</span>
                      <span>Change</span>
                    </div>
                    {whatIf.comparison.map((row) => (
                      <div className="compare-row" key={row.key}>
                        <span className="compare-label">{WHATIF_LABEL[row.key] ?? row.label}</span>
                        <span>{formatValue(row.recommended, row.unit)}</span>
                        <span className="compare-yours">{formatValue(row.scenario, row.unit)}</span>
                        <span className={`delta ${changeClass(row)}`}>{formatChange(row)}</span>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="empty">Calculating…</p>
                )}

              </section>

            </div>

          </div>

        )}


        <footer className="foot">
          SparkEV · update #{driver.prediction_id} · refreshed every second
        </footer>

      </main>


      {/* ==================================================
          TECHNICAL DETAILS (slide-in panel)
          Response times, model outputs, explanation
          methods and on-device federated learning
      ================================================== */}

      {showTech && (

        <TechDrawer onClose={toggleTech}>

          <h3 className="drawer-h">Response times</h3>
          <div className="lat-grid">
            {[
              ["Model 1", "Driving behaviour", driver.model_1_latency_ms],
              ["Model 2", "Charge level and urgency", driver.model_2_latency_ms],
              ["Model 3", "Battery health (SOH)", driver.model_3_latency_ms],
              ["Model 4", "Energy and arrival charge", driver.model_4_latency_ms],
              ["Total on ESP32", "Models 1 to 4", driver.edge_inference_latency_ms],
              ["Station ranking", "Server", charging.latency_ms],
              ["SHAP explanations", "Server, all four models", driver.explanations?.shap_ms],
              ["Total on server", "Ranking, SHAP and response", driver.server_latency_ms],
            ].map(([name, what, ms]) => (
              <div className="lat" key={name}>
                <span className="lat-name">{name}</span>
                <span className="lat-what">{what}</span>
                <span className="lat-ms">{ms ?? "–"}<small> ms</small></span>
              </div>
            ))}
          </div>

          <h3 className="drawer-h">Model outputs</h3>
          <table className="dtable">
            <tbody>
              <tr><th>Model 1 · XGBoost</th><td>Stress {driver.stress} / 100 ({driver.stress_status})</td></tr>
              <tr><th>Model 2 · federated MLP</th><td>Charge need {driver.model_2_charge_need_pct}%, target SOC {driver.model_2_target_soc}%</td></tr>
              <tr><th>Model 2 · federated MLP</th><td>Urgency {driver.model_2_urgency} (class {driver.model_2_urgency_class})</td></tr>
              <tr><th>Model 3 · federated MLP</th><td>SOH {driver.soh_now}% → {driver.soh}% in 10 cycles ({formatSohChange(driver.soh, driver.soh_now)} pts, {batteryRisk} risk)</td></tr>
              <tr><th>Model 4 · Elastic Net</th><td>{driver.model_4_energy_kwh_per_km} kWh/km, arrival SOC {driver.model_4_arrival_soc_pct}%</td></tr>
            </tbody>
          </table>

          {driver.explanations && (
            <>
              <h3 className="drawer-h">Explanations</h3>
              <ul className="dlist">
                {SHAP_PANELS.map(({ key }) => {
                  const e = driver.explanations[key];
                  if (!e) return null;
                  return (
                    <li key={key}>
                      <strong>{e.module}</strong> {e.title}: {e.method}
                      {e.model_round != null ? `, global round ${e.model_round}` : ""}
                    </li>
                  );
                })}
                <li>
                  <strong>What-If</strong> Battery wear and parked ageing from measured cell ageing
                  data (KIT dataset, 25 °C); driving stress from Model 1; energy and range from the
                  Model 4 physics-based vehicle model.
                </li>
              </ul>
            </>
          )}

          <h3 className="drawer-h">On-device federated learning</h3>
          <p className="drawer-note">
            {flReport
              ? `Session ${flReport.session} on the car (driver ${flReport.driver_id}): `
                + `${Math.round(flReport.session_ms)} ms for all models, `
                + `lowest free RAM ${Math.round(flReport.free_heap_min / 1024)} KB. `
                + "Raw data stays on the car; only model weights are exchanged."
              : "Waiting for the car's first federated round (it runs while the car is parked or charging)."}
          </p>
          {flReport && (
            <div className="dtable-wrap">
              <table className="dtable dtable--fl">
                <thead>
                  <tr>
                    <th>Model</th>
                    <th>Samples</th>
                    <th>Train</th>
                    <th>Loss start → local → global</th>
                    <th>Upload</th>
                    <th>Round</th>
                  </tr>
                </thead>
                <tbody>
                  {flReport.tasks.map((task) => (
                    <tr key={task.task}>
                      <td>{flStatus.tasks[task.task]?.label ?? task.task}</td>
                      <td>{task.samples}</td>
                      <td>{task.train_ms.toFixed(1)} ms</td>
                      <td>
                        {formatLoss(task.loss_before)} → {formatLoss(task.loss_local)}
                        {" → "}{formatLoss(task.loss_global)}
                      </td>
                      <td>{formatBytes(task.bytes_up)}</td>
                      <td>
                        {task.online
                          ? `#${task.server_round} · ${task.participants} cars`
                          : (task.trained ? "local only" : "failed")}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          <h3 className="drawer-h">System</h3>
          <div className="sys">
            <span><i className="live-dot" /> ESP32 data received</span>
            <span><i className="live-dot" /> FastAPI connected</span>
            <span>Prediction #{driver.prediction_id}</span>
          </div>

        </TechDrawer>

      )}

    </div>

  );

}


// ==================================================
// SMALL COMPONENTS
// ==================================================

const TOPIC_ICON = { Driving: Gauge, Charging: Zap, Battery: BatteryMedium, Trip: Route };


function Tip({ insight }) {
  const Icon = TOPIC_ICON[insight.topic] || (insight.kind === "action" ? Lightbulb : Sparkles);
  return (
    <div className={`tip tip--${insight.kind === "action" ? "action" : "info"}`}>
      <span className="tip-icon"><Icon size={17} strokeWidth={2} /></span>
      <div className="tip-body">
        <span className="tip-topic">{insight.topic}</span>
        <p>{insight.text}</p>
      </div>
    </div>
  );
}


function withUnit(value, unit) {
  return unit === "%" ? `${value}%` : `${value} ${unit}`;
}


// Circular gauge
function Ring({ value, max, tone, children }) {
  const r = 42;
  const c = 2 * Math.PI * r;
  const frac = Math.max(0, Math.min(1, Number(value) / max));
  return (
    <div className={`ring ring--${tone}`}>
      <svg viewBox="0 0 100 100" aria-hidden="true">
        <circle className="ring-track" cx="50" cy="50" r={r} />
        <circle
          className="ring-value"
          cx="50" cy="50" r={r}
          strokeDasharray={`${c * frac} ${c}`}
          transform="rotate(-90 50 50)"
        />
      </svg>
      <div className="ring-label">{children}</div>
    </div>
  );
}


function Slider({ label, valueLabel, min, max, step, value, onChange }) {
  const pct = (100 * (Number(value) - Number(min))) / (Number(max) - Number(min));
  return (
    <label className="field">
      <span className="field-label">
        {label}
        <b>{valueLabel}</b>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        style={{ "--fill": `${pct}%` }}
        onChange={(e) => onChange(e.target.value)}
      />
    </label>
  );
}


function TechDrawer({ onClose, children }) {
  return (
    <div className="drawer-layer">
      <div className="drawer-backdrop" onClick={onClose} />
      <aside className="drawer" role="dialog" aria-modal="true" aria-label="Technical details">
        <div className="drawer-top">
          <div>
            <h2>Technical details</h2>
            <p>Models, response times and federated learning behind this app</p>
          </div>
          <button type="button" className="icon-btn" onClick={onClose} aria-label="Close">
            <X size={18} />
          </button>
        </div>
        <div className="drawer-body">{children}</div>
      </aside>
    </div>
  );
}


// Side view of an electric car (original illustration)
function CarArt({ charging = false }) {
  return (
    <svg className="car" viewBox="0 0 540 230" role="img" aria-label="Your car">
      <defs>
        <linearGradient id="car-body" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#3a4452" />
          <stop offset="0.55" stopColor="#1c222b" />
          <stop offset="1" stopColor="#0f1318" />
        </linearGradient>
        <linearGradient id="car-glass" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#7fb8d6" stopOpacity="0.55" />
          <stop offset="1" stopColor="#0d1620" stopOpacity="0.9" />
        </linearGradient>
        <radialGradient id="car-glow" cx="0.5" cy="0.5" r="0.5">
          <stop offset="0" stopColor="var(--accent)" stopOpacity="0.35" />
          <stop offset="1" stopColor="var(--accent)" stopOpacity="0" />
        </radialGradient>
        <linearGradient id="car-line" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0" stopColor="var(--accent)" stopOpacity="0" />
          <stop offset="0.5" stopColor="var(--accent)" stopOpacity="0.9" />
          <stop offset="1" stopColor="var(--accent)" stopOpacity="0" />
        </linearGradient>
      </defs>

      {/* floor glow and shadow */}
      <ellipse cx="270" cy="196" rx="250" ry="26" fill="url(#car-glow)" />
      <ellipse cx="270" cy="192" rx="215" ry="9" fill="#000" opacity="0.55" />

      {/* body */}
      <path
        d="M30 150 C30 132 40 120 66 116 L122 108 C156 80 198 66 262 64 L338 64
           C382 66 414 82 446 104 L490 112 C512 116 522 130 522 144 L522 156
           C522 162 518 166 512 166 L452 166 A46 46 0 0 0 360 166 L182 166
           A46 46 0 0 0 90 166 L40 166 C34 166 30 162 30 156 Z"
        fill="url(#car-body)"
        stroke="#4a5563"
        strokeWidth="1"
      />

      {/* glasshouse */}
      <path
        d="M136 110 C166 86 204 74 262 72 L332 72 C368 74 396 88 420 106 Z"
        fill="url(#car-glass)"
      />
      <path d="M276 72 L272 108" stroke="#1c222b" strokeWidth="5" />

      {/* beltline accent and door line */}
      <path d="M70 124 L500 120" stroke="url(#car-line)" strokeWidth="1.6" />
      <path d="M272 112 L268 162" stroke="#0b0e12" strokeWidth="1.2" opacity="0.8" />
      <rect x="300" y="128" width="22" height="3" rx="1.5" fill="#5a6574" />
      <rect x="196" y="128" width="22" height="3" rx="1.5" fill="#5a6574" />

      {/* lights */}
      <path d="M496 118 L520 126 L520 132 L494 126 Z" fill="#e8f6ff" />
      <path d="M32 132 L56 128 L56 134 L32 138 Z" fill="#ff4d5e" />

      {/* charge port */}
      <circle cx="78" cy="128" r="4" fill={charging ? "var(--accent)" : "#2a313b"}>
        {charging && <animate attributeName="opacity" values="1;0.35;1" dur="1.8s" repeatCount="indefinite" />}
      </circle>

      {/* wheels */}
      {[136, 406].map((cx) => (
        <g key={cx}>
          <circle cx={cx} cy="166" r="38" fill="#0a0c10" />
          <circle cx={cx} cy="166" r="26" fill="#1a1f27" stroke="#59636f" strokeWidth="2" />
          {[0, 72, 144, 216, 288].map((a) => (
            <line
              key={a}
              x1={cx} y1="166"
              x2={cx + 24 * Math.cos((a * Math.PI) / 180)}
              y2={166 + 24 * Math.sin((a * Math.PI) / 180)}
              stroke="#6b7684"
              strokeWidth="4"
              strokeLinecap="round"
            />
          ))}
          <circle cx={cx} cy="166" r="6" fill="#2a313b" stroke="var(--accent)" strokeWidth="1.5" />
        </g>
      ))}
    </svg>
  );
}


export default App;
