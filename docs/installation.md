# Installation Guide

This guide explains how to set up and run every part of SparkEV: the server, the dashboard, the
ESP32 firmware and the research notebooks.

## 1. Requirements

| Part | Requirement |
|---|---|
| Server and tools | Python 3.11 or later (tested with 3.13) |
| Dashboard | Node.js 20 or later |
| Firmware | Arduino IDE 2.x with ESP32 board support |
| Hardware | ESP32 development board, SH1106 OLED display (128 x 64, I2C) |

## 2. Get the code

```bash
git clone <this-repository-url>
cd SparkEV
```

The folder created by `git clone` has the same name as the GitHub repository. If the repository
has a different name, use that name in the `cd` command.

## 3. Add the datasets

The datasets are not included in this repository because of their size and licences. Download
them from the sources listed in the [README](../README.md#datasets).

The server needs the following files in `data/raw/`:

| Files | Used by |
|---|---|
| `LFP01_fast_feature.xlsx`, `NMC02_fast_feature.xlsx`, `Lab_fast_feature.xlsx`, `NMC01_Car01_fast_feature.xlsx`, `NMC01_slow_feature.xlsx`, `NMC01_Car02_fast_feature.xlsx` | Module 3 |
| `processed_data_longer_than_30.xlsx` | Module 2 |

The server checks for these files when it starts and stops with an error if one is missing.

## 4. Set up Python

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

To also run the research notebooks:

```bash
pip install -r requirements-notebooks.txt
```

## 5. Run the server

```bash
uvicorn server:app --app-dir backend --host 0.0.0.0 --port 8000
```

Open http://localhost:8000. The response should be:

```json
{"project": "SparkEV", "status": "running", "model_1": "loaded", "model_2": "loaded", "model_3": "loaded", ...}
```

The option `--host 0.0.0.0` makes the server reachable from other devices on the same network,
which the ESP32 needs. See the [API reference](api.md) for all endpoints.

## 6. Run the dashboard

In a second terminal:

```bash
cd dashboard
npm install
npm run dev
```

Open http://localhost:5173. The dashboard reads the latest result from the server every second.
Until a result arrives, it shows that it is waiting for data.

## 7. Set up the ESP32

### Wiring

| OLED pin | ESP32 pin |
|---|---|
| VCC | 3.3V |
| GND | GND |
| SDA | GPIO 21 |
| SCL | GPIO 22 |

The firmware expects the display at I2C address `0x3C`.

### Settings

Create your settings file from the template:

```bash
cp esp32/firmware/SparkEV_Integrated/secrets.example.h esp32/firmware/SparkEV_Integrated/secrets.h
```

Edit `secrets.h`:

| Setting | Value |
|---|---|
| `WIFI_SSID` | Your Wi-Fi network name |
| `WIFI_PASSWORD` | Your Wi-Fi password |
| `SERVER_URL` | `http://<your-computer-ip>:8000/esp32/result` |

To find your computer's local IP address, run `ipconfig getifaddr en0` on macOS or `ipconfig` on
Windows. The ESP32 and the computer must be on the same network.

`secrets.h` is listed in `.gitignore`, so your credentials are never committed.

### Upload

1. In the Arduino IDE, install the **Adafruit GFX** and **Adafruit SH110X** libraries from the
   Library Manager.
2. Open `esp32/firmware/SparkEV_Integrated/SparkEV_Integrated.ino`.
3. Select your board (for example **ESP32 Dev Module**) and port.
4. Set **Tools > Partition Scheme** to **Huge APP (3MB No OTA/1MB SPIFFS)**. The program is about
   2 MB because it contains all three models, and the default setting only allows 1.2 MB.
5. Upload.
6. Open the Serial Monitor at 115200 baud to follow the progress.

The ESP32 connects to Wi-Fi, runs the three models, shows each result on the display and sends
the results to the server. The dashboard updates automatically.

To switch the demonstration driver, change `#define TEST_DRIVER 17` at the top of the sketch to
`17` or `24` and upload again.

## 8. Test without the ESP32

The server can run the models on the computer for testing:

```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"driver_id": 17}'
```

The result is also shown on the dashboard. This endpoint is for testing only. In the deployed
setup, all predictions are made on the ESP32.

## 9. Run the research notebooks

The notebooks are in `notebooks/`, with one folder per module. They were developed by different
team members, some in Google Colab and some in a local Jupyter environment.

```bash
jupyter notebook notebooks/
```

Each notebook sets its data location near the top (for example `data_path` or `MODULE_DIR`).
Change this to point to your copy of the data, then run the notebook from top to bottom. To run
a notebook in Google Colab instead, upload it and place the data in your Google Drive.

## Troubleshooting

| Problem | Solution |
|---|---|
| Server stops with `Model 3 dataset not found` | A dataset file is missing from `data/raw/`. See step 3. |
| Dashboard shows "Waiting for ESP32 data" | No result has reached the server yet. Check that the ESP32 is running, or use the test request in step 8. |
| ESP32 cannot reach the server | Check that `SERVER_URL` uses your computer's local IP address, not `localhost`, that both devices are on the same network, and that the server was started with `--host 0.0.0.0`. |
| Upload fails with `text section exceeds available space in board` | The partition scheme is too small. Set **Tools > Partition Scheme** to **Huge APP (3MB No OTA/1MB SPIFFS)**. |
| Display stays blank | Check the wiring and that the display uses I2C address `0x3C`. |
