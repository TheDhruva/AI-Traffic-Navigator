# AI Smart Traffic Navigator 🚦

An intelligent, real-time traffic signal optimization system designed for dynamic traffic conditions, specifically tailored for Indian cities. The system leverages computer vision (YOLOv8) to analyze live video feeds, dynamically computes priority scores for each traffic arm, and physically controls traffic signals via an Arduino.

## 🌟 Key Features

* **Real-time Object Detection:** Uses YOLOv8 to detect vehicles (cars, buses, trucks, motorcycles, bicycles, auto-rickshaws) and calculate traffic density based on Passenger Car Units (PCU).
* **Dynamic Signal Control:** Computes the optimal green time dynamically based on density, wait time, and traffic flow.
* **Emergency Vehicle Preemption:** Immediately overrides normal traffic cycles to clear paths for ambulances and fire trucks.
* **Pedestrian Safety Phase:** Automatically triggers a pedestrian crossing phase when enough people gather at the crosswalk.
* **Hazard Extension:** Extends the green light automatically if hazards (e.g., animals on the road) are detected.
* **Optical Flow Analysis:** Measures traffic movement speed to adjust priority scores and prevent green lights on empty or stopped lanes.
* **Hardware & Simulation Modes:** Operates real traffic lights via an Arduino over USB serial, or runs in a full Pygame-based 2D simulation if no hardware is connected.
* **Live Web Dashboard:** A real-time Streamlit dashboard providing metrics, charts, alert banners, and manual system overrides.

## 🏗️ Architecture Stack

* **Computer Vision:** OpenCV, Ultralytics YOLOv8
* **Controller Logic:** Python threading, custom scoring algorithm
* **Hardware Communication:** PySerial (Arduino Uno)
* **Simulation:** Pygame
* **Analytics Dashboard:** Streamlit

## 📁 Project Structure

```text
AI Traffic Navigator/
├── main.py                  # Main entry point orchestrating all threads
├── config.py                # Global configuration, thresholds, ROI coords
├── requirements.txt         # Project dependencies
├── README.md                # Project documentation
├── detection/               # Computer Vision pipeline
│   ├── detector.py          # YOLOv8 inference wrapper
│   ├── density.py           # PCU density estimation
│   ├── flow.py              # Optical flow (movement) analysis
│   └── emergency.py         # Emergency & pedestrian detection
├── controller/              # Decision Engine
│   ├── algorithm.py         # Priority scoring and phase logic
│   └── state.py             # Thread-safe shared intersection state
├── hardware/                # Physical I/O
│   └── arduino.py           # PySerial communication with Arduino
├── simulation/              # 2D Visualization
│   ├── pygame_sim.py        # Pygame intersection simulation
│   └── vehicles.py          # Simulated vehicle physics & movement
├── dashboard/               # Live Analytics
│   └── app.py               # Streamlit web dashboard
└── utils/                   # Helpers for drawing overlays & preprocessing
```

## 🛠️ Installation

1. **Clone the repository and create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

*(Note: YOLOv8 weights `yolov8n.pt` will automatically download on the first run.)*

## 🚀 Usage

### Running the Core System
The main system launches the detection loop, the controller logic, hardware communication, and the Pygame visualizer.

* **Full system (Auto-detects Arduino):**
  ```bash
  python main.py
  ```
* **Demo Mode (Runs simulation using a test video, skips Arduino):**
  ```bash
  python main.py --demo
  ```
* **Use Webcam / Live Camera Feed:**
  ```bash
  python main.py --webcam
  ```
* **Provide custom video source:**
  ```bash
  python main.py --source path/to/video.mp4
  ```
* **Headless Mode (For deploying on servers without GUIs):**
  ```bash
  python main.py --no-sim
  ```

### Launching the Dashboard

The system features a live web dashboard that reads state data dumped by the main process. Open a **second terminal** and run:

```bash
streamlit run dashboard/app.py
```
This dashboard provides real-time traffic statistics, PCU metrics, active phases, and the ability to manually trigger emergency or pedestrian phases.

## 🔌 Arduino Integration

If an Arduino is connected (via Windows `COM` ports or Linux `/dev/ttyACM*`), the software will auto-detect it and send serial commands. 
The communication protocol sends ascii strings like `N:GREEN:30\n` corresponding to: `[ARM_INITIAL]:[PHASE]:[DURATION]`.

* To explicitly specify the port:
  ```bash
  python main.py --port COM5
  ```

## ⚖️ License
This project is open-source and available under the MIT License.
