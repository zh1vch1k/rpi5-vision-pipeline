# 🚀 Hybrid Edge Video Analytics Pipeline

High-performance real-time video processing and tracking framework running on **Raspberry Pi 5**.

The project links a low-latency **Python CV/ML pipeline** (YOLOv8, Optical Flow/MOG2, Poetry) with a fast **C++ FFmpeg streamer** using Zero-Copy **POSIX Shared Memory** and **Semaphores**.

---

## 🛠 Architecture Overview

```
                        [ Camera Stream ]
                               │
                               ▼
        ┌────────────────────────────────────────┐
        │            Python Engine (Inference)    │
        │  • Motion Detection / MOG2 Trigger      │
        │  • Bounding Box Crop & YOLO Verification│
        │  • Optical Flow (Lucas-Kanade)          │
        └───────────────────┬────────────────────┘
                             │  Zero-Copy Shared Memory
                             │  + POSIX Semaphores (IPC)
                             ▼
        ┌────────────────────────────────────────┐
        │             C++ Stream Engine           │
        │  • Fast SHM Reader (shm_memory_reader)  │
        │  • Config Parser (config.hpp)           │
        │  • RTSP/UDP Streamer (FFmpeg)           │
        └───────────────────┬────────────────────┘
                             │
                             ▼
                       [ Video Output ]
```

---

## ✨ Features

* **Zero-Copy IPC:** High-throughput frame transfer between Python and C++ using `POSIX Shared Memory` and `sem_open` for synchronization.
* **Smart Motion Triggering:** Reduces NPU/CPU load by running YOLOv8 inference only on cropped regions of interest (ROI) triggered by motion detection.
* **Typed Configuration:** Unified `config.json` configuration parsed into strict C++ structures via `nlohmann/json`.
* **Isolated Environment:** Managed with `Poetry` for Python dependencies and clean modern C++ for the streaming backend.

---

## 📁 Repository Structure

```text
├── config.json            # Main shared configuration file
├── config.hpp              # C++ Header-only JSON configuration parser
├── shm_memory_writer.py   # Python IPC Shared Memory & Semaphore Manager
├── shm_memory_reader.cpp  # C++ SHM Reader module
├── ffmpeg_streamer.cpp    # C++ FFmpeg streaming handler
├── lucas_canade.py        # Optical Flow tracking module
├── main.py                # Python CV Pipeline entry point
├── main.cpp               # C++ Streamer entry point
└── pyproject.toml         # Poetry dependency configuration
```

---

## 🚦 Quick Start

### 1. Prerequisites

* **C++ Compiler:** Supporting C++17 or higher
* **Python:** 3.10+
* **Dependencies:** `poetry`, `ffmpeg`, `nlohmann-json`

### 2. Installation & Setup

```bash
# Install Python dependencies
poetry install

# Compile C++ Streamer
g++ -std=c++17 main.cpp ffmpeg_streamer.cpp shm_memory_reader.cpp -o streamer -lpthread -lrt
```

### 3. Running the Pipeline

Start the Python processing node first to initialize the shared memory segment:

```bash
# Terminal 1: Run Python CV Engine
poetry run python main.py

# Terminal 2: Run C++ Streamer
./streamer
```

---

## ⚙️ Configuration (`config.json`)

```json
{
  "network": {
    "ip": "127.0.0.1",
    "port": "1221"
  },
  "video": {
    "width": 640,
    "height": 360,
    "fps": 60
  },
  "ipc": {
    "buffer_path": "/rpi5_pipeline",
    "semaphore_name": "/rpi5_semaphore"
  }
}
```
