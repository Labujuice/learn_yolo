# YOLO Object Detection & Tracking Demo

This project provides high-performance real-time object detection and tracking in both **Python** and **C++**. It utilizes [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) for precise detection and integrates multiple OpenCV single-object trackers for robust target locking.

## Key Features

*   **Hybrid Detection & Tracking (Python)**:
    *   **YOLOv8/v11**: Handles global multi-object detection and persistent ID assignment.
    *   **Advanced OpenCV Trackers**: Built-in support for NanoTrack, DaSiamRPN, CSRT, KCF, and MIL with real-time switching.
*   **Top Control Panel (New!)**:
    *   **Mode Toggle**: Switch between `YOLO Mode` (click to track detected objects) and `Manual Mode` (drag to track any ROI).
    *   **Engine Toggle**: Switch between different tracking algorithms on-the-fly via UI buttons.
    *   **Performance Metrics**: Real-time display of calculation time (ms) and execution frequency (Hz) for both detection and tracking modules.
*   **Interactive Camera Selection**: Automatically scans and lists supported UVC camera resolutions and frame rates.
*   **Performance Optimization**: Throttling mechanism allows independent frequency limits for detection and tracking to balance CPU/GPU load.
*   **Object Browser Window**: A secondary window displaying a grid of detected object crops for easy monitoring and selection.

## Folder Structure

```plaintext
├── python/
│   ├── objdet.py              # Main script (Detection + UI Panel + Tracking)
│   ├── tracking_example.py    # Simplified OpenCV-only tracking example
│   ├── bytetrack.yaml         # YOLO internal tracker configuration
│   └── *.onnx / *.pt          # Model weights for trackers and YOLO
│
└── cpp/
    ├── main.cpp               # C++ detection demo (OpenCV DNN)
    ├── yolomodel_pt2onmx.py   # Model conversion tool (.pt -> .onnx)
    └── yolov8n.onnx           # ONNX model for C++ version
```

## Python Implementation Guide

### Requirements

*   Python 3.10+
*   `pip install ultralytics opencv-contrib-python numpy`
*   *(Virtual environment recommended)*

### Quick Start

1.  **Basic Run**:
    ```bash
    cd python
    python objdet.py
    ```
    *If no arguments are provided, an interactive menu will appear to help you select a camera mode.*

2.  **Advanced Run**:
    ```bash
    # Specify resolution, detection frequency, and tracking frequency
    python objdet.py --source 0 --source_cfg 1280x720@30 --det-freq 10 --track-freq 30
    ```

### CLI Arguments

| Argument | Description | Example |
| :--- | :--- | :--- |
| `--source` | Camera index or video file path | `0` or `video.mp4` |
| `--source_cfg` | Hardware resolution and FPS | `1920x1080@60` |
| `--model` | YOLO model file path | `yolo11n.pt` |
| `--classes` | Filter by COCO class IDs | `--classes 0 2` (person & car) |
| `--conf` | Confidence threshold (0~1) | `--conf 0.5` |
| `--det-freq` | Limit detection frequency (Hz) | `--det-freq 5` |
| `--track-freq` | Limit tracking frequency (Hz) | `--track-freq 30` |

### On-Screen Controls

*   **Top UI Panel**:
    *   **YOLO Mode**: Click on a bounding box in the video or an item in the `Objects` window to lock onto it.
    *   **Manual Mode**: Press and **drag the left mouse button** to draw an ROI. Tracking starts upon release.
    *   **Tracker Selection**: Click `NANO`, `DaSiam`, `CSRT`, `KCF`, or `MIL` to switch the tracking engine instantly.
*   **Keyboard**:
    *   `q`: Exit the application.

---

## C++ Implementation (OpenCV DNN)

The C++ version is optimized for raw performance and supports YOLO models via ONNX.

### Steps

1.  **Convert Model**:
    ```bash
    cd cpp
    python yolomodel_pt2onmx.py --input ../python/yolov8n.pt
    ```
2.  **Compile & Run**:
    ```bash
    g++ main.cpp -o objdet_cpp `pkg-config --cflags --libs opencv4`
    ./objdet_cpp
    ```

## Important Notes

*   **Wayland Support**: On Linux systems using Wayland (e.g., Ubuntu/Gnome), the script automatically sets `QT_QPA_PLATFORM=xcb` to ensure window rendering and font support.
*   **Model Files**: When using `NANOTRACK` or `DASIAMRPN`, ensure the required `.onnx` weight files are present in the `python/` directory.

## License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**. 
Please respect the individual licenses of the third-party libraries and models used (e.g., Ultralytics YOLO, OpenCV).
