# YOLO Object Detection & Tracking Demo

This project provides real-time object detection and tracking demonstrations in both **Python** and **C++** using [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) models with OpenCV. It supports live camera input, object cropping, and a dynamic object browsing window that displays detected objects.

## Features

- **Python**: Real-time object detection and tracking using the `ultralytics` library.
- **C++**: Real-time object detection using OpenCV's DNN module with an ONNX model. (Note: Tracking is simplified and does not persist IDs across frames).
- Object browsing window: Shows cropped images of detected objects, arranged in a grid.
- Each object is displayed only after being continuously detected for a set threshold (default: 1s).
- Objects remain in the browsing window for a set delay after disappearing (default: 3s).
- Object slots are fixed: objects keep their position in the browsing window until they disappear.
- Bounding boxes and labels are drawn on both the preview and object browsing windows.
- Easy model switching: just change the `MODEL_NAME` in the code.

## Folder Structure

```plaintext
├── python/
│   ├── objdet.py              # Main Python detection and tracking script
│   ├── bytetrack.yaml         # Tracker configuration for Python script
│   └── yolov8n.pt             # YOLOv8 nano model (default)
│
└── cpp/
    ├── main.cpp               # Main C++ detection script
    ├── yolomodel_pt2onmx.py   # Script to convert .pt model to .onnx for C++
    └── yolov8n.onnx           # ONNX model for C++ version
```

## Python Implementation

The Python version uses the powerful `ultralytics` library to handle both detection and object tracking, providing persistent object IDs across frames.

### Requirements

- Python 3.8+
- Ultralytics (`pip install ultralytics`)
- OpenCV (`pip install opencv-python`)
- NumPy (`pip install numpy`)

### Usage

1.  **Download a YOLO model**  
    Place your YOLO model (e.g., `yolov8n.pt`) in the `python/` directory.

2.  **Run the script**
    Navigate to the `python` directory and run the script.

    *   **Basic execution (uses default camera 0):**
        ```bash
        cd python
        python objdet.py
        ```

    *   **Interactive camera selection:**
        Use the `--source` flag to choose a camera and its resolution/FPS.
        ```bash
        python objdet.py --source
        ```

    *   **Filter specific classes:**
        Use `--classes` to detect only certain objects (e.g., person, car, bus). The class IDs are based on the COCO dataset.
        ```bash
        # Detect person (0), car (2), and bus (5)
        python objdet.py --classes 0 2 5
        ```

    *   **Adjust confidence threshold:**
        Use `--conf` to set a custom confidence level.
        ```bash
        python objdet.py --conf 0.5
        ```

3.  **Controls**
    - Press `q` in the preview window to exit.

### Configuration

#### Runtime Arguments

You can configure the script at runtime using the following command-line arguments:
- `--source`: Show an interactive menu to select camera and resolution.
- `--classes [ID ...]`: Filter detection by class IDs (e.g., `0` for person).
- `--conf [0.0-1.0]`: Set the confidence threshold for detections (default: `0.4`).
- `--imgsz [SIZE]`: Set the image size for inference (default: `640`).

#### In-Script Parameters (`objdet.py`)

- `MODEL_NAME`: Default model file name (e.g., `'yolov8n.pt'`).
- `TRACKER_CONFIG`: Tracker configuration file (e.g., `'bytetrack.yaml'`).
- `APPEAR_THRESHOLD`: Seconds an object must be detected before display (default: 1).
- `REMOVE_DELAY`: Seconds after disappearance before removing from display (default: 3).
- `OBJECTS_PER_ROW`: Number of objects per row in the browsing window.

---

## C++ Implementation

The C++ version uses OpenCV's DNN module to run a YOLO ONNX model. It offers high performance but features a simplified object identification mechanism where IDs are not persistent across frames (i.e., no true object tracking).

### Requirements

- A C++ compiler (e.g., g++, Clang, MSVC).
- **OpenCV** (version 4.5.4 or newer recommended) installed and configured for your environment.

### Setup & Usage

1.  **Convert a YOLO model to ONNX format**  
    The C++ version requires an `.onnx` model. You can convert a `.pt` model using the provided script.
    ```bash
    # Make sure you have ultralytics installed (pip install ultralytics)
    cd cpp
    python yolomodel_pt2onmx.py --input ../python/yolov8n.pt
    ```
    This will create `yolov8n.onnx` in the `cpp/` directory.

2.  **Compile the C++ code**  
    You need to link against your installed OpenCV libraries. Here is an example command for Linux/macOS:
    ```bash
    # Make sure you are in the cpp/ directory
    g++ main.cpp -o objdet_cpp `pkg-config --cflags --libs opencv4`
    ```
    *Note: The `pkg-config` command might differ based on your OS and OpenCV installation (`opencv4` vs `opencv`). For Windows (Visual Studio), you'll need to configure the include/library paths in your project settings.*

3.  **Run the executable**
    ```bash
    ./objdet_cpp
    ```

4.  **Controls**
    - Press `q` in the preview window to exit.

## Notes

- The object browsing window dynamically resizes based on the number of objects.
- For best results, use a compatible USB camera.

## License

This project is for educational and research purposes. Please respect the licenses of the libraries and models used.
