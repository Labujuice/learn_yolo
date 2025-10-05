# YOLO Object Detection & Tracking Demo

This project demonstrates real-time object detection and tracking using [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) models with OpenCV. It supports live camera input, object cropping, and a dynamic object browsing window that displays detected objects with their IDs and class names.

## Features

- Real-time object detection and tracking using YOLOv8 or custom YOLO models.
- Object browsing window: Shows cropped images of detected objects, arranged in a grid.
- Each object is displayed only after being continuously detected for a set threshold (default: 1s).
- Objects remain in the browsing window for a set delay after disappearing (default: 3s).
- Object slots are fixed: objects keep their position in the browsing window until they disappear.
- Bounding boxes and labels are drawn on both the preview and object browsing windows.
- Easy model switching: just change the `MODEL_NAME` in the code.

## Folder Structure

```
python/
├── objdet.py         # Main detection and tracking script
├── yolov8n.pt        # YOLOv8 nano model (default)
├── yolo11n.pt        # (Optional) Custom YOLO model
├── yolo12n.pt        # (Optional) Custom YOLO model
```

## Requirements

- Python 3.8+
- [Ultralytics](https://github.com/ultralytics/ultralytics) (`pip install ultralytics`)
- OpenCV (`pip install opencv-python`)
- numpy

## Usage

1. **Download a YOLO model**  
   Place your YOLO model (e.g., `yolov8n.pt`) in the `python/` directory.

2. **Run the script**
   ```bash
   cd python
   python objdet.py
   ```

3. **Switch models**  
   Edit `MODEL_NAME` in `objdet.py` to use a different model file if needed.

4. **Controls**
   - Press `q` in the preview window to exit.

## Configuration

You can adjust these parameters in `objdet.py`:

- `CAMERA_SOURCE`: Camera index (default: 0)
- `MODEL_NAME`: Model file name (e.g., `'yolov8n.pt'`)
- `APPEAR_THRESHOLD`: Seconds an object must be detected before display (default: 1)
- `REMOVE_DELAY`: Seconds after disappearance before removing from display (default: 3)
- `OBJECTS_PER_ROW`: Number of objects per row in the browsing window

## Notes

- The object browsing window dynamically resizes based on the number of objects.
- Object slots are persistent: new objects fill empty slots left by disappeared objects.
- For best results, use a compatible USB camera and a supported YOLO model.

## License

This project is for educational and research purposes. Please respect the licenses of YOLO and Ultralytics.

---
```# YOLO Object Detection & Tracking Demo

This project demonstrates real-time object detection and tracking using [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) models with OpenCV. It supports live camera input, object cropping, and a dynamic object browsing window that displays detected objects with their IDs and class names.

## Features

- Real-time object detection and tracking using YOLOv8 or custom YOLO models.
- Object browsing window: Shows cropped images of detected objects, arranged in a grid.
- Each object is displayed only after being continuously detected for a set threshold (default: 1s).
- Objects remain in the browsing window for a set delay after disappearing (default: 3s).
- Object slots are fixed: objects keep their position in the browsing window until they disappear.
- Bounding boxes and labels are drawn on both the preview and object browsing windows.
- Easy model switching: just change the `MODEL_NAME` in the code.

## Folder Structure

```
python/
├── objdet.py         # Main detection and tracking script
├── yolov8n.pt        # YOLOv8 nano model (default)
├── yolo11n.pt        # (Optional) Custom YOLO model
├── yolo12n.pt        # (Optional) Custom YOLO model
```

## Requirements

- Python 3.8+
- [Ultralytics](https://github.com/ultralytics/ultralytics) (`pip install ultralytics`)
- OpenCV (`pip install opencv-python`)
- numpy

## Usage

1. **Download a YOLO model**  
   Place your YOLO model (e.g., `yolov8n.pt`) in the `python/` directory.

2. **Run the script**
   ```bash
   cd python
   python objdet.py
   ```

3. **Switch models**  
   Edit `MODEL_NAME` in `objdet.py` to use a different model file if needed.

4. **Controls**
   - Press `q` in the preview window to exit.

## Configuration

You can adjust these parameters in `objdet.py`:

- `CAMERA_SOURCE`: Camera index (default: 0)
- `MODEL_NAME`: Model file name (e.g., `'yolov8n.pt'`)
- `APPEAR_THRESHOLD`: Seconds an object must be detected before display (default: 1)
- `REMOVE_DELAY`: Seconds after disappearance before removing from display (default: 3)
- `OBJECTS_PER_ROW`: Number of objects per row in the browsing window

## Notes

- The object browsing window dynamically resizes based on the number of objects.
- Object slots are persistent: new objects fill empty slots left by disappeared objects.
- For best results, use a compatible USB camera and a supported YOLO model.

## License

This project is for educational and research purposes. Please respect the licenses of YOLO and Ultralytics.

---