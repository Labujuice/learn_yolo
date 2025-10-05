import cv2
import time
import numpy as np
from ultralytics import YOLO

# --- Configuration Parameters ---
# 0 usually refers to the first camera (UVC camera) on your computer
# If you have multiple cameras, you may need to try 1, 2, ...
CAMERA_SOURCE = 0 
# Choose a YOLO model, e.g., 'n' (nano) or 's' (small); Switch models as needed
MODEL_NAME = 'yolov8n.pt' 
# MODEL_NAME = 'yolo11n.pt'

# Set tracker config (optional, for tracking object IDs)
TRACKER_CONFIG = 'bytetrack.yaml' # Choose a tracker

# Load YOLO model
try:
    model = YOLO(MODEL_NAME)
    print(f"Model loaded successfully: {MODEL_NAME}")
except Exception as e:
    print(f"Failed to load model: {e}")
    exit()

# Start UVC camera (VideoCapture)
cap = cv2.VideoCapture(CAMERA_SOURCE)
if not cap.isOpened():
    print(f"Error: Unable to open camera source {CAMERA_SOURCE}")
    exit()

print(f"Camera connected successfully using model: {MODEL_NAME}. Press 'q' to exit real-time detection.")

CROP_SIZE = 128
REMOVE_DELAY = 10  # Delay in seconds before removing disappeared objects
REFRESH_INTERVAL = 1  # Refresh object window every few seconds
OBJECTS_PER_ROW = 8   # Number of objects per row

# Track each obj_id: {'crop': ..., 'last_seen': ..., 'name': ...}
object_dict = {}
last_panel_update = time.time()

# --- Real-time Processing Loop ---
while True:
    # Read a frame
    ret, frame = cap.read()
    if not ret:
        print("Unable to read frame, exiting.")
        break

    # Perform real-time tracking
    results = model.track(
        frame, 
        conf=0.6, # Confidence threshold
        persist=True,
        tracker=TRACKER_CONFIG # Enable tracking
    )

    # Get result image with bounding boxes and tracking IDs
    # .plot() automatically draws results on the image
    annotated_frame = results[0].plot()
    cv2.imshow(f"YOLO Real-Time Tracking [{MODEL_NAME}] (Press 'q' to exit)", annotated_frame)

    now = time.time()
    boxes = results[0].boxes
    names = results[0].names if hasattr(results[0], 'names') else {}
    current_ids = set()
    if boxes is not None and hasattr(boxes, 'id'):
        for i, box in enumerate(boxes):
            # Get tracking ID
            obj_id = int(box.id.item()) if box.id is not None else None
            if obj_id is not None:
                # Get object bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    crop_resized = cv2.resize(crop, (CROP_SIZE, CROP_SIZE))
                    # Get class name
                    cls_id = int(box.cls.item()) if hasattr(box, 'cls') and box.cls is not None else -1
                    obj_name = names[cls_id] if cls_id in names else f"ID{obj_id}"
                    # Update or add object
                    object_dict[obj_id] = {'crop': crop_resized, 'last_seen': now, 'name': obj_name}
                    current_ids.add(obj_id)

    # Remove objects that disappeared for more than REMOVE_DELAY seconds
    remove_ids = []
    for obj_id, info in object_dict.items():
        if now - info['last_seen'] > REMOVE_DELAY:
            remove_ids.append(obj_id)
    for obj_id in remove_ids:
        del object_dict[obj_id]

    # Refresh object display window every REFRESH_INTERVAL seconds
    if now - last_panel_update > REFRESH_INTERVAL:
        if object_dict:
            crops = []
            labels = []
            for obj_id, info in object_dict.items():
                # Draw label (object name + ID) above the crop
                label_img = np.zeros((24, CROP_SIZE, 3), dtype=np.uint8)
                text = f"{info['name']} (ID:{obj_id})"
                cv2.putText(label_img, text, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1, cv2.LINE_AA)
                # Stack label and crop vertically
                crop_with_label = np.vstack([label_img, info['crop']])
                crops.append(crop_with_label)
            # Arrange crops in a grid (OBJECTS_PER_ROW per row)
            rows = []
            for i in range(0, len(crops), OBJECTS_PER_ROW):
                row_crops = crops[i:i+OBJECTS_PER_ROW]
                # Pad row if not enough objects
                if len(row_crops) < OBJECTS_PER_ROW:
                    pad = OBJECTS_PER_ROW - len(row_crops)
                    for _ in range(pad):
                        row_crops.append(np.zeros_like(crop_with_label))
                rows.append(np.hstack(row_crops))
            objects_panel = np.vstack(rows)
            cv2.imshow("Objects", objects_panel)
        else:
            # Show black screen if no new objects
            cv2.imshow("Objects", np.zeros((CROP_SIZE+24, CROP_SIZE, 3), dtype=np.uint8))
        last_panel_update = now

    # Check if 'q' key is pressed to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup Resources ---
cap.release()
cv2.destroyAllWindows()
print("Experiment finished.")