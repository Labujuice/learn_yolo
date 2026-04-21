import cv2
import time
import argparse
import numpy as np
from ultralytics import YOLO

# --- Configuration Parameters ---
# Choose a YOLO model, e.g., 'n' (nano) or 's' (small); Switch models as needed
MODEL_NAME = 'yolov8n.pt' 
# MODEL_NAME = 'yolo11n.pt'
# MODEL_NAME = 'yolov8n.pt'

# Set tracker config (optional, for tracking object IDs)
TRACKER_CONFIG = 'bytetrack.yaml' # Choose a tracker

# --- Argument Parsing for Class Filtering ---
parser = argparse.ArgumentParser(description="YOLO Object Detection and Tracking")
parser.add_argument(
    '--classes', 
    nargs='*', 
    type=int, 
    help="List of class IDs to detect. If not specified, all classes are detected. e.g., --classes 0 2"
)
parser.add_argument(
    '--conf', 
    type=float, 
    default=0.4, 
    help="Confidence threshold for detection (e.g., 0.25)."
)
parser.add_argument(
    '--imgsz', 
    type=int, 
    default=640, 
    help="Image size for inference (e.g., 640, 1280)."
)
parser.add_argument(
    '--source',
    action='store_true',
    help="Show an interactive menu to select camera source and resolution."
)
args = parser.parse_args()

TARGET_CLASSES = args.classes if args.classes is not None else []

CROP_SIZE = 48
APPEAR_THRESHOLD = 1    # Seconds an object must be continuously detected before display
REMOVE_DELAY = 3        # Seconds after disappearance before removing from display
REFRESH_INTERVAL = 0.2  # Refresh object window every few seconds
OBJECTS_PER_ROW = 8   # Number of objects per row

# Track each obj_id: {'crop': ..., 'last_seen': ..., 'first_seen': ..., 'name': ..., 'visible': ...}
object_dict = {}
last_panel_update = time.time()

# --- Global variables for selection ---
selected_obj_id = None
last_known_boxes = [] # To store the latest bounding boxes for click detection
cv_tracker = None
is_cv_tracking = False
CV_TRACKER_TYPE = 'NANOTRACK' # Options: 'DASIAMRPN', 'NANOTRACK', 'CSRT', 'KCF', 'MIL'

# Tracker Model Paths
NANOTRACK_BACKBONE = "nanotrack_backbone.onnx"
NANOTRACK_HEAD = "nanotrack_head.onnx"

# DaSiamRPN requires THREE files
DASIAMRPN_MODEL = "dasiamrpn_model.onnx"
DASIAMRPN_KERNEL_CLS1 = "dasiamrpn_kernel_cls1.onnx"
DASIAMRPN_KERNEL_R1 = "dasiamrpn_kernel_r1.onnx"

tracking_request = None # To handle requests from the mouse callback


def select_camera_and_mode():
    """
    Scans for available cameras, lets the user select one, and then select a supported
    resolution and frame rate.
    Returns the selected camera index, width, and height.
    """
    # 1. Scan for available cameras
    available_cameras = []
    print("Scanning for available cameras...")
    for i in range(10):  # Check up to 10 camera indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    
    if not available_cameras:
        print("Error: No cameras found.")
        return None, None, None

    # 2. Let user select a camera
    print("\nPlease select a camera:")
    for idx in available_cameras:
        print(f"  [{idx}] Camera {idx}")
    
    camera_idx = -1
    while camera_idx not in available_cameras:
        try:
            camera_idx = int(input(f"Enter camera index {available_cameras}: "))
        except ValueError:
            print("Invalid input. Please enter a number.")

    # 3. List supported resolutions and FPS for the selected camera
    print(f"\nChecking supported modes for Camera {camera_idx}...")
    cap = cv2.VideoCapture(camera_idx)
    supported_modes = set()
    common_resolutions = [(1920, 1080), (1280, 720), (640, 480), (320, 240)]
    common_fps = [60, 30, 24, 15]

    for w, h in common_resolutions:
        for fps in common_fps:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            cap.set(cv2.CAP_PROP_FPS, fps)
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(cap.get(cv2.CAP_PROP_FPS))
            if actual_w == w and actual_h == h and actual_fps > 0:
                 supported_modes.add((actual_w, actual_h, actual_fps))

    cap.release()

    if not supported_modes:
        print("Warning: Could not determine supported modes. Using default settings.")
        return camera_idx, None, None

    # 4. Let user select a mode
    modes = sorted(list(supported_modes), key=lambda x: (x[0], x[2]), reverse=True)
    print("\nPlease select a resolution and FPS:")
    for i, (w, h, fps) in enumerate(modes):
        print(f"  [{i}] {w}x{h} @ {fps} FPS")
    
    mode_idx = -1
    while not (0 <= mode_idx < len(modes)):
        try:
            mode_idx = int(input(f"Enter mode index [0-{len(modes)-1}]: "))
        except ValueError:
            print("Invalid input. Please enter a number.")
            
    selected_mode = modes[mode_idx]
    return camera_idx, selected_mode[0], selected_mode[1], selected_mode[2]

# Load YOLO model
try:
    model = YOLO(MODEL_NAME)
    # model = YOLO(MODEL_NAME).to('cuda') 
    print(f"Model loaded successfully: {MODEL_NAME}")
except Exception as e:
    print(f"Failed to load model: {e}")
    exit()

# --- Camera Setup ---
if args.source:
    CAMERA_SOURCE, FRAME_WIDTH, FRAME_HEIGHT, FPS = select_camera_and_mode()
    if CAMERA_SOURCE is None:
        exit()
else:
    # Use default camera source without interactive selection
    CAMERA_SOURCE = 0  # Default camera
    FRAME_WIDTH, FRAME_HEIGHT, FPS = None, None, None
    print("Using default camera source 0. Use --source for interactive selection.")

# Start UVC camera (VideoCapture)
cap = cv2.VideoCapture(CAMERA_SOURCE)
if FRAME_WIDTH and FRAME_HEIGHT and FPS:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)

if not cap.isOpened():
    print(f"Error: Unable to open camera source {CAMERA_SOURCE}")
    exit()

actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# --- OpenCV Tracker Management ---
def create_cv_tracker():
    """Creates an OpenCV tracker based on the specified type, with error handling."""
    if CV_TRACKER_TYPE == 'DASIAMRPN':
        try:
            import os
            required_files = [DASIAMRPN_MODEL, DASIAMRPN_KERNEL_CLS1, DASIAMRPN_KERNEL_R1]
            if any(not os.path.exists(f) for f in required_files):
                print(f"\nError: DaSiamRPN models missing! Please ensure all 3 files are in the python/ directory.")
                return None
            params = cv2.TrackerDaSiamRPN_Params()
            params.model = DASIAMRPN_MODEL
            params.kernel_cls1 = DASIAMRPN_KERNEL_CLS1
            params.kernel_r1 = DASIAMRPN_KERNEL_R1
            params.backend = cv2.dnn.DNN_BACKEND_OPENCV
            params.target = cv2.dnn.DNN_TARGET_CPU
            return cv2.TrackerDaSiamRPN_create(params)
        except Exception as e:
            print(f"Error creating DaSiamRPN: {e}")
            return None

    if CV_TRACKER_TYPE == 'NANOTRACK':
        try:
            import os
            if not os.path.exists(NANOTRACK_BACKBONE) or not os.path.exists(NANOTRACK_HEAD):
                print(f"\nError: NanoTrack models not found!")
                return None
            params = cv2.TrackerNano_Params()
            params.backbone = NANOTRACK_BACKBONE
            params.neckhead = NANOTRACK_HEAD
            params.backend = cv2.dnn.DNN_BACKEND_OPENCV
            params.target = cv2.dnn.DNN_TARGET_CPU
            return cv2.TrackerNano_create(params)
        except Exception as e:
            print(f"Error creating NanoTrack: {e}")
            return None

    tracker_builders = {
        'CSRT': cv2.legacy.TrackerCSRT.create,
        'KCF': cv2.legacy.TrackerKCF.create,
        'MIL': cv2.legacy.TrackerMIL.create,
        'MOSSE': cv2.legacy.TrackerMOSSE.create,
        'MEDIANFLOW': cv2.legacy.TrackerMedianFlow.create,
        'TLD': cv2.legacy.TrackerTLD.create,
    }
    try:
        builder = tracker_builders.get(CV_TRACKER_TYPE)
        if builder:
            return builder()
        else:
            print(f"Error: Invalid tracker type '{CV_TRACKER_TYPE}'.")
            return None
    except AttributeError:
        print("\nError: Your OpenCV version is missing tracker modules.")
        return None

# --- Mouse Callback for Object Selection ---
def select_object_callback(event, x, y, flags, param):
    global selected_obj_id, tracking_request
    window_name = param['name']

    if event == cv2.EVENT_LBUTTONDOWN:
        found_id = None
        if window_name == objects_window_name:
            col = x // CROP_SIZE
            row = y // (CROP_SIZE + 24)
            slot_idx = row * OBJECTS_PER_ROW + col
            if 0 <= slot_idx < len(object_slots):
                found_id = object_slots[slot_idx]
        else:
            for box in last_known_boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                if x1 <= x <= x2 and y1 <= y <= y2:
                    found_id = int(box.id.item())
                    break
        
        if found_id is not None:
            if selected_obj_id == found_id:
                selected_obj_id = None
                tracking_request = {'action': 'stop'}
            else:
                selected_obj_id = found_id
                tracking_request = {'action': 'start', 'id': found_id}
        elif is_cv_tracking:
            tracking_request = {'action': 'stop'}

def draw_dashed_rectangle(img, pt1, pt2, color, thickness=1, dash_length=10):
    x1, y1 = pt1
    x2, y2 = pt2
    for i in range(x1, x2, dash_length * 2):
        cv2.line(img, (i, y1), (min(i + dash_length, x2), y1), color, thickness)
        cv2.line(img, (i, y2), (min(i + dash_length, x2), y2), color, thickness)
    for i in range(y1, y2, dash_length * 2):
        cv2.line(img, (x1, i), (x1, min(i + dash_length, y2)), color, thickness)
        cv2.line(img, (x2, i), (x2, min(i + dash_length, y2)), color, thickness)

main_window_name = f"YOLO Real-Time Tracking [{MODEL_NAME}] (Press 'q' to exit)"
objects_window_name = "Objects"
cv2.namedWindow(main_window_name)
cv2.namedWindow(objects_window_name)
cv2.setMouseCallback(main_window_name, select_object_callback, param={'name': main_window_name})
cv2.setMouseCallback(objects_window_name, select_object_callback, param={'name': objects_window_name})

print(f"Camera {CAMERA_SOURCE} connected successfully. Using model: {MODEL_NAME}. Press 'q' to exit.")

MAX_OBJECTS = OBJECTS_PER_ROW * 8
object_slots = [None] * MAX_OBJECTS

def assign_slot(obj_id):
    for idx, oid in enumerate(object_slots):
        if oid == obj_id: return idx
    for idx, oid in enumerate(object_slots):
        if oid is None:
            object_slots[idx] = obj_id
            return idx
    return None

def remove_slot(obj_id):
    for idx, oid in enumerate(object_slots):
        if oid == obj_id: object_slots[idx] = None

def start_cv_tracking(frame, obj_id):
    global cv_tracker, is_cv_tracking, selected_obj_id
    box_to_track = None
    for box in last_known_boxes:
        if box.id is not None and int(box.id.item()) == obj_id:
            box_to_track = box
            break
    
    if box_to_track:
        coords = box_to_track.xyxy[0].tolist()
        x1, y1, x2, y2 = map(int, coords)
        h_f, w_f, _ = frame.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_f, x2), min(h_f, y2)
        width, height = x2 - x1, y2 - y1

        if width <= 0 or height <= 0: return

        if CV_TRACKER_TYPE in ['NANOTRACK', 'DASIAMRPN']:
            shrink = 0.05
            dw, dh = width * shrink, height * shrink
            x1, y1 = int(x1 + dw), int(y1 + dh)
            width, height = int(width - 2 * dw), int(height - 2 * dh)
            x1, y1 = max(5, min(w_f - 10, x1)), max(5, min(h_f - 10, y1))
            width, height = max(10, min(w_f - x1 - 5, width)), max(10, min(h_f - y1 - 5, height))

        bbox_cv = (int(x1), int(y1), int(width), int(height))
        cv_tracker = create_cv_tracker()
        if not cv_tracker: return

        try:
            res = cv_tracker.init(frame, bbox_cv)
            if res is None or res:
                is_cv_tracking = True
                selected_obj_id = obj_id
                print(f"Started {CV_TRACKER_TYPE} tracking for ID: {selected_obj_id}")
            else:
                is_cv_tracking = False
        except Exception as e:
            print(f"Tracker init error: {e}")
            is_cv_tracking = False

def stop_cv_tracking():
    global cv_tracker, is_cv_tracking, selected_obj_id
    is_cv_tracking, cv_tracker, selected_obj_id = False, None, None
    print("Stopped CV tracking.")

while True:
    ret, frame = cap.read()
    if not ret: break

    tracking_frame = frame.copy()

    if tracking_request:
        if tracking_request['action'] == 'start':
            start_cv_tracking(tracking_frame, tracking_request['id'])
        elif tracking_request['action'] == 'stop':
            stop_cv_tracking()
        tracking_request = None

    tracker_bbox = None
    if is_cv_tracking and cv_tracker is not None:
        ok, bbox = cv_tracker.update(tracking_frame)
        if ok:
            tracker_bbox = bbox
        else:
            stop_cv_tracking()

    now = time.time()
    results = model.track(frame, conf=args.conf, imgsz=args.imgsz, persist=True, tracker=TRACKER_CONFIG, classes=TARGET_CLASSES if TARGET_CLASSES else None, verbose=False)

    boxes = results[0].boxes
    names = results[0].names if hasattr(results[0], 'names') else {}
    current_ids = set()
    last_known_boxes = boxes if boxes is not None and hasattr(boxes, 'id') else []

    if boxes is not None and hasattr(boxes, 'id'):
        for box in boxes:
            obj_id = int(box.id.item()) if box.id is not None else None
            if obj_id is not None:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    crop_resized = cv2.resize(crop, (CROP_SIZE, CROP_SIZE))
                    cls_id = int(box.cls.item()) if hasattr(box, 'cls') else -1
                    obj_name = names[cls_id] if cls_id in names else f"ID{obj_id}"
                    if obj_id not in object_dict:
                        object_dict[obj_id] = {'crop': crop_resized, 'last_seen': now, 'first_seen': now, 'name': obj_name, 'visible': True}
                    else:
                        object_dict[obj_id].update({'crop': crop_resized, 'last_seen': now, 'visible': True})
                    current_ids.add(obj_id)

    for oid in object_dict:
        if oid not in current_ids: object_dict[oid]['visible'] = False

    remove_ids = [oid for oid, info in object_dict.items() if not info['visible'] and now - info['last_seen'] > REMOVE_DELAY]
    for oid in remove_ids:
        del object_dict[oid]
        remove_slot(oid)
    
    if selected_obj_id is not None and selected_obj_id not in object_dict and not is_cv_tracking:
        selected_obj_id = None

    if now - last_panel_update > REFRESH_INTERVAL:
        slot_crops = [np.zeros((CROP_SIZE+24, CROP_SIZE, 3), dtype=np.uint8) for _ in range(MAX_OBJECTS)]
        for obj_id, info in object_dict.items():
            if now - info['first_seen'] >= APPEAR_THRESHOLD:
                slot_idx = assign_slot(obj_id)
                if slot_idx is not None:
                    label_img = np.zeros((24, CROP_SIZE, 3), dtype=np.uint8)
                    cv2.putText(label_img, f"ID:{obj_id}", (2, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)
                    crop_with_label = np.vstack([label_img, info['crop']])
                    color = (0, 255, 0) if info['visible'] else (0, 165, 255)
                    cv2.rectangle(crop_with_label, (0, 24), (CROP_SIZE-1, CROP_SIZE+23), color, 2)
                    if obj_id == selected_obj_id:
                        if is_cv_tracking: cv2.rectangle(crop_with_label, (0, 24), (CROP_SIZE-1, CROP_SIZE+23), (255, 0, 0), 2)
                        else: draw_dashed_rectangle(crop_with_label, (0, 24), (CROP_SIZE-1, CROP_SIZE+23), (0, 255, 255), 2, 5)
                    slot_crops[slot_idx] = crop_with_label
        rows = [np.hstack(slot_crops[i:i+OBJECTS_PER_ROW]) for i in range(0, MAX_OBJECTS, OBJECTS_PER_ROW)]
        cv2.imshow(objects_window_name, np.vstack(rows))
        last_panel_update = now

    annotated_frame = results[0].plot()

    if tracker_bbox is not None:
        p1 = (int(tracker_bbox[0]), int(tracker_bbox[1]))
        p2 = (int(tracker_bbox[0] + tracker_bbox[2]), int(tracker_bbox[1] + tracker_bbox[3]))
        cv2.rectangle(annotated_frame, p1, p2, (255, 0, 0), 2)
        cv2.putText(annotated_frame, f"{CV_TRACKER_TYPE} ID: {selected_obj_id}", (p1[0], p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    if selected_obj_id is not None and not is_cv_tracking:
        for box in last_known_boxes:
            if box.id is not None and int(box.id.item()) == selected_obj_id:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                draw_dashed_rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                break
    cv2.imshow(main_window_name, annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
print("Experiment finished.")
