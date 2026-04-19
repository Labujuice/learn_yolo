import cv2
import time
import argparse
import numpy as np
from ultralytics import YOLO

# --- Configuration Parameters ---
# Choose a YOLO model, e.g., 'n' (nano) or 's' (small); Switch models as needed
# MODEL_NAME = 'yolov8n.pt' 
# MODEL_NAME = 'yolo11n.pt'
MODEL_NAME = 'yolov8n.pt'

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
CV_TRACKER_TYPE = 'CSRT' # CSRT (accurate), KCF (fast), MOSSE (fastest)

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
    tracker_builders = {
        'CSRT': cv2.TrackerCSRT_create,
        'KCF': cv2.TrackerKCF_create,
        'MIL': cv2.TrackerMIL_create
    }
    try:
        builder = tracker_builders.get(CV_TRACKER_TYPE)
        if builder:
            return builder()
        else:
            print(f"Error: Invalid tracker type '{CV_TRACKER_TYPE}'. Valid options are: {list(tracker_builders.keys())}")
            return None
    except AttributeError:
        print("\nError: Your OpenCV version is missing tracker modules (AttributeError).")
        print("This usually means the 'contrib' version is not installed.")
        print("Please run: pip install opencv-contrib-python\n")
        return None

# --- Mouse Callback for Object Selection ---
def select_object_callback(event, x, y, flags, param):
    global selected_obj_id, tracking_request
    window_name = param['name']

    if event == cv2.EVENT_LBUTTONDOWN:
        found_id = None
        if window_name == objects_window_name:
            # Click is in the Objects panel
            col = x // CROP_SIZE
            row = y // (CROP_SIZE + 24)
            slot_idx = row * OBJECTS_PER_ROW + col
            if 0 <= slot_idx < len(object_slots):
                found_id = object_slots[slot_idx]
        else:
            # Click is in the main preview window
            for box in last_known_boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                if x1 <= x <= x2 and y1 <= y <= y2:
                    found_id = int(box.id.item())
                    break # Found the topmost object
        
        if found_id is not None:
            if selected_obj_id == found_id:
                selected_obj_id = None # Deselect if clicking the same object
                tracking_request = {'action': 'stop'}
            else:
                selected_obj_id = found_id
                tracking_request = {'action': 'start', 'id': found_id, 'frame': param['frame_copy']}
        elif is_cv_tracking:
            tracking_request = {'action': 'stop'}

def draw_dashed_rectangle(img, pt1, pt2, color, thickness=1, dash_length=10):
    """Draws a dashed rectangle on an image."""
    x1, y1 = pt1
    x2, y2 = pt2

    # Top and bottom edges
    for i in range(x1, x2, dash_length * 2):
        cv2.line(img, (i, y1), (min(i + dash_length, x2), y1), color, thickness)
        cv2.line(img, (i, y2), (min(i + dash_length, x2), y2), color, thickness)

    # Left and right edges
    for i in range(y1, y2, dash_length * 2):
        cv2.line(img, (x1, i), (x1, min(i + dash_length, y2)), color, thickness)
        cv2.line(img, (x2, i), (x2, min(i + dash_length, y2)), color, thickness)

# Create windows and set mouse callbacks
main_window_name = f"YOLO Real-Time Tracking [{MODEL_NAME}] (Press 'q' to exit)"
objects_window_name = "Objects"
cv2.namedWindow(main_window_name)
cv2.namedWindow(objects_window_name)
cv2.setMouseCallback(main_window_name, select_object_callback, param={'name': main_window_name})
cv2.setMouseCallback(objects_window_name, select_object_callback, param={'name': objects_window_name})


print(f"Camera {CAMERA_SOURCE} connected successfully ({actual_w}x{actual_h}). Using model: {MODEL_NAME}. Press 'q' to exit.")

# Add these lines before the main loop to manage fixed slots for object display
MAX_OBJECTS = OBJECTS_PER_ROW * 8  # You can adjust the max number of slots as needed
object_slots = [None] * MAX_OBJECTS  # Each slot holds an obj_id or None

def assign_slot(obj_id):
    # If already assigned, return its slot
    for idx, oid in enumerate(object_slots):
        if oid == obj_id:
            return idx
    # Find first empty slot
    for idx, oid in enumerate(object_slots):
        if oid is None:
            object_slots[idx] = obj_id
            return idx
    # If no empty slot, do not assign (or you can implement replacement policy)
    return None

def remove_slot(obj_id):
    for idx, oid in enumerate(object_slots):
        if oid == obj_id:
            object_slots[idx] = None

# --- Tracking Control Functions (used in main loop) ---
def start_cv_tracking(frame, obj_id):
    global cv_tracker, is_cv_tracking, selected_obj_id
    box_to_track = None
    for box in last_known_boxes:
        if box.id is not None and int(box.id.item()) == obj_id:
            box_to_track = box
            break
    
    if box_to_track:
        coords = box_to_track.xyxy[0].tolist()
        x1, y1, x2, y2 = map(int, coords) # Convert to integers

        # --- Sanity check for the bounding box ---
        # Ensure the bounding box is within the frame boundaries
        h, w, _ = frame.shape
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        width = x2 - x1
        height = y2 - y1

        if width <= 0 or height <= 0:
            print(f"Error: Invalid bounding box for tracker init (width={width}, height={height}). Aborting track.")
            return

        # --- NEW: Handle overly large bounding boxes ---
        # If the bbox area is more than 70% of the frame area, it's too large.
        # Shrink it to 60% of its size, centered, to give the tracker context.
        frame_area = w * h
        bbox_area = width * height
        if bbox_area / frame_area > 0.7:
            print(f"Info: Bounding box is very large ({int(bbox_area/frame_area*100)}% of frame). Shrinking for tracker stability.")
            scale_factor = 0.6
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            x1 = x1 + (width - new_width) // 2
            y1 = y1 + (height - new_height) // 2
            width, height = new_width, new_height

        bbox_cv = (x1, y1, width, height)
        
        cv_tracker = create_cv_tracker()
        if not cv_tracker:
            print("Error: Failed to create OpenCV tracker object.")
            return

        is_initialized = cv_tracker.init(frame, bbox_cv)
        if is_initialized:
             is_cv_tracking = True
             selected_obj_id = obj_id
             print(f"Started OpenCV tracking for Object ID: {selected_obj_id}")
        else:
            h, w, _ = frame.shape
            print(f"Failed to initialize OpenCV tracker for ID: {obj_id}.")
            print(f"  - Bounding Box: {bbox_cv} (x, y, w, h)")
            print(f"  - Frame Shape: ({w}, {h}) (w, h)")
            print("  - Possible reasons: Object features are not clear, object is heavily occluded, or it's at the very edge of the frame.")
            is_cv_tracking = False

def stop_cv_tracking():
    global cv_tracker, is_cv_tracking, selected_obj_id
    is_cv_tracking = False
    cv_tracker = None
    selected_obj_id = None
    print("Stopped OpenCV tracking.")

# --- Real-time Processing Loop ---
while True:
    # Read a frame
    ret, frame = cap.read()
    if not ret:
        print("Unable to read frame, exiting.")
        break

    # Create a clean copy of the frame for tracking purposes
    tracking_frame = frame.copy()

    # Update the frame copy in the mouse callback parameter dictionary
    cv2.setMouseCallback(main_window_name, select_object_callback, param={'name': main_window_name, 'frame_copy': tracking_frame})
    cv2.setMouseCallback(objects_window_name, select_object_callback, param={'name': objects_window_name, 'frame_copy': tracking_frame})

    # Handle tracking requests from the mouse callback
    if tracking_request:
        if tracking_request['action'] == 'start':
            print("START_TRACKING_REQUEST received.")
            start_cv_tracking(tracking_request['frame'], tracking_request['id'])
        elif tracking_request['action'] == 'stop':
            print("STOP_TRACKING_REQUEST received.")
            stop_cv_tracking()
        tracking_request = None # Reset request

    # --- OpenCV Tracker Update (if active) ---
    if is_cv_tracking and cv_tracker is not None:
        ok, bbox = cv_tracker.update(tracking_frame)
        if ok:
            # Tracking success, draw a blue box for the CV tracker
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1) # Draw on the display frame
            cv2.putText(frame, f"CV Tracking ID: {selected_obj_id}", (p1[0], p1[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        else:
            # Tracking failure
            print(f"OpenCV tracker lost object ID: {selected_obj_id}. Reverting to YOLO detection.")
            stop_cv_tracking()

    now = time.time()
    # Perform real-time tracking
    results = model.track(
        frame,
        # --- Parameters for Small Object Detection ---
        conf=args.conf,
        imgsz=args.imgsz,
        persist=True,
        tracker=TRACKER_CONFIG,
        classes=TARGET_CLASSES if TARGET_CLASSES else None, # Filter by class
        verbose=False # Debugging output from the model (set to True for more details
    )

    boxes = results[0].boxes
    names = results[0].names if hasattr(results[0], 'names') else {}
    current_ids = set()
    last_known_boxes = boxes if boxes is not None and hasattr(boxes, 'id') else []

    # Update object_dict with current detections
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
                    if obj_id not in object_dict:
                        object_dict[obj_id] = {
                            'crop': crop_resized,
                            'last_seen': now,
                            'first_seen': now,
                            'name': obj_name,
                            'visible': True
                        }
                    else:
                        object_dict[obj_id]['crop'] = crop_resized
                        object_dict[obj_id]['last_seen'] = now
                        object_dict[obj_id]['visible'] = True
                    current_ids.add(obj_id)

    # Mark objects not currently visible
    for obj_id in object_dict:
        if obj_id not in current_ids:
            object_dict[obj_id]['visible'] = False

    # Remove objects that have disappeared for more than REMOVE_DELAY seconds
    remove_ids = []
    for obj_id, info in object_dict.items():
        if not info['visible'] and now - info['last_seen'] > REMOVE_DELAY:
            remove_ids.append(obj_id)
    for obj_id in remove_ids:
        del object_dict[obj_id]
        remove_slot(obj_id)  # Free the slot
    
    # If selected object disappears, deselect it
    if selected_obj_id is not None and selected_obj_id not in object_dict:
        # Only reset if not in CV tracking mode (CV tracker has its own lost logic)
        if not is_cv_tracking:
            print(f"Selected object {selected_obj_id} has disappeared.")
            selected_obj_id = None


    # Refresh object display window every REFRESH_INTERVAL seconds
    if now - last_panel_update > REFRESH_INTERVAL:
        # Prepare crops for slots
        slot_crops = [np.zeros((CROP_SIZE+24, CROP_SIZE, 3), dtype=np.uint8) for _ in range(MAX_OBJECTS)]
        for obj_id, info in object_dict.items():
            # Only show objects that have been visible for at least APPEAR_THRESHOLD seconds
            if now - info['first_seen'] >= APPEAR_THRESHOLD:
                slot_idx = assign_slot(obj_id)
                if slot_idx is not None:
                    # Draw label (object name + ID) above the crop
                    label_img = np.zeros((24, CROP_SIZE, 3), dtype=np.uint8)
                    text = f"{info['name']} (ID:{obj_id})"
                    cv2.putText(label_img, text, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1, cv2.LINE_AA)
                    crop_with_label = np.vstack([label_img, info['crop']])
                    # Draw colored rectangle around the crop
                    color = (0, 255, 0) if info['visible'] else (0, 165, 255)  # Green or Orange
                    cv2.rectangle(crop_with_label, (0, 24), (CROP_SIZE-1, CROP_SIZE+23), color, 2)

                    # Draw selection highlight
                    if obj_id == selected_obj_id: # If this object is the selected one
                        if is_cv_tracking:
                            # Use a solid blue box to indicate active CV tracking
                            cv2.rectangle(crop_with_label, (0, 24), (CROP_SIZE-1, CROP_SIZE+23), (255, 0, 0), 2)
                        else:
                            # Use a dashed yellow box for simple selection
                            draw_dashed_rectangle(crop_with_label, (0, 24), (CROP_SIZE-1, CROP_SIZE+23), (0, 255, 255), 2, 5)

                    slot_crops[slot_idx] = crop_with_label
        # Arrange crops in grid
        rows = []
        for i in range(0, MAX_OBJECTS, OBJECTS_PER_ROW):
            row_crops = slot_crops[i:i+OBJECTS_PER_ROW]
            rows.append(np.hstack(row_crops))
        if rows:
            objects_panel = np.vstack(rows)
            cv2.imshow(objects_window_name, objects_panel)
        last_panel_update = now

    # Show annotated frame as usual
    annotated_frame = results[0].plot()

    # If not in CV tracking mode, draw the yellow selection highlight from YOLO's data
    if selected_obj_id is not None and not is_cv_tracking:
        for box in last_known_boxes:
            if box.id is not None and int(box.id.item()) == selected_obj_id:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                draw_dashed_rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                break # Found it, no need to check others

    cv2.imshow(main_window_name, annotated_frame)

    # Check if 'q' key is pressed to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup Resources ---
cap.release()
cv2.destroyAllWindows()
print("Experiment finished.")