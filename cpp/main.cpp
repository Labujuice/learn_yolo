#include <opencv2/opencv.hpp>
#include <chrono>
#include <vector>
#include <string>
#include <map>
#include <memory>
#include <iostream>

// Note: This is a conceptual translation. For actual YOLO inference in C++, 
// you need to use a C++ YOLO library such as YOLOv8 ONNX, OpenCV DNN, or TensorRT.
// The following code uses OpenCV for video and window management, and simulates detection/tracking.

using namespace cv;
using namespace std;
using namespace std::chrono;

struct ObjectInfo {
    Mat crop;
    steady_clock::time_point last_seen;
    steady_clock::time_point first_seen;
    string name;
    bool visible;
};

const int CAMERA_SOURCE = 0;
const string MODEL_NAME = "yolov8n.pt"; // Placeholder, not used in this mockup
const int CROP_SIZE = 128;
const double APPEAR_THRESHOLD = 1.0; // seconds
const double REMOVE_DELAY = 3.0;      // seconds
const double REFRESH_INTERVAL = 0.2;  // seconds
const int OBJECTS_PER_ROW = 8;
const int MAX_OBJECTS = OBJECTS_PER_ROW * 8;

vector<int> object_slots(MAX_OBJECTS, -1); // Each slot holds an obj_id or -1
map<int, ObjectInfo> object_dict;

int assign_slot(int obj_id) {
    // If already assigned, return its slot
    for (int idx = 0; idx < object_slots.size(); ++idx) {
        if (object_slots[idx] == obj_id)
            return idx;
    }
    // Find first empty slot
    for (int idx = 0; idx < object_slots.size(); ++idx) {
        if (object_slots[idx] == -1) {
            object_slots[idx] = obj_id;
            return idx;
        }
    }
    return -1;
}

void remove_slot(int obj_id) {
    for (int idx = 0; idx < object_slots.size(); ++idx) {
        if (object_slots[idx] == obj_id) {
            object_slots[idx] = -1;
        }
    }
}

// Simulate detection results (replace with real YOLO inference in practice)
struct Detection {
    int obj_id;
    Rect bbox;
    string name;
};

vector<Detection> mock_detect(const Mat& frame, int frame_idx) {
    // Simulate two moving objects
    vector<Detection> dets;
    int x = 50 + (frame_idx % 200);
    dets.push_back({1, Rect(x, 100, 80, 80), "person"});
    if (frame_idx > 30 && frame_idx < 200)
        dets.push_back({2, Rect(200, 200 + (frame_idx % 100), 80, 80), "bottle"});
    return dets;
}

int main() {
    VideoCapture cap(CAMERA_SOURCE);
    if (!cap.isOpened()) {
        cerr << "Error: Unable to open camera source " << CAMERA_SOURCE << endl;
        return -1;
    }
    cout << "Camera connected successfully using model: " << MODEL_NAME << ". Press 'q' to exit real-time detection." << endl;

    auto last_panel_update = steady_clock::now();
    int frame_idx = 0;

    while (true) {
        Mat frame;
        cap >> frame;
        if (frame.empty()) {
            cerr << "Unable to read frame, exiting." << endl;
            break;
        }
        auto now = steady_clock::now();

        // Simulate detection and tracking
        vector<Detection> detections = mock_detect(frame, frame_idx);
        set<int> current_ids;
        for (const auto& det : detections) {
            Rect bbox = det.bbox & Rect(0, 0, frame.cols, frame.rows);
            if (bbox.width > 0 && bbox.height > 0) {
                Mat crop = frame(bbox).clone();
                resize(crop, crop, Size(CROP_SIZE, CROP_SIZE));
                if (object_dict.find(det.obj_id) == object_dict.end()) {
                    object_dict[det.obj_id] = {crop, now, now, det.name, true};
                } else {
                    object_dict[det.obj_id].crop = crop;
                    object_dict[det.obj_id].last_seen = now;
                    object_dict[det.obj_id].visible = true;
                }
                current_ids.insert(det.obj_id);
            }
        }
        // Mark objects not currently visible
        for (auto& [obj_id, info] : object_dict) {
            if (current_ids.find(obj_id) == current_ids.end()) {
                info.visible = false;
            }
        }
        // Remove objects that have disappeared for more than REMOVE_DELAY seconds
        vector<int> remove_ids;
        for (const auto& [obj_id, info] : object_dict) {
            if (!info.visible && duration_cast<duration<double>>(now - info.last_seen).count() > REMOVE_DELAY) {
                remove_ids.push_back(obj_id);
            }
        }
        for (int obj_id : remove_ids) {
            object_dict.erase(obj_id);
            remove_slot(obj_id);
        }

        // Draw bounding boxes on preview
        for (const auto& det : detections) {
            rectangle(frame, det.bbox, Scalar(0, 255, 0), 2);
            putText(frame, det.name + " (ID:" + to_string(det.obj_id) + ")", Point(det.bbox.x, det.bbox.y - 10),
                    FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);
        }
        imshow("YOLO Real-Time Tracking [C++] (Press 'q' to exit)", frame);

        // Refresh object display window every REFRESH_INTERVAL seconds
        if (duration_cast<duration<double>>(now - last_panel_update).count() > REFRESH_INTERVAL) {
            // Prepare crops for slots
            vector<Mat> slot_crops;
            for (int idx = 0; idx < object_slots.size(); ++idx) {
                int obj_id = object_slots[idx];
                if (obj_id != -1 && object_dict.find(obj_id) != object_dict.end()) {
                    auto& info = object_dict[obj_id];
                    double appear_time = duration_cast<duration<double>>(now - info.first_seen).count();
                    if (appear_time >= APPEAR_THRESHOLD) {
                        Mat label_img = Mat::zeros(24, CROP_SIZE, CV_8UC3);
                        string text = info.name + " (ID:" + to_string(obj_id) + ")";
                        putText(label_img, text, Point(5, 18), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,255,255), 1);
                        Mat crop_with_label;
                        vconcat(label_img, info.crop, crop_with_label);
                        rectangle(crop_with_label, Point(0,24), Point(CROP_SIZE-1,CROP_SIZE+23),
                                  info.visible ? Scalar(0,255,0) : Scalar(0,165,255), 2);
                        slot_crops.push_back(crop_with_label);
                    } else {
                        slot_crops.push_back(Mat::zeros(CROP_SIZE+24, CROP_SIZE, CV_8UC3));
                    }
                } else {
                    slot_crops.push_back(Mat::zeros(CROP_SIZE+24, CROP_SIZE, CV_8UC3));
                }
            }
            // Only show rows with at least one object
            int last_nonempty = -1;
            for (int i = slot_crops.size() - 1; i >= 0; --i) {
                if (countNonZero(slot_crops[i].reshape(1)) > 0) {
                    last_nonempty = i;
                    break;
                }
            }
            int used_slots = last_nonempty + 1;
            int used_rows = (used_slots + OBJECTS_PER_ROW - 1) / OBJECTS_PER_ROW;
            if (used_rows == 0) used_rows = 1;
            vector<Mat> rows;
            for (int i = 0; i < used_rows; ++i) {
                vector<Mat> row_crops;
                for (int j = 0; j < OBJECTS_PER_ROW; ++j) {
                    int idx = i * OBJECTS_PER_ROW + j;
                    if (idx < slot_crops.size())
                        row_crops.push_back(slot_crops[idx]);
                    else
                        row_crops.push_back(Mat::zeros(CROP_SIZE+24, CROP_SIZE, CV_8UC3));
                }
                Mat row;
                hconcat(row_crops, row);
                rows.push_back(row);
            }
            Mat objects_panel;
            vconcat(rows, objects_panel);
            imshow("Objects", objects_panel);
            last_panel_update = now;
        }

        char key = (char)waitKey(1);
        if (key == 'q' || key == 'Q') break;
        frame_idx++;
    }

    cap.release();
    destroyAllWindows();
    cout << "Experiment finished." << endl;
    return 0;
}