import cv2
import sys

# --- 設定 ---
# 選擇一個追蹤器，取消註解你想使用的那一行
# tracker_type = 'CSRT'      # 推薦，準確率高
# tracker_type = 'KCF'       # 速度與準確率平衡
tracker_type = 'MIL'     # 速度極快

def create_tracker(tracker_type):
    """Creates a tracker based on the specified type, with error handling."""
    try:
        if tracker_type == 'CSRT':
            return cv2.TrackerCSRT_create()
        if tracker_type == 'KCF':
            return cv2.TrackerKCF_create()
        if tracker_type == 'MOSSE':
            return cv2.TrackerMOSSE_create()
        if tracker_type == 'MIL':
            return cv2.TrackerMIL_create()
        # ... 其他追蹤器
        print(f"錯誤：無效的追蹤器類型 '{tracker_type}'")
        return None
    except AttributeError:
        print("\n錯誤：您的 OpenCV 版本缺少追蹤器模組 (AttributeError)。")
        print("這通常是因為未安裝 'contrib' 版本的 OpenCV。")
        print("請嘗試執行以下指令來修復：")
        print("  pip uninstall opencv-python opencv-contrib-python")
        print("  pip install opencv-contrib-python\n")
        return None

# --- 主程式 ---
if __name__ == '__main__':
    # 建立追蹤器
    tracker = create_tracker(tracker_type)
    if not tracker:
        sys.exit()

    # 開啟攝影機
    video = cv2.VideoCapture(0)
    if not video.isOpened():
        print("無法開啟攝影機")
        sys.exit()

    # 讀取第一幀
    ok, frame = video.read()
    if not ok:
        print("無法讀取影像")
        sys.exit()

    # 讓使用者手動選擇要追蹤的物件 (ROI: Region of Interest)
    # 按下 Enter 或 Space 確認選取，按下 C 取消
    bbox = cv2.selectROI("Select Object to Track", frame, False)
    
    if not bbox or bbox[2] == 0 or bbox[3] == 0:
        print("未選取物件，程式結束。")
        sys.exit()

    # 初始化追蹤器
    ok = tracker.init(frame, bbox)
    cv2.destroyWindow("Select Object to Track")

    while True:
        # 讀取新的一幀
        ok, frame = video.read()
        if not ok:
            break

        # 開始計時
        timer = cv2.getTickCount()

        # 更新追蹤器
        ok, bbox = tracker.update(frame)

        # 計算 FPS
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        # 繪製邊界框
        if ok:
            # 追蹤成功
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (0, 255, 0), 2, 1)
            status_text = "Tracking"
            status_color = (0, 255, 0)
        else:
            # 追蹤失敗
            status_text = "Tracking Failure"
            status_color = (0, 0, 255)

        # 顯示狀態和 FPS
        cv2.putText(frame, tracker_type + " Tracker", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 170, 50), 2)
        cv2.putText(frame, status_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 顯示結果
        cv2.imshow("Tracking", frame)

        # 按下 'q' 鍵結束
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()
