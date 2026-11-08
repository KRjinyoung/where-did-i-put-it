"""1인칭 카메라로 물건을 탐지해서 이벤트(봄/사라짐)와 마지막으로 본 장면을 기록.

실행 (~/vision-llm 에서): python where-did-i-put-it/vision_tracker.py
"""

import os
import sys
import time
from collections import deque

import cv2
from ultralytics import YOLO

from item_memory import ITEM_NAMES, PLACE_NAMES, replace_file, save_memory, update_items


# Jetson엔 TensorRT INT8 엔진이 있고, 노트북엔 없으니 일반 .pt 사용 (없으면 ultralytics가 자동 다운로드)
ENGINE_PATH = "src/models/YOLO/yolo11n_int8.engine"
YOLO_MODEL_PATH = ENGINE_PATH if os.path.exists(ENGINE_PATH) else "yolo11n.pt"

MEMORY_PATH = "src/json/item_memory.json"
FRAME_DIR = "src/output"

# Jetson USB 카메라 (수업 때 쓰던 v4l2src 파이프라인)
USB_PIPELINE = (
    "v4l2src device=/dev/video0 ! "
    "video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! "
    "videoconvert ! video/x-raw,format=BGR ! "
    "queue leaky=downstream max-size-buffers=1 "
    "max-size-bytes=0 max-size-time=0 ! "
    "appsink drop=true max-buffers=1 sync=false"
)


def save_frame(frame, class_id, box):
    """물건 bbox를 그린 전체 프레임 저장 (나중에 Gemma 멀티모달에 넘길 이미지)."""
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box

    image = frame.copy()
    cv2.rectangle(image, (int(x1 * width), int(y1 * height)), (int(x2 * width), int(y2 * height)), (0, 255, 0), 3)

    path = os.path.join(FRAME_DIR, f"last_frame_{class_id}.jpg")
    temp_path = os.path.join(FRAME_DIR, f"last_frame_{class_id}_tmp.jpg")

    if cv2.imwrite(temp_path, image):
        replace_file(temp_path, path)


def open_camera():
    if sys.platform == "linux":
        return cv2.VideoCapture(USB_PIPELINE, cv2.CAP_GSTREAMER)
    return cv2.VideoCapture(0)  # 노트북(Windows) 내장 카메라


def main():
    os.makedirs(os.path.dirname(MEMORY_PATH), exist_ok=True)
    os.makedirs(FRAME_DIR, exist_ok=True)

    yolo = YOLO(YOLO_MODEL_PATH)
    cap = open_camera()

    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    items = {}
    last_frames = {}  # 물건별로 마지막으로 보인 프레임 ('사라짐' 때 이 장면을 저장)
    event_log = deque(maxlen=20)  # 최근 이벤트 20개만 유지

    print("q : 종료")

    while True:
        ret, frame = cap.read()

        if not ret:
            # 몸에 단 카메라라 USB가 잠깐 끊길 수 있음 → 종료하지 않고 다시 연결
            print("카메라 프레임 없음, 다시 연결합니다.")
            cap.release()
            time.sleep(1)
            cap = open_camera()
            continue

        # 1. 물건 + 가구 탐지
        result = yolo.predict(
            source=frame,
            conf=0.25,
            iou=0.5,
            classes=list(ITEM_NAMES) + list(PLACE_NAMES),
            verbose=False,
        )[0]

        # 2. 물건(클래스당 가장 확신도 높은 1개)과 가구로 분리, 좌표는 0~1 정규화
        detections = {}
        places = []

        for box in result.boxes:  # NMS 결과는 confidence 내림차순
            class_id = int(box.cls[0].item())
            xyxy = tuple(box.xyxyn[0].cpu().tolist())

            if class_id in ITEM_NAMES:
                detections.setdefault(class_id, xyxy)
                last_frames[class_id] = frame
            else:
                places.append((PLACE_NAMES[class_id], xyxy))

        # 3. 상태 갱신 + 이벤트 감지
        events = update_items(items, detections, places, time.time())

        # 4. 이벤트가 있을 때만 저장
        for event in events:
            print(f"[{event['event']}] {event['name']} - 근처: {', '.join(event['near']) or '없음'}")
            event_log.append(event)

            # 봄: 지금 프레임, 사라짐: 마지막으로 보인 프레임
            save_frame(last_frames[event["class_id"]], event["class_id"], items[event["class_id"]]["box"])

        if events:
            save_memory(MEMORY_PATH, items, event_log)

        cv2.imshow("where-did-i-put-it", result.plot())

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
