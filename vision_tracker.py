"""1인칭 카메라로 물건을 탐지해서 '마지막으로 본 장면'을 기억한다.

- 물건이 2초 이상 계속 보이면 '봄', 3초 이상 안 보이면 '사라짐'
- 몸에 단 카메라라 고개만 돌려도 화면 좌표가 바뀐다.
  그래서 위치는 좌표 대신 '같이 보인 물건·가구'와 마지막 프레임 사진으로 기억한다.
- 결과: where-did-i-put-it/output/ 안에 item_memory.json, last_frame_<class_id>.jpg

실행 (~/vision-llm 에서): python where-did-i-put-it/vision_tracker.py
"""

import json
import math
import os
import time
from collections import deque

import cv2
from ultralytics import YOLO


YOLO_MODEL_PATH = "src/models/YOLO/yolo11n_int8.engine"
# 결과는 이 파일 옆 output/ 폴더에 저장 (실행 위치와 상관없이)
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
MEMORY_PATH = os.path.join(OUTPUT_DIR, "item_memory.json")

# USB 카메라 (수업 때 쓰던 v4l2src 파이프라인)
CAMERA_PIPELINE = (
    "v4l2src device=/dev/video0 ! "
    "video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! "
    "videoconvert ! video/x-raw,format=BGR ! "
    "queue leaky=downstream max-size-buffers=1 "
    "max-size-bytes=0 max-size-time=0 ! "
    "appsink drop=true max-buffers=1 sync=false"
)

# 찾을 물건: COCO class id → 한국어 이름
ITEM_NAMES = {
    24: "백팩",
    25: "우산",
    26: "핸드백",
    39: "병",
    41: "컵",
    63: "노트북",
    64: "마우스",
    65: "리모컨",
    66: "키보드",
    67: "휴대폰",
    73: "책",
    76: "가위",
}

# 위치 설명용 가구: 같이 보이면 "책상 근처"
PLACE_NAMES = {
    56: "의자",
    57: "소파",
    59: "침대",
    60: "책상",
    62: "TV",
}

SEEN_SECONDS = 2.0  # 이 시간 이상 계속 보이면 '봄' (한두 프레임 오탐 무시)
LOST_SECONDS = 3.0  # 이 시간 이상 안 보이면 '사라짐'
NEAR_COUNT = 3      # 근처 물건은 가까운 순으로 최대 3개


def detect(yolo, frame):
    """YOLO 탐지 → 물건 {class_id: 박스}(클래스당 1개), 가구 [(이름, 박스)]. 좌표는 0~1 정규화."""
    result = yolo.predict(
        source=frame,
        conf=0.25,
        iou=0.5,
        classes=list(ITEM_NAMES) + list(PLACE_NAMES),
        verbose=False,
    )[0]

    detections = {}
    places = []

    for box in result.boxes:  # confidence 내림차순이라 클래스별 첫 박스가 가장 확실한 것
        class_id = int(box.cls[0])
        xyxy = tuple(box.xyxyn[0].cpu().tolist())

        if class_id in ITEM_NAMES:
            detections.setdefault(class_id, xyxy)
        else:
            places.append((PLACE_NAMES[class_id], xyxy))

    return result, detections, places


def box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def nearby_names(box, others):
    """같은 프레임에 보인 다른 물건·가구 이름을 가까운 순으로."""
    center = box_center(box)
    others = sorted(others, key=lambda other: math.dist(center, box_center(other[1])))
    return [name for name, _ in others[:NEAR_COUNT]]


def update_items(items, detections, places, now):
    """한 프레임 탐지 결과로 물건 상태를 갱신하고, 새 이벤트 [(class_id, '봄'|'사라짐')]를 반환.

    status: None(후보) --2초 계속 보임--> seen('봄') --3초 안 보임--> lost('사라짐') --2초 보임--> seen
    후보가 3초 안 보이면 삭제 (잠깐 잘못 탐지된 물건은 기억하지 않음)
    """
    # ponytail: 클래스당 1개만 추적 (컵 2개면 하나로 취급). 여러 개가 필요하면 yolo.track ID 사용.
    events = []
    everything = [(ITEM_NAMES[class_id], box) for class_id, box in detections.items()] + places

    for class_id, box in detections.items():
        item = items.setdefault(class_id, {"name": ITEM_NAMES[class_id], "status": None, "since": now, "last_seen": now})

        # 한동안 안 보이다 다시 보임 → 다시 2초 확인
        # (lost는 2초 확정 전까지 lost 유지 → 잠깐 보였다 사라져도 '사라짐'이 중복으로 안 남음)
        if now - item["last_seen"] > LOST_SECONDS:
            item["since"] = now

        # 보일 때마다 최신 장면으로 덮어씀 → 사라지면 마지막으로 본 장면이 남음
        item["last_seen"] = now
        item["box"] = box
        item["near"] = nearby_names(box, [other for other in everything if other[0] != item["name"]])

        if item["status"] != "seen" and now - item["since"] >= SEEN_SECONDS:
            item["status"] = "seen"
            events.append((class_id, "봄"))

    for class_id, item in list(items.items()):
        if item["status"] == "lost" or now - item["last_seen"] <= LOST_SECONDS:
            continue

        if item["status"] is None:
            del items[class_id]
        else:
            item["status"] = "lost"
            events.append((class_id, "사라짐"))

    return events


def save_json(path, data):
    """임시 파일에 쓰고 os.replace로 교체 → voice_assistant가 반쯤 쓰인 파일을 읽지 않음."""
    temp_path = path + ".tmp"

    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

    os.replace(temp_path, path)


def save_frame(frame, class_id, box):
    """물건에 초록 박스를 그린 프레임 저장 (voice_assistant가 Gemma 멀티모달에 넘길 사진)."""
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box

    image = frame.copy()
    cv2.rectangle(image, (int(x1 * width), int(y1 * height)), (int(x2 * width), int(y2 * height)), (0, 255, 0), 3)

    temp_path = os.path.join(OUTPUT_DIR, f"tmp_{class_id}.jpg")
    cv2.imwrite(temp_path, image)
    os.replace(temp_path, os.path.join(OUTPUT_DIR, f"last_frame_{class_id}.jpg"))


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    yolo = YOLO(YOLO_MODEL_PATH)
    cap = cv2.VideoCapture(CAMERA_PIPELINE, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    items = {}
    last_frames = {}              # 물건별로 마지막으로 보인 프레임 ('사라짐' 때 이 장면을 저장)
    event_log = deque(maxlen=20)  # 최근 이벤트 20개만 유지

    print("q : 종료")

    while True:
        ret, frame = cap.read()

        if not ret:
            # 몸에 단 카메라라 USB가 잠깐 끊길 수 있음 → 종료하지 않고 다시 연결
            print("카메라 프레임 없음, 다시 연결합니다.")
            cap.release()
            time.sleep(1)
            cap = cv2.VideoCapture(CAMERA_PIPELINE, cv2.CAP_GSTREAMER)
            continue

        result, detections, places = detect(yolo, frame)

        for class_id in detections:
            last_frames[class_id] = frame

        now = time.time()
        events = update_items(items, detections, places, now)

        for class_id, kind in events:
            item = items[class_id]
            print(f"[{kind}] {item['name']} - 근처: {', '.join(item['near']) or '없음'}")

            event_log.append({"time": now, "name": item["name"], "event": kind, "near": item["near"]})
            save_frame(last_frames[class_id], class_id, item["box"])

        if events:
            save_json(MEMORY_PATH, {
                "items": [dict(item, class_id=class_id) for class_id, item in items.items() if item["status"]],
                "events": list(event_log),
            })

        cv2.imshow("where-did-i-put-it", result.plot())

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
