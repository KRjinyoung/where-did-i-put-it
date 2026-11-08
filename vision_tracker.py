"""1인칭 카메라로 물건을 탐지해서 '마지막으로 본 장면'을 기억한다.

- 찾을 물건(휴대폰, 마우스)이 보이면 '봄', 3초 이상 안 보이면 '사라짐'
- 몸에 단 카메라라 고개만 돌려도 화면 좌표가 바뀐다.
  그래서 위치는 좌표 대신 '같이 보인 물건·가구'와 마지막 프레임 사진으로 기억한다.
- 결과: where-did-i-put-it/output/ 안에 item_memory.json, last_frame_<class_id>.jpg

실행 (이 폴더에서): .venv/bin/python vision_tracker.py
"""

import json
import math
import os
import time

import cv2
from ultralytics import YOLO


# 모든 경로는 이 파일이 있는 폴더 기준 → 어디서 실행해도 이 폴더 안의 모델·결과만 사용
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

YOLO_MODEL_PATH = os.path.join(BASE_DIR, "src/models/YOLO/yolo11n_int8.engine")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
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
    64: "마우스",
    67: "휴대폰",
}

# 위치 설명용 물건·가구: 같이 보이면 "키보드 근처", "책상 근처"
LANDMARK_NAMES = {
    39: "병",
    41: "컵",
    56: "의자",
    57: "소파",
    59: "침대",
    60: "책상",
    62: "TV",
    63: "노트북",
    66: "키보드",
    73: "책",
}

CONFIDENCE = 0.3    # 이보다 확신이 낮은 탐지는 무시 (1인칭 영상은 흐리고 각도가 심해서 휴대폰이 0.3~0.5로 잡히는 경우가 많음)
LOST_SECONDS = 3.0  # 이 시간 이상 안 보이면 '사라짐' (잠깐 가려지거나 깜빡여도 유지)
NEAR_COUNT = 3      # 근처 물건은 가까운 순으로 최대 3개


def detect(yolo, frame):
    """YOLO 탐지 → 찾을 물건 {class_id: 박스}(클래스당 1개), 주변 [(이름, 박스)]. 좌표는 0~1 정규화."""
    result = yolo.predict(
        source=frame,
        conf=CONFIDENCE,
        iou=0.5,
        classes=list(ITEM_NAMES) + list(LANDMARK_NAMES),
        verbose=False,
    )[0]

    detections = {}
    landmarks = []

    for box in result.boxes:  # confidence 내림차순이라 클래스별 첫 박스가 가장 확실한 것
        class_id = int(box.cls[0])
        xyxy = tuple(box.xyxyn[0].cpu().tolist())

        if class_id in ITEM_NAMES:
            detections.setdefault(class_id, xyxy)
        else:
            landmarks.append((LANDMARK_NAMES[class_id], xyxy))

    return result, detections, landmarks


def box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def nearby_names(box, others):
    """같은 프레임에 보인 다른 물건·가구 이름을 가까운 순으로."""
    center = box_center(box)
    others = sorted(others, key=lambda other: math.dist(center, box_center(other[1])))
    names = dict.fromkeys(name for name, _ in others)  # 같은 이름 중복 제거 (키보드 2개 → 키보드), 순서 유지
    return list(names)[:NEAR_COUNT]


def update_items(items, detections, landmarks, now):
    """한 프레임 탐지 결과로 물건 상태를 갱신하고, 새 이벤트 [(class_id, '봄'|'사라짐')]를 반환.

    status: lost --보임--> seen('봄') --3초 안 보임--> lost('사라짐')
    """
    # ponytail: 클래스당 1개만 추적 (휴대폰 2개면 하나로 취급). 여러 개가 필요하면 yolo.track ID 사용.
    events = []
    everything = [(ITEM_NAMES[class_id], box) for class_id, box in detections.items()] + landmarks

    for class_id, box in detections.items():
        item = items.setdefault(class_id, {"name": ITEM_NAMES[class_id], "status": "lost"})

        # 보일 때마다 최신 장면으로 덮어씀 → 사라지면 마지막으로 본 장면이 남음
        item["last_seen"] = now
        item["box"] = box
        item["near"] = nearby_names(box, [other for other in everything if other[0] != item["name"]])

        if item["status"] == "lost":
            item["status"] = "seen"
            events.append((class_id, "봄"))

    for class_id, item in items.items():
        if item["status"] == "seen" and now - item["last_seen"] > LOST_SECONDS:
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
    last_frames = {}  # 물건별로 마지막으로 보인 프레임 ('사라짐' 때 이 장면을 저장)

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

        result, detections, landmarks = detect(yolo, frame)

        for class_id in detections:
            last_frames[class_id] = frame

        now = time.time()
        events = update_items(items, detections, landmarks, now)

        for class_id, kind in events:
            item = items[class_id]
            print(f"[{kind}] {item['name']} - 근처: {', '.join(item['near']) or '없음'}")
            save_frame(last_frames[class_id], class_id, item["box"])

        if events:
            save_json(MEMORY_PATH, {"items": [dict(item, class_id=class_id) for class_id, item in items.items()]})

        cv2.imshow("where-did-i-put-it", result.plot())

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
