"""Detect items with a first-person camera and remember the 'last scene they were seen in'.

- When a target item (phone, mouse) is visible it is 'seen'; when not visible for 3+ seconds it is 'lost'
- The camera is body-worn, so screen coordinates change whenever the head turns.
  So location is remembered by 'nearby items/furniture' and the last frame photo instead of coordinates.
- Output: item_memory.json and last_frame_<class_id>.jpg in where-did-i-put-it/output/

Run (from this folder): .venv/bin/python vision_tracker.py
"""

import json
import math
import os
import time

import cv2
from ultralytics import YOLO


# All paths are relative to this file's folder → only this folder's models and outputs are used, wherever it runs from
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

YOLO_MODEL_PATH = os.path.join(BASE_DIR, "src/models/YOLO/yolo11n_int8.engine")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MEMORY_PATH = os.path.join(OUTPUT_DIR, "item_memory.json")

# USB camera (the v4l2src pipeline used in class)
CAMERA_PIPELINE = (
    "v4l2src device=/dev/video0 ! "
    "video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! "
    "videoconvert ! video/x-raw,format=BGR ! "
    "queue leaky=downstream max-size-buffers=1 "
    "max-size-bytes=0 max-size-time=0 ! "
    "appsink drop=true max-buffers=1 sync=false"
)

# Target items: COCO class id → Korean name (Korean because Gemma answers and Piper speaks in Korean)
ITEM_NAMES = {
    64: "마우스",  # mouse
    67: "휴대폰",  # cell phone
}

# Items/furniture used to describe location: if seen together, "near the keyboard", "near the desk"
LANDMARK_NAMES = {
    39: "병",      # bottle
    41: "컵",      # cup
    56: "의자",    # chair
    57: "소파",    # couch
    59: "침대",    # bed
    60: "책상",    # dining table (desk)
    62: "TV",
    63: "노트북",  # laptop
    66: "키보드",  # keyboard
    73: "책",      # book
}

CONFIDENCE = 0.3    # ignore detections below this (first-person video is blurry with steep angles, so the phone often scores 0.3–0.5)
LOST_SECONDS = 3.0  # not visible for this long → 'lost' (brief occlusion or flicker keeps the state)
NEAR_COUNT = 3      # up to 3 nearby objects, closest first


def detect(yolo, frame):
    """YOLO detection → target items {class_id: box} (one per class), surroundings [(name, box)]. Coordinates normalized to 0–1."""
    result = yolo.predict(
        source=frame,
        conf=CONFIDENCE,
        iou=0.5,
        classes=list(ITEM_NAMES) + list(LANDMARK_NAMES),
        verbose=False,
    )[0]

    detections = {}
    landmarks = []

    for box in result.boxes:  # sorted by confidence descending, so the first box per class is the most confident
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
    """Names of other items/furniture seen in the same frame, closest first."""
    center = box_center(box)
    others = sorted(others, key=lambda other: math.dist(center, box_center(other[1])))
    names = dict.fromkeys(name for name, _ in others)  # drop duplicate names (2 keyboards → keyboard), keep order
    return list(names)[:NEAR_COUNT]


def update_items(items, detections, landmarks, now):
    """Update item states from one frame's detections and return new events [(class_id, 'seen'|'lost')].

    status: lost --visible--> seen('seen') --not visible for 3 s--> lost('lost')
    """
    # ponytail: tracks only one item per class (2 phones are treated as one). Use yolo.track IDs if multiple are needed.
    events = []
    everything = [(ITEM_NAMES[class_id], box) for class_id, box in detections.items()] + landmarks

    for class_id, box in detections.items():
        item = items.setdefault(class_id, {"name": ITEM_NAMES[class_id], "status": "lost"})

        # Overwrite with the latest scene every time it is visible → when it disappears, the last scene remains
        item["last_seen"] = now
        item["box"] = box
        item["near"] = nearby_names(box, [other for other in everything if other[0] != item["name"]])

        if item["status"] == "lost":
            item["status"] = "seen"
            events.append((class_id, "seen"))

    for class_id, item in items.items():
        if item["status"] == "seen" and now - item["last_seen"] > LOST_SECONDS:
            item["status"] = "lost"
            events.append((class_id, "lost"))

    return events


def save_json(path, data):
    """Write to a temp file and swap in with os.replace → voice_assistant never reads a half-written file."""
    temp_path = path + ".tmp"

    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

    os.replace(temp_path, path)


def save_frame(frame, class_id, box):
    """Save the frame with a green box around the item (the photo voice_assistant passes to Gemma multimodal)."""
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
        print("Cannot open the camera.")
        return

    items = {}
    last_frames = {}  # last frame each item was visible in (this scene is saved on 'lost')

    print("q : quit")

    while True:
        ret, frame = cap.read()

        if not ret:
            # Body-worn camera, so USB may drop briefly → reconnect instead of quitting
            print("No camera frame, reconnecting.")
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
            print(f"[{kind}] {item['name']} - near: {', '.join(item['near']) or 'nothing'}")
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
