"""물건을 마지막으로 본 장면 기억 로직.

카메라·모델 없이 동작하는 순수 로직만 모아둔 모듈.
vision_tracker.py(기록)와 voice_assistant.py(조회)가 같이 사용한다.
좌표는 모두 0~1로 정규화된 값 (YOLO의 xyxyn).

1인칭(몸에 단) 카메라라서 화면 속 위치는 고개만 돌려도 바뀐다.
그래서 위치는 "화면 어디"가 아니라 "같이 보인 물건·가구"와 마지막 프레임 사진으로 기억한다.
"""

import json
import math
import os
import time


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

PLACE_SECONDS = 2.0  # 이 시간 이상 계속 보이면 '봄'으로 확정 (한두 프레임 오탐 무시)
LOST_SECONDS = 3.0   # 이 시간 이상 안 보이면 '사라짐'
NEAR_COUNT = 3       # 근처 물건은 가까운 순으로 최대 3개


def box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def nearby_names(box, others):
    """같은 프레임에 보인 다른 물건·가구 이름을 화면상 가까운 순으로."""
    center = box_center(box)
    ranked = sorted(others, key=lambda other: math.dist(center, box_center(other[1])))
    return [name for name, _ in ranked][:NEAR_COUNT]


def make_event(class_id, item, kind, now):
    return {
        "time": now,
        "class_id": class_id,
        "name": item["name"],
        "event": kind,
        "near": item["near"],
    }


def update_items(items, detections, places, now):
    """한 프레임의 탐지 결과로 물건 상태를 갱신하고, 새로 생긴 이벤트 목록을 반환.

    items:      {class_id: 상태 dict}  (이 함수가 직접 수정)
    detections: {class_id: (x1, y1, x2, y2)}  물건 클래스당 박스 1개
    places:     [(가구 이름, (x1, y1, x2, y2)), ...]

    상태(status) 흐름:
        None(처음 봄) --2초 계속 보임--> seen ('봄')
        seen --3초 안 보임--> lost ('사라짐', 마지막으로 본 장면 유지)
        lost --다시 2초 계속 보임--> seen ('봄')
        None --3초 안 보임--> 삭제 (확정 전이라 기억할 필요 없음)
    """
    # ponytail: 클래스당 1개만 추적 (컵 2개면 하나로 취급). 같은 클래스 여러 개 필요하면 ID 추적(yolo.track) 도입.
    events = []
    everything = [(ITEM_NAMES[class_id], box) for class_id, box in detections.items()] + places

    for class_id, box in detections.items():
        item = items.get(class_id)

        if item is None:
            item = items[class_id] = {"name": ITEM_NAMES[class_id], "status": None, "since": now}
        elif now - item["last_seen"] > LOST_SECONDS:
            # 한동안 안 보이다 다시 보임 → 다시 PLACE_SECONDS 확인
            # (lost는 확정될 때까지 lost 유지: 잠깐 보였다 사라져도 '사라짐' 중복 안 남)
            item["since"] = now

        # 보일 때마다 최신 장면으로 갱신 → 사라지면 마지막으로 본 장면이 남음
        item["last_seen"] = now
        item["box"] = box
        item["near"] = nearby_names(box, [other for other in everything if other[0] != item["name"]])

        if item["status"] != "seen" and now - item["since"] >= PLACE_SECONDS:
            item["status"] = "seen"
            events.append(make_event(class_id, item, "봄", now))

    # 오래 안 보인 물건 처리
    for class_id, item in list(items.items()):
        if item["status"] == "lost" or now - item["last_seen"] <= LOST_SECONDS:
            continue

        if item["status"] is None:
            del items[class_id]
        else:
            item["status"] = "lost"
            events.append(make_event(class_id, item, "사라짐", now))

    return events


def replace_file(temp_path, path):
    """임시 파일을 실제 파일로 교체.
    Windows에선 OneDrive·백신이 방금 쓴 파일을 잠깐 잡고 있어서 PermissionError가 나므로 최대 1초 재시도."""
    for _ in range(10):
        try:
            os.replace(temp_path, path)
            return
        except PermissionError:
            time.sleep(0.1)

    os.replace(temp_path, path)


def save_memory(path, items, events):
    """확정된 물건 상태 + 최근 이벤트를 JSON으로 저장 (임시 파일 → os.replace로 원자적 교체)."""
    memory = {
        "items": [dict(item, class_id=class_id) for class_id, item in items.items() if item["status"] is not None],
        "events": list(events),
    }

    temp_path = path + ".tmp"

    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(memory, file, ensure_ascii=False, indent=2)

    replace_file(temp_path, path)


def ago_text(seconds):
    if seconds < 60:
        return "방금 전"
    if seconds < 3600:
        return f"{int(seconds // 60)}분 전"
    return f"{int(seconds // 3600)}시간 전"


def item_to_text(item, now):
    """물건 하나의 기억을 Gemma에게 줄 한 줄 문장으로 변환."""
    near = ", ".join(item["near"])

    if item["status"] == "seen":
        return f"{item['name']}: 지금 보고 있음" + (f" ({near} 근처)." if near else ".")

    where = f" {near} 근처에서" if near else ""
    return f"{item['name']}: {ago_text(now - item['last_seen'])}{where} 마지막으로 봄."
