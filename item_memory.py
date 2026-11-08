"""물건 위치 기억 로직.

카메라·모델 없이 동작하는 순수 로직만 모아둔 모듈.
vision_tracker.py(기록)와 voice_assistant.py(조회)가 같이 사용한다.
좌표는 모두 0~1로 정규화된 값 (YOLO의 xyxyn).
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

# 위치 설명용 가구: 물건 중심이 이 bbox 안에 있으면 "책상 근처"
PLACE_NAMES = {
    56: "의자",
    57: "소파",
    59: "침대",
    60: "책상",
    62: "TV",
}

PLACE_SECONDS = 2.0   # 같은 자리에서 이 시간 이상 보이면 위치 확정
LOST_SECONDS = 3.0    # 이 시간 이상 안 보이면 '사라짐'
MOVE_DISTANCE = 0.15  # 중심이 화면 크기 대비 이만큼 넘게 움직이면 '움직이는 중'


def box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def position_text(center):
    """화면을 3x3으로 나눠서 위치를 말로 표현."""
    x, y = center
    horizontal = "왼쪽" if x < 1 / 3 else "오른쪽" if x > 2 / 3 else "가운데"
    vertical = "위" if y < 1 / 3 else "아래" if y > 2 / 3 else "가운데"

    if horizontal == "가운데" and vertical == "가운데":
        return "화면 가운데"
    if vertical == "가운데":
        return f"화면 {horizontal}"
    if horizontal == "가운데":
        return f"화면 {vertical}쪽"
    return f"화면 {horizontal} {vertical}"


def place_of(center, places):
    """물건 중심을 포함하는 가구 중 가장 작은 것의 이름. 없으면 None."""
    x, y = center
    inside = []

    for name, (x1, y1, x2, y2) in places:
        if x1 <= x <= x2 and y1 <= y <= y2:
            inside.append(((x2 - x1) * (y2 - y1), name))

    return min(inside)[1] if inside else None


def make_event(class_id, item, kind, now):
    return {
        "time": now,
        "class_id": class_id,
        "name": item["name"],
        "event": kind,
        "position": item.get("position"),
        "near": item.get("near"),
    }


def update_items(items, detections, places, now):
    """한 프레임의 탐지 결과로 물건 상태를 갱신하고, 새로 생긴 이벤트 목록을 반환.

    items:      {class_id: 상태 dict}  (이 함수가 직접 수정)
    detections: {class_id: (x1, y1, x2, y2)}  물건 클래스당 박스 1개
    places:     [(가구 이름, (x1, y1, x2, y2)), ...]

    상태(status) 흐름:
        None(처음 봄) --2초 고정--> placed --크게 움직임--> moving --2초 고정--> placed
        placed/moving --3초 안 보임--> lost --다시 2초 고정--> placed ('이동' 또는 '다시 보임')
        None --3초 안 보임--> 삭제 (확정 전이라 기억할 필요 없음)
    """
    # ponytail: 클래스당 1개만 추적 (컵 2개면 하나로 취급). 같은 클래스 여러 개 필요하면 ID 추적(yolo.track) 도입.
    events = []

    for class_id, box in detections.items():
        center = box_center(box)
        item = items.get(class_id)

        if item is None:
            item = items[class_id] = {
                "name": ITEM_NAMES[class_id],
                "status": None,
                "since": now,
                "candidate": center,
            }
        elif now - item["last_seen"] > LOST_SECONDS or math.dist(center, item["candidate"]) > MOVE_DISTANCE:
            # 사라졌다가 다시 보이거나 크게 움직임 → 이 자리에서 다시 PLACE_SECONDS 확인
            # (lost는 확정될 때까지 lost 유지: 잠깐 보였다 사라져도 '사라짐' 중복 안 남)
            item.update(since=now, candidate=center)
            if item["status"] == "placed":
                item["status"] = "moving"

        item["last_seen"] = now
        item["box"] = box

        # 같은 자리에 PLACE_SECONDS 이상 있으면 위치 확정
        if item["status"] != "placed" and now - item["since"] >= PLACE_SECONDS:
            if item["status"] is None:
                kind = "놓임"
            elif math.dist(center, item["center"]) > MOVE_DISTANCE:
                kind = "이동"
            elif item["status"] == "lost":
                kind = "다시 보임"
            else:
                kind = None  # 잠깐 움직였다가 제자리 → 바뀐 것 없음

            item["status"] = "placed"

            if kind:
                item.update(
                    center=center,
                    position=position_text(center),
                    near=place_of(center, places),
                    placed_time=now,
                )
                events.append(make_event(class_id, item, kind, now))

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


def memory_to_text(memory, now):
    """저장된 기억을 Gemma에게 줄 자연어 목록으로 변환."""
    if not memory["items"]:
        return "아직 기억한 물건이 없습니다."

    lines = []

    for item in memory["items"]:
        where = item["position"]
        if item["near"]:
            where += f" {item['near']} 근처"

        if item["status"] == "lost":
            lines.append(f"{item['name']}: 지금은 보이지 않음. {ago_text(now - item['last_seen'])} {where}에서 마지막으로 봄.")
        elif item["status"] == "moving":
            lines.append(f"{item['name']}: 지금 옮겨지는 중. 이전 위치는 {where}.")
        else:
            lines.append(f"{item['name']}: {where}에 있음. {ago_text(now - item['placed_time'])} 놓임.")

    return "\n".join(lines)
