"""item_memory 로직 확인용. 카메라 없이 실행: python test_item_memory.py"""

from item_memory import memory_to_text, update_items


def kinds(events):
    return [event["event"] for event in events]


desk = [("책상", (0.0, 0.5, 0.5, 1.0))]
phone_left = (0.1, 0.7, 0.2, 0.8)    # 화면 왼쪽 아래, 책상 안
phone_right = (0.8, 0.7, 0.9, 0.8)   # 화면 오른쪽 아래, 가구 없음

# 1) 기본 흐름: 놓임 → 이동 → 사라짐 → 다시 보임
items = {}

# 처음 보이면 바로 확정하지 않고, 2초 이상 같은 자리에 있어야 '놓임'
assert update_items(items, {67: phone_left}, desk, now=0) == []
events = update_items(items, {67: phone_left}, desk, now=2.5)
assert kinds(events) == ["놓임"]
assert events[0]["position"] == "화면 왼쪽 아래" and events[0]["near"] == "책상"

# 1초 깜빡임(미탐지)은 무시
assert update_items(items, {}, desk, now=3.5) == []
assert update_items(items, {67: phone_left}, desk, now=4) == []

# 오른쪽으로 옮기면 2초 뒤 '이동'
assert update_items(items, {67: phone_right}, [], now=5) == []
assert items[67]["status"] == "moving"
events = update_items(items, {67: phone_right}, [], now=7.5)
assert kinds(events) == ["이동"] and events[0]["position"] == "화면 오른쪽 아래" and events[0]["near"] is None

# 3초 넘게 안 보이면 '사라짐', 같은 자리에 다시 2초 있으면 '다시 보임'
assert kinds(update_items(items, {}, [], now=11)) == ["사라짐"]
assert update_items(items, {67: phone_right}, [], now=12) == []
assert kinds(update_items(items, {67: phone_right}, [], now=14.5)) == ["다시 보임"]

# Gemma에게 줄 텍스트
memory = {"items": [dict(item, class_id=class_id) for class_id, item in items.items()]}
assert memory_to_text(memory, now=14.5 + 120) == "휴대폰: 화면 오른쪽 아래에 있음. 2분 전 놓임."
assert memory_to_text({"items": []}, now=0) == "아직 기억한 물건이 없습니다."

# 2) 사라진 물건이 잠깐(2초 미만) 보였다 다시 사라지면 '사라짐'을 또 남기지 않음 (노트북 실험에서 발견한 버그)
items = {}
update_items(items, {67: phone_left}, desk, now=0)
assert kinds(update_items(items, {67: phone_left}, desk, now=2.5)) == ["놓임"]
assert kinds(update_items(items, {}, desk, now=6)) == ["사라짐"]
assert update_items(items, {67: phone_left}, desk, now=7) == []
assert update_items(items, {}, desk, now=11) == []
assert items[67]["status"] == "lost"

# 3) 놓인 물건이 잠깐 크게 움직였다가(손에 가려 bbox 흔들림 등) 제자리로 오면 이벤트 없음 (노트북 실험 2차)
items = {}
update_items(items, {67: phone_left}, desk, now=0)
assert kinds(update_items(items, {67: phone_left}, desk, now=2.5)) == ["놓임"]
assert update_items(items, {67: phone_right}, desk, now=3) == []
assert update_items(items, {67: phone_left}, desk, now=3.5) == []
assert update_items(items, {67: phone_left}, desk, now=6) == []
assert items[67]["status"] == "placed" and items[67]["placed_time"] == 2.5

# 4) 확정 전에 사라진 물건은 기억하지 않음
items = {}
update_items(items, {41: (0.4, 0.4, 0.5, 0.5)}, [], now=0)
update_items(items, {}, [], now=4)
assert 41 not in items

print("OK")
