"""item_memory 로직 확인용. 카메라 없이 실행: python test_item_memory.py"""

from item_memory import item_to_text, update_items


def kinds(events):
    return [event["event"] for event in events]


desk = ("책상", (0.0, 0.5, 1.0, 1.0))
mouse = (0.1, 0.1, 0.2, 0.2)
phone = (0.4, 0.4, 0.5, 0.5)

# 1) 기본 흐름: 봄 → 사라짐(마지막 장면 유지) → 다시 봄
items = {}

# 처음 보이면 바로 확정하지 않고, 2초 이상 계속 보여야 '봄'
assert update_items(items, {67: phone, 64: mouse}, [desk], now=0) == []
assert kinds(update_items(items, {67: phone, 64: mouse}, [desk], now=2.5)) == ["봄", "봄"]
assert items[67]["near"] == ["책상", "마우스"]  # 가까운 순

# 1인칭이라 고개를 돌려 화면 위치가 바뀌어도 이벤트 없음
assert update_items(items, {67: (0.8, 0.8, 0.9, 0.9)}, [], now=3) == []

# 마우스는 2.5초 이후 안 보임 → 3초 넘으면 '사라짐'
assert kinds(update_items(items, {67: phone}, [desk], now=6)) == ["사라짐"]
assert items[64]["status"] == "lost"

# 휴대폰도 사라짐. 마지막으로 본 장면(책상 근처)이 남음
assert kinds(update_items(items, {}, [], now=9.5)) == ["사라짐"]
assert items[67]["near"] == ["책상"]
assert item_to_text(items[67], now=6 + 120) == "휴대폰: 2분 전 책상 근처에서 마지막으로 봄."

# 다시 2초 보이면 '봄'
assert update_items(items, {67: phone}, [], now=20) == []
assert kinds(update_items(items, {67: phone}, [], now=22.5)) == ["봄"]
assert item_to_text(items[67], now=23) == "휴대폰: 지금 보고 있음."

# 2) 사라진 물건이 잠깐(2초 미만) 보였다 다시 사라지면 '사라짐'을 또 남기지 않음 (노트북 실험에서 발견한 버그)
items = {}
update_items(items, {67: phone}, [], now=0)
assert kinds(update_items(items, {67: phone}, [], now=2.5)) == ["봄"]
assert kinds(update_items(items, {}, [], now=6)) == ["사라짐"]
assert update_items(items, {67: phone}, [], now=7) == []
assert update_items(items, {}, [], now=11) == []
assert items[67]["status"] == "lost"

# 3) 확정 전에 사라진 물건은 기억하지 않음
items = {}
update_items(items, {41: (0.4, 0.4, 0.5, 0.5)}, [], now=0)
update_items(items, {}, [], now=4)
assert 41 not in items

print("OK")
