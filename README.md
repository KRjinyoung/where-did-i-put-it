# where-did-i-put-it

Jetson Orin 온디바이스 "어디 뒀지?" 음성 물건 찾기 도우미.
몸에 단(1인칭) 카메라로 물건을 마지막으로 본 장면을 기억하고, 물어보면 Whisper → Gemma 4 멀티모달 → Piper로 음성 답변한다. 인터넷 없이 완전 로컬에서 동작.

가천대 「Physical AI의 Vision-LLM 융합 시청각 멀티모달 시스템」 최종 프로젝트 (Project03). 초기 설계는 [PLAN.md](PLAN.md).

## 파이프라인

```
[vision_tracker.py]  카메라 → YOLO11n INT8 → 물건별 '봄/사라짐' + 같이 보인 물건·가구 → src/json/item_memory.json
                                                                                 + src/output/last_frame_<id>.jpg
[voice_assistant.py] Enter → STT(Whisper) → Gemma(찾는 물건 고르기) → Gemma 멀티모달(기억 + 마지막 장면 사진) → TTS(Piper)
```

- 2초 이상 계속 보이면 `봄`, 3초 이상 안 보이면 `사라짐` (YOLO 깜빡임·오탐 무시)
- 1인칭 카메라는 고개만 돌려도 화면 좌표가 바뀌므로, 위치는 좌표 대신 같이 보인 물건·가구와 마지막 프레임으로 기억
- 기억에 없는 물건은 Gemma를 거치지 않고 "본 적이 없어요"로 답해서 위치를 지어내지 않음

## 파일

| 파일 | 역할 |
|---|---|
| `vision_tracker.py` | 카메라 → YOLO → 물건 상태·이벤트 판단 → 기억 JSON, 마지막 장면 사진 저장 |
| `voice_assistant.py` | 질문(음성 또는 글자) → 기억 JSON + 사진 → Gemma → 음성 답변 |

두 프로그램은 `item_memory.json` 파일로만 연결된다 (트래커는 계속 기록, 도우미는 질문할 때만 읽음).

## 실행 방법

수업 환경(`~/vision-llm`)에 이미 있는 것: `.venv`(ultralytics, llama-cpp-python), `src/models/YOLO/yolo11n_int8.engine`, `src/models/Gemma4/*.gguf`(모델 + mmproj), `whisper.cpp` 빌드, `.piper_venv` + `src/models/Piper/ko_KR-kss-medium.onnx`.

```bash
cd ~/vision-llm
git clone https://github.com/KRjinyoung/where-did-i-put-it.git

# 터미널 1: 카메라 기록 (q 종료)
.venv/bin/python where-did-i-put-it/vision_tracker.py

# 터미널 2: 질문 (Enter 후 5초 동안 말하기, 글자 입력도 가능, q 종료)
.venv/bin/python where-did-i-put-it/voice_assistant.py
```

참고
- 마이크는 `voice_assistant.py`의 `MIC_DEVICE`에 이름으로 지정 (`arecord -l`로 확인)
- Gemma + mmproj가 약 4.2GB라 VS Code 등 다른 무거운 프로그램은 끄고 실행
- Windows에서 SSH로 접속해 한글을 입력하려면 먼저 `chcp 65001`

## 한계

- COCO 클래스만 탐지 (지갑·열쇠·이어폰 없음)
- 같은 종류 물건은 1개만 추적 (컵 2개 = 1개)
- 가까이서 비스듬히 보면 오탐이 있음 (예: 휴대폰 뒷면을 노트북으로)
