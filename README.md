# where-did-i-put-it

Jetson Orin 온디바이스 "어디 뒀지?" 음성 물건 찾기 도우미.
YOLO11 INT8 + MediaPipe로 물건 위치·이벤트를 기록하고, Whisper → Gemma 4 멀티모달 → Piper로 음성 질의응답한다. 인터넷 없이 완전 로컬에서 동작.

가천대 「Physical AI의 Vision-LLM 융합 시청각 멀티모달 시스템」 최종 프로젝트 (Project03). 상세 설계는 [PLAN.md](PLAN.md).

## 파이프라인

```
[vision_tracker.py]  카메라 → YOLO(물건) + MediaPipe(손) → 물건별 마지막 위치 + 이벤트(놓임/이동/집어감/가려짐) → src/json/item_memory.json
[voice_assistant.py] Enter → STT(Whisper) → Gemma(찾는 물건 판별) → Gemma 멀티모달(마지막 목격 프레임) → TTS(Piper)
```

## 진행 체크리스트

- [ ] 1. 기본: tracker(위치·놓임/이동/사라짐) + assistant(STT → Gemma 텍스트 → TTS) end-to-end
- [ ] 2. 개선 2: 질문 시 Gemma 멀티모달 주변 묘사
- [ ] 3. 개선 1: MediaPipe 손 감지로 "집어감 / 가려짐" 구분
- [ ] 4. README 실행 방법 + 발표 PDF, 제출 repo `Vision-LLM-202434616/Project03/`로 복사

마감: 2026-11-08 (일) 시연·발표 당일.

## 파일

| 파일 | 역할 |
|---|---|
| `item_memory.py` | 물건 상태 갱신·이벤트 감지·JSON 저장·자연어 변환 (카메라/모델 없는 순수 로직) |
| `vision_tracker.py` | 카메라 → YOLO → `item_memory.update_items` → `src/json/item_memory.json`, `src/output/last_frame_<class>.jpg` |
| `voice_assistant.py` | Enter → STT → 기억 JSON → Gemma → TTS |
| `test_item_memory.py` | 이벤트 로직 확인 (`python test_item_memory.py` → `OK`) |

## 실행 방법

수업 환경(`~/vision-llm`)에 이미 있는 것: `.venv`(ultralytics, llama-cpp-python), `src/models/YOLO/yolo11n_int8.engine`, `src/models/Gemma4/*.gguf`, `whisper.cpp` 빌드, `.piper_venv` + `src/models/Piper/ko_KR-kss-medium.onnx`.

```bash
cd ~/vision-llm
git clone https://github.com/KRjinyoung/where-did-i-put-it.git
source .venv/bin/activate

# 마이크 번호 확인 후 voice_assistant.py의 MIC_DEVICE 수정
arecord -l

# 터미널 1
python where-did-i-put-it/vision_tracker.py

# 터미널 2
python where-did-i-put-it/voice_assistant.py
```

카메라와 YOLO 모델은 자동 선택: Linux(Jetson)는 USB 카메라 `v4l2src` 파이프라인 + `yolo11n_int8.engine`, 그 외(노트북)는 0번 카메라 + `yolo11n.pt`.

### 노트북에서 비전 부분만 실험 (Windows)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install opencv-python ultralytics
.\.venv\Scripts\python.exe vision_tracker.py
```
