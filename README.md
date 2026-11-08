# where-did-i-put-it

Jetson Orin 온디바이스 "어디 뒀지?" 음성 물건 찾기 도우미.
몸에 단(1인칭) 카메라로 물건을 마지막으로 본 장면을 기억하고, 물어보면 Whisper → Gemma 4 멀티모달 → Piper로 음성 답변한다. 인터넷 없이 완전 로컬에서 동작.

가천대 「Physical AI의 Vision-LLM 융합 시청각 멀티모달 시스템」 최종 프로젝트 (Project03).

## 파이프라인

```
[vision_tracker.py]  카메라 → YOLO11n INT8 → 물건별 '봄/사라짐' + 같이 보인 물건·가구 → output/item_memory.json
                                                                                 + output/last_frame_<id>.jpg
[voice_assistant.py] Enter → STT(Whisper) → Gemma(찾는 물건 고르기) → Gemma 멀티모달(기억 + 마지막 장면 사진) → TTS(Piper)
```

- 찾을 물건은 휴대폰, 마우스 2개. 보이면 `봄`, 3초 이상 안 보이면 `사라짐` (잠깐 가려지거나 깜빡여도 유지)
- 확신도 0.3 미만 탐지는 무시 (1인칭 영상은 흐리고 각도가 심해서 너무 높이면 휴대폰을 놓침)
- 컵·키보드·노트북·책상 등은 찾는 대상이 아니라 "어디 근처였는지" 설명용
- 1인칭 카메라는 고개만 돌려도 화면 좌표가 바뀌므로, 위치는 좌표 대신 같이 보인 물건·가구와 마지막 프레임으로 기억
- 기억에 없는 물건은 Gemma를 거치지 않고 "본 적이 없어요"로 답해서 위치를 지어내지 않음

## 파일

| 파일 | 역할 |
|---|---|
| `vision_tracker.py` | 카메라 → YOLO → 물건 상태·이벤트 판단 → 기억 JSON, 마지막 장면 사진 저장 |
| `voice_assistant.py` | 질문(음성 또는 글자) → 기억 JSON + 사진 → Gemma → 음성 답변 |

두 프로그램은 `item_memory.json` 파일로만 연결된다 (트래커는 계속 기록, 도우미는 질문할 때만 읽음).

## 실행 방법

모든 경로는 이 폴더 기준이라, 아래 파일만 이 폴더 안에 있으면 어디서든 실행된다. 용량이 커서 git에는 없음(`.gitignore`).

```
where-did-i-put-it/
├── vision_tracker.py
├── voice_assistant.py
├── .venv/                       # ultralytics, llama-cpp-python (수업 환경의 .venv)
├── .piper_venv/                 # piper-tts
├── src/models/
│   ├── YOLO/yolo11n_int8.engine
│   ├── Gemma4/google_gemma-4-E2B-it-Q4_K_M.gguf
│   ├── Gemma4/mmproj-google_gemma-4-E2B-it-f16.gguf
│   └── Piper/ko_KR-kss-medium.onnx (+ .onnx.json)
├── whisper.cpp/build-cpu/bin/   # whisper-cli + libwhisper·libggml .so
├── whisper.cpp/models/ggml-base.bin
└── output/                      # 실행하면 생김 (기억 JSON, 사진, 음성)
```

수업 환경(`~/vision-llm`)에서 그대로 가져오기:

```bash
S=~/vision-llm
mkdir -p src/models whisper.cpp/build-cpu whisper.cpp/models
cp -a $S/src/models/YOLO $S/src/models/Gemma4 $S/src/models/Piper src/models/
cp -a $S/whisper.cpp/build-cpu/bin whisper.cpp/build-cpu/
cp -a $S/whisper.cpp/models/ggml-base.bin whisper.cpp/models/
cp -a $S/.venv $S/.piper_venv .
```

실행 (이 폴더에서):

```bash
# 터미널 1: 카메라 기록 (q 종료)
.venv/bin/python vision_tracker.py

# 터미널 2: 질문 (Enter 후 5초 동안 말하기, 글자 입력도 가능, q 종료)
.venv/bin/python voice_assistant.py
```

참고
- 마이크는 `voice_assistant.py`의 `MIC_DEVICE`에 이름으로 지정 (`arecord -l`로 확인)
- Gemma + mmproj가 약 4.2GB라 VS Code 등 다른 무거운 프로그램은 끄고 실행
## 한계

- COCO 클래스만 탐지 (지갑·열쇠·이어폰 없음)
- 같은 종류 물건은 1개만 추적 (컵 2개 = 1개)
- 가까이서 비스듬히 보면 오탐이 있음 (예: 휴대폰 뒷면을 노트북으로)
