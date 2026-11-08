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

## 주요 함수

### vision_tracker.py

| 함수 | 입력 → 출력 | 하는 일 |
|---|---|---|
| `detect(yolo, frame)` | 프레임 → 찾을 물건 `{class_id: 박스}`, 주변 물건 `[(이름, 박스)]` | YOLO로 휴대폰·마우스와 주변 물건 10종만 탐지 (확신도 0.3 이상). 찾을 물건은 클래스당 가장 확실한 박스 1개만. 좌표는 0~1 정규화 |
| `nearby_names(box, others)` | 물건 박스, 같은 화면의 다른 물건 → `["키보드", ...]` | 박스 중심끼리 거리로 가까운 순 최대 3개, 같은 이름은 하나로. 1인칭이라 좌표 대신 "무엇 옆"으로 위치를 기억 |
| `update_items(items, detections, landmarks, now)` | 이번 프레임 탐지 결과 → 새 이벤트 `[(class_id, "봄" 또는 "사라짐")]` | 보이면 `last_seen`·`box`·`near`를 최신으로 덮어쓰고, `lost`였으면 `seen`으로 바꾸며 '봄'. `seen`인데 3초 넘게 안 보이면 `lost`로 바꾸며 '사라짐'. 계속 덮어쓰므로 사라질 때 마지막 장면이 남음 |
| `save_frame(frame, class_id, box)` | → `output/last_frame_<id>.jpg` | 물건에 초록 박스를 그려 저장. Gemma가 사진 속 어떤 물건인지 알 수 있게 표시 |
| `save_json(path, data)` | → `output/item_memory.json` | `.tmp`에 쓰고 `os.replace`로 한 번에 교체. 도우미가 반쯤 쓰인 파일을 읽지 않음 |
| `main()` | | 카메라 프레임마다 `detect` → 물건별 마지막 프레임 보관 → `update_items` → 이벤트가 있으면 사진·JSON 저장. 카메라가 끊기면 다시 연결, `q` 종료 |

### voice_assistant.py

| 함수 | 입력 → 출력 | 하는 일 |
|---|---|---|
| `speech_to_text()` | → `"마우스 어디 뒀지?"` | `arecord`로 5초 녹음(`pasuspender`로 PulseAudio 잠시 정지) → `whisper-cli -l ko` 받아쓰기 |
| `text_to_speech(text)` | 답변 → 스피커 | Piper로 wav 생성 → `aplay` 재생 |
| `load_items()` | → `{"마우스": {...}}` | 기억 JSON을 물건 이름으로 찾기 쉽게 변환 (파일 없으면 빈 dict) |
| `item_to_text(item)` | 기억 → `"마우스: 5분 전 키보드 근처에서 마지막으로 봄."` | 마지막으로 본 시각을 "N분 전"으로 바꿔 Gemma에게 줄 한 줄 문장 생성 |
| `image_to_data_url(path)` | jpg → base64 data URL | Gemma 멀티모달 입력 형식으로 변환 |
| `ask_gemma(llm, system_prompt, user_content, max_tokens)` | 프롬프트 → 답변 문자열 | Gemma 호출 공통 함수 (`temperature=0.0`) |
| `answer_question(llm, question)` | 질문 → 답변 | ① Gemma(텍스트)가 기억 목록 중 찾는 물건을 한 단어로 고름 ("핸드폰"도 "휴대폰"으로) ② 기억에 없으면 Gemma 없이 "본 적이 없어요" ③ 있으면 기억 문장 + 마지막 사진을 Gemma(멀티모달)에 넣어 위치 설명 |
| `main()` | | Gemma + mmproj 로드 → 워밍업 1회(첫 질문 32초 → 4초) → Enter면 음성, 글자 입력이면 그대로 질문 → 괄호 토큰(`[끝]` 등)만 있으면 다시 묻기 → 답변 출력·재생, `q` 종료 |

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

실행 (이 폴더에서, 터미널 2개):

```bash
# 터미널 1: 카메라 기록 (q 종료)
source .venv/bin/activate
python vision_tracker.py

# 터미널 2: 질문 ('준비 완료' 후 Enter → 5초 동안 말하기, 글자 입력도 가능, q 종료)
source .venv/bin/activate
python voice_assistant.py
```

venv를 켜지 않고 `.venv/bin/python vision_tracker.py`처럼 바로 실행해도 된다.
`.piper_venv`는 코드가 경로로 직접 실행하므로 따로 켤 필요 없음.

시연 순서
1. 터미널 1에서 휴대폰·마우스를 보면 `[봄]`, 고개를 돌려 3초 지나면 `[사라짐]`
2. 터미널 2에서 Enter → "마우스 어디 뒀지?" → 위치를 음성으로 답변
3. "지갑 어디 있어?" → "그 물건은 아직 본 적이 없어요" (기억에 없는 물건은 지어내지 않음)

참고
- 마이크는 `voice_assistant.py`의 `MIC_DEVICE`에 이름으로 지정 (`arecord -l`로 확인)
- Gemma + mmproj가 약 4.2GB라 VS Code 등 다른 무거운 프로그램은 끄고 실행
## 한계

- COCO 클래스만 탐지 (지갑·열쇠·이어폰 없음)
- 같은 종류 물건은 1개만 추적 (컵 2개 = 1개)
- 가까이서 비스듬히 보면 오탐이 있음 (예: 휴대폰 뒷면을 노트북으로)
