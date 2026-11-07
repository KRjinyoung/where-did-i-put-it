# 최종 프로젝트 (Project03): "어디 뒀지?" 음성 물건 찾기 도우미

## Context

강의 최종 프로젝트 정보 (README.md, `[ANSWER]_06-Vision_LLM_Multimodal_Systems.ipynb` "Final Project" 셀 기준):

- **과제 문구**: "Gemma와 YOLO, 그리고 모든 기법들을 통합하여, 객체를 탐지하고 **특정 이벤트를 감지**하여 **유저 음성 입력**을 받아 **음성으로 출력**하는 파이프라인을 설계"
- **일정**: 11/7(토) 설계·개발 시작, **11/8(일) 개발 + 시연 및 발표** (당일 제출)
- **제출**: 본인 GitHub repo에 `Project03/` 폴더 = 독립 실행 가능한 파일 + `README.md`(실행 방법, 의존성, 참고사항) + **발표자료 `.pdf`**. 제출 시각 = commit 시각. 모호한 commit 메시지("update", "fix", "test")는 감점.
- **배점**: 기술적 완성도·실행 안정성 30% / 코드 구조·가독성 20% / 제출 형식 10% / **창의적 설계 30%** / 시연·발표 10%. 합계에 **이해도 가중치 w (0~1) 곱함** → 발표 Q&A에서 코드 전부 설명 가능해야 함. AI 과의존 감점.
- **실행 환경**: Jetson Orin, 인터넷 없이 로컬. Jetson에 브라우저 설치 금지.
- 현재 작업 폴더는 강사 repo clone (`origin = kyu-rae-kim/...`) → 제출은 본인 repo로.

사용자 아이디어: 물건 잊어버렸을 때 음성으로 알려주기 → 선택: **B안 "어디 뒀지 찾기"**, 대상은 **COCO 클래스만**.

## 사용할 수업 자산 (재사용, 새로 만들 것 최소)

| 기능 | 출처 | 재사용 대상 |
|---|---|---|
| YOLO INT8 TensorRT 엔진 | 04 노트북 G.2~4 | `src/models/YOLO/yolo11n_int8.engine` |
| 카메라 입력 | 06 A.6 (CSI `nvarguscamerasrc` GStreamer pipeline) / B.4 (`cv2.VideoCapture(0)` USB) | 그대로 복사 |
| YOLO 결과 → dict → JSON, `tmp` 저장 후 `os.replace` 원자적 교체 | 06 A.7 첫 번째 .py | 그대로 패턴 사용 |
| JSON mtime 감시 + `detections_to_text()` + Gemma 호출 | 06 A.7 두 번째 .py | 그대로 패턴 사용 |
| Gemma 로드 (`google_gemma-4-E2B-it-Q4_K_M.gguf`, `n_gpu_layers=-1, n_ctx=2048, n_batch=32`) | 05 F / 06 | 그대로 |
| 프롬프트 형식 Instruction / Constraint / Output Format | 06 전반 | 그대로 |
| `speech_to_text()` (pasuspender + arecord 5초 16kHz mono + whisper-cli `-l ko`) | 06 C.3 | 함수 복사 |
| `text_to_speech()` (`.piper_venv/bin/python -m piper`, `ko_KR-kss-medium.onnx` + aplay) | 06 C.3 | 함수 복사 |

COCO 대상 클래스 (id): backpack 24, umbrella 25, handbag 26, bottle 39, cup 41, laptop 63, mouse 64, remote 65, keyboard 66, cell phone 67, book 73, scissors 76 + 위치 기준용 person 0, chair 56, couch 57, bed 59, dining table 60, tv 62. (지갑·열쇠·이어폰은 COCO 없음 → 발표 때 한계로 언급)

## 설계: 프로세스 2개 + JSON 파일 1개 (수업 A.7 분리 구조 그대로)

```
[vision_tracker.py]  카메라 → YOLO → 물건별 마지막 위치 기억 + 이벤트 감지 → src/json/item_memory.json
[voice_assistant.py] Enter → STT(Whisper) → item_memory.json 자연어 변환 → Gemma → TTS(Piper)
```

### 1. `Project03/vision_tracker.py`
- 매 프레임 YOLO(`classes=` 위 목록, conf 0.25). 
- 물건 클래스별 상태 dict: `last_seen` 시각, `bbox`, `position`(화면 좌/중/우 × 위/아래 — bbox 중심 3등분), `near`(가장 많이 겹치는 가구/큰 물체, 예 "dining table"→"책상"), `person_near`(마지막 목격 때 person bbox와 겹쳤는지), `status`.
- **이벤트 감지** (과제 요구 "특정 이벤트"):
  - `놓임`: 새로 등장 후 N초(예 2s) 연속 탐지 → 위치 확정
  - `이동`: 확정 위치 중심이 임계값 이상 이동
  - `사라짐`: M초(예 3s) 미탐지 → `person_near`면 "사람이 가져간 듯" 기록
  - YOLO 깜빡임 대비 시간 기반 debounce (N, M은 상수로 두고 현장 튜닝)
- 이벤트 발생 시만 JSON 저장 (`tmp` → `os.replace`). 이벤트 로그 최근 20개 유지 (`deque(maxlen=20)`).
- 위치 확정/이동 이벤트 때 전체 프레임에 해당 물건 bbox만 그려서 `src/output/last_frame_<class>.jpg` 저장 (crop은 주변이 안 보여서 개선 2에 못 씀). 저장도 tmp → `os.replace`.
- 화면: `result.plot()` + `q` 종료.

### 2. `Project03/voice_assistant.py`
- Gemma 1회 로드. 루프: `input("Enter 누르고 질문")` → `speech_to_text()` → `memory_to_text(JSON_PATH)` (한국어 이름 매핑 dict, 예 "cell phone"→"휴대폰", "N분 전 화면 왼쪽 아래 책상 근처에서 마지막으로 보임", 사라짐 이벤트 포함) → Gemma.
- 시스템 프롬프트: Instruction "사용자가 찾는 물건의 마지막 위치를 알려주시오" / Constraint "기억 목록에 없는 물건은 '본 적 없다'고 답하고 추측하지 마시오" / Output Format "한국어 두 문장 이내, 음성으로 읽기 쉽게".
- 질문 속 물건 이름 매칭은 Gemma에게 맡김 (키워드 파서 안 만듦; "핸드폰/폰/휴대폰" 동의어 처리 자연스럽게 됨).
- `text_to_speech(answer)`.

### 3. `Project03/README.md`, 발표자료 `.pdf`
- README: 의존성(ultralytics, llama-cpp-python, whisper.cpp 빌드, piper venv), 모델 경로, `MIC_DEVICE` 확인법(`arecord -l`), 두 터미널 실행 순서.
- 발표: 문제 정의 → 파이프라인 그림 → 이벤트 로직 → 시연 → 한계(COCO 클래스, 가려짐, 카메라 시야 밖) → 개선안.

### 개선 1: MediaPipe 손 감지 → "집어감" 판별 (`vision_tracker.py`에 추가)
- 02 노트북 M절 hand landmarker 코드 재사용 (`src/models/MediaPipe/hand_landmarker.task`).
- 매 프레임 손 landmark(정규화 좌표 × 프레임 크기 = 픽셀) 중 하나라도 물건 bbox 안 → 그 물건 `hand_touch_time = now`.
- `사라짐` 판정 시: `now - hand_touch_time < 2s` → `"손으로 집어감"`, 아니면 `"가려짐/시야 밖"`. 기존 `person_near`는 이걸로 대체.
- 새 로직은 bbox 포함 판정 몇 줄뿐. FPS 떨어지면 손 감지를 2프레임마다 1회로.

### 개선 2: 질문 시 Gemma 멀티모달 주변 묘사 (`voice_assistant.py`)
- Gemma 로드를 06 B절처럼 `Gemma4ChatHandler(clip_model_path=MMPROJ_PATH)`로 변경.
- 1차 호출 (텍스트만): 질문 + 기억 목록 → 찾는 물건 클래스명 한 단어 (06 신호등 "한 단어" 프롬프트 패턴).
- 2차 호출 (이미지 포함): 질문 + 그 물건 기억 텍스트 + `last_frame_<class>.jpg` → "빨간 책 옆 책상 위, 2분 전" 같은 답변.
- 기억 목록에 없거나 사라진 물건이면 2차 호출 생략하고 텍스트 답변만.
- 지연: STT 수 초 + Gemma 2회 + TTS ≈ 10초 내외. 시연 때 "찾는 중이에요" TTS 먼저 재생해서 체감 지연 줄임.

### 구현 순서 (단계마다 commit, 깨지면 이전 commit으로 시연)
1. 기본 B안: tracker(위치·놓임/이동/사라짐) + assistant(STT→Gemma 텍스트→TTS) end-to-end 동작 → commit
2. 개선 2 (수업 코드 거의 그대로라 리스크 작음) → commit
3. 개선 1 → commit
4. README + 발표 PDF → commit

skipped: 위치 히스토리/후속 질문, 떠날 때 알림, 사람 재식별, wake-word — 하루 일정에 과함. 시간 남으면 히스토리부터.

## 프로젝트 레포 (기록용, 새로 생성)

- **이름**: `where-did-i-put-it`
- **Description (GitHub, 350자 이내)**:
  `Jetson Orin 온디바이스 "어디 뒀지?" 음성 물건 찾기 도우미 — YOLO11 INT8 + MediaPipe로 물건 위치·이벤트 기록, Whisper → Gemma 4 멀티모달 → Piper로 음성 질의응답 (완전 오프라인)`
- **공개 범위**: public (과제 제출이 GitHub repo 기준이라 강사가 볼 수 있어야 함)
- 생성: `gh repo create KRJinyoung/where-did-i-put-it --public --description "<위 문구>" --clone` → 위치 `C:\Users\jinyoung\OneDrive\Documents\gachon\3-2\ondeviceAI\where-did-i-put-it`
- 초기 커밋 내용:
  - `PLAN.md` = 이 계획 파일 복사본 (다음 세션에서 이어하기용)
  - `README.md` = 프로젝트 소개(문제·파이프라인·실행 방법 자리), 진행 체크리스트
  - `.gitignore` = `*.gguf *.engine *.onnx *.pt *.task *.wav src/output/ src/json/ .venv/ .piper_venv/ __pycache__/`
- 코드는 레포 루트에 `vision_tracker.py`, `voice_assistant.py`. 마감 전에 과제 제출 repo `KRjinyoung/Vision-LLM-202434616`의 `Project03/`로 복사 (+ README, 발표 PDF). 이 레포는 개발 기록용.
- Jetson에선 `~/vision-llm/` 아래 clone, 실행은 `~/vision-llm`을 cwd로 (`src/models/...` 상대경로 그대로 쓰려고).

## 이어하기 장치
- 메모리에 project 메모 저장: 컨셉(B안 + 개선 1·2), 레포 이름·경로, 진행 단계, 마감 11/8.
- 레포 `README.md` 체크리스트(구현 순서 1~4)를 단계마다 갱신해서 commit → 다음 세션은 `PLAN.md` + 체크리스트만 보면 이어짐.

## 작업 위치
- 코드는 Jetson `~/vision-llm/`에서 실행 (모델·whisper.cpp·.piper_venv가 거기 있음). 이 Windows 폴더에서 작성한다면 본인 제출 repo로 옮겨서 Jetson에서 pull.
- 구현은 사용자가 직접 작성/이해하는 것이 점수(w)에 직결 → 코드 작성 시 함수마다 이유 설명 가능하게 짧게 유지.

## 검증 (Jetson에서)
1. `vision_tracker.py`만 실행 → 책상에 휴대폰·컵 놓기 → 2초 후 JSON에 `놓임` 이벤트, 위치 문자열 확인 → 휴대폰 치우기 → 3초 후 `사라짐` 확인.
2. `voice_assistant.py` 별도 터미널 → "내 핸드폰 어디 있어?" 말하기 → STT 텍스트 출력 확인 → 음성 응답 "약 1분 전 화면 왼쪽 책상 위에서 마지막으로 봤어요" 류 확인.
3. 없는 물건 질문("지갑 어디 있어?") → "본 적 없다" 응답 확인 (hallucination 억제).
4. 개선 1: 휴대폰을 손으로 집어 치움 → `손으로 집어감`; 책으로 덮음 → `가려짐/시야 밖` 구분되는지.
5. 개선 2: 답변에 이미지 기반 주변 묘사(주변 물체·색) 포함되는지, 전체 응답 시간 측정.
6. `tegrastats`로 동시 실행 메모리 확인 (YOLO TRT + MediaPipe + Gemma+mmproj). 부족하면 `n_ctx` 축소 또는 손 감지 프레임 간격 늘리기.
