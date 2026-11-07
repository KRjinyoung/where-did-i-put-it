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

## 실행 방법

(구현 후 작성: 의존성, 모델 경로, `arecord -l`로 `MIC_DEVICE` 확인, 두 터미널 실행 순서)
