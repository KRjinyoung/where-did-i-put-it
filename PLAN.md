# Final Project (Project03): "Where did I put it?" voice item-finding assistant

## Context

Course final project info (from README.md and the "Final Project" cell of `[ANSWER]_06-Vision_LLM_Multimodal_Systems.ipynb`):

- **Assignment**: "Integrate Gemma, YOLO and all the techniques to design a pipeline that detects objects, **detects specific events**, takes **user voice input** and **responds with voice**"
- **Schedule**: 11/7 (Sat) design and development start, **11/8 (Sun) development + demo and presentation** (same-day submission)
- **Submission**: a `Project03/` folder in your own GitHub repo = standalone runnable files + `README.md` (how to run, dependencies, notes) + **presentation `.pdf`**. Submission time = commit time. Vague commit messages ("update", "fix", "test") lose points.
- **Grading**: technical completeness and stability 30% / code structure and readability 20% / submission format 10% / **creative design 30%** / demo and presentation 10%. The total is multiplied by an **understanding weight w (0–1)** → must be able to explain all code in the presentation Q&A. Over-reliance on AI loses points.
- **Runtime**: Jetson Orin, local with no internet. No browser may be installed on the Jetson.
- The current working folder is a clone of the instructor repo (`origin = kyu-rae-kim/...`) → submit to your own repo.

User idea: tell me by voice when I lose something → chosen: **Option B "find where I put it"**, targets **COCO classes only**.

## Course assets to reuse (minimize new code)

| Feature | Source | Reused |
|---|---|---|
| YOLO INT8 TensorRT engine | notebook 04 G.2–4 | `src/models/YOLO/yolo11n_int8.engine` |
| Camera input | 06 A.6 (CSI `nvarguscamerasrc` GStreamer pipeline) / B.4 (`cv2.VideoCapture(0)` USB) | copied as is |
| YOLO result → dict → JSON, write to `tmp` then atomic `os.replace` | 06 A.7 first .py | same pattern |
| JSON mtime watch + `detections_to_text()` + Gemma call | 06 A.7 second .py | same pattern |
| Gemma load (`google_gemma-4-E2B-it-Q4_K_M.gguf`, `n_gpu_layers=-1, n_ctx=2048, n_batch=32`) | 05 F / 06 | as is |
| Prompt format Instruction / Constraint / Output Format | 06 throughout | as is |
| `speech_to_text()` (pasuspender + arecord 5 s 16 kHz mono + whisper-cli `-l ko`) | 06 C.3 | function copied |
| `text_to_speech()` (`.piper_venv/bin/python -m piper`, `ko_KR-kss-medium.onnx` + aplay) | 06 C.3 | function copied |

Target COCO classes (id): backpack 24, umbrella 25, handbag 26, bottle 39, cup 41, laptop 63, mouse 64, remote 65, keyboard 66, cell phone 67, book 73, scissors 76 + for location reference person 0, chair 56, couch 57, bed 59, dining table 60, tv 62. (Wallet, keys and earphones are not in COCO → mention as a limitation in the presentation)

## Design: 2 processes + 1 JSON file (same split structure as course A.7)

```
[vision_tracker.py]  camera → YOLO → remember last location per item + detect events → src/json/item_memory.json
[voice_assistant.py] Enter → STT(Whisper) → item_memory.json to natural language → Gemma → TTS(Piper)
```

### 1. `Project03/vision_tracker.py`
- YOLO on every frame (`classes=` list above, conf 0.25).
- Per-class state dict: `last_seen` time, `bbox`, `position` (screen left/center/right × top/bottom — bbox center in thirds), `near` (furniture/large object with the most overlap, e.g. "dining table" → "desk"), `person_near` (whether it overlapped a person bbox when last seen), `status`.
- **Event detection** (the assignment's "specific events"):
  - `placed`: detected continuously for N s (e.g. 2 s) after appearing → location confirmed
  - `moved`: confirmed location center moves beyond a threshold
  - `lost`: not detected for M s (e.g. 3 s) → if `person_near`, record "probably taken by a person"
  - Time-based debounce against YOLO flicker (N and M as constants, tuned on site)
- Save JSON only on events (`tmp` → `os.replace`). Keep the last 20 events (`deque(maxlen=20)`).
- On placed/moved events, save the full frame with only that item's bbox drawn to `src/output/last_frame_<class>.jpg` (a crop hides the surroundings, so it is useless for improvement 2). Also saved via tmp → `os.replace`.
- Display: `result.plot()` + `q` to quit.

### 2. `Project03/voice_assistant.py`
- Load Gemma once. Loop: `input("Press Enter and ask")` → `speech_to_text()` → `memory_to_text(JSON_PATH)` (Korean name mapping dict, e.g. "cell phone" → "휴대폰", "last seen near the desk at the bottom left of the screen N minutes ago", including lost events) → Gemma.
- System prompt: Instruction "Tell the user the last location of the item they are looking for" / Constraint "For items not in the memory list, answer 'never seen it' and do not guess" / Output Format "At most two Korean sentences, easy to read aloud".
- Matching item names in the question is left to Gemma (no keyword parser; synonyms like "핸드폰/폰/휴대폰" are handled naturally).
- `text_to_speech(answer)`.

### 3. `Project03/README.md`, presentation `.pdf`
- README: dependencies (ultralytics, llama-cpp-python, whisper.cpp build, piper venv), model paths, how to find `MIC_DEVICE` (`arecord -l`), order of running the two terminals.
- Presentation: problem definition → pipeline diagram → event logic → demo → limitations (COCO classes, occlusion, out of camera view) → improvements.

### Improvement 1: MediaPipe hand detection → "picked up" detection (added to `vision_tracker.py`)
- Reuse the hand landmarker code from notebook 02 section M (`src/models/MediaPipe/hand_landmarker.task`).
- Each frame, if any hand landmark (normalized coords × frame size = pixels) is inside an item bbox → set that item's `hand_touch_time = now`.
- On `lost`: `now - hand_touch_time < 2s` → `"picked up by hand"`, otherwise `"occluded / out of view"`. This replaces the old `person_near`.
- The new logic is only a few lines of bbox containment. If FPS drops, run hand detection once every 2 frames.

### Improvement 2: Gemma multimodal scene description on question (`voice_assistant.py`)
- Change the Gemma load to `Gemma4ChatHandler(clip_model_path=MMPROJ_PATH)` as in 06 section B.
- 1st call (text only): question + memory list → target class name as one word (06 traffic-light "one word" prompt pattern).
- 2nd call (with image): question + that item's memory text + `last_frame_<class>.jpg` → an answer like "on the desk next to the red book, 2 minutes ago".
- If the item is not in memory or is lost, skip the 2nd call and answer with text only.
- Latency: STT a few s + 2 Gemma calls + TTS ≈ about 10 s. In the demo, play a "searching..." TTS first to reduce perceived latency.

### Implementation order (commit at each step; if something breaks, demo from the previous commit)
1. Basic option B: tracker (location, placed/moved/lost) + assistant (STT → Gemma text → TTS) working end to end → commit
2. Improvement 2 (almost identical to course code, low risk) → commit
3. Improvement 1 → commit
4. README + presentation PDF → commit

skipped: location history / follow-up questions, alerts when leaving, person re-identification, wake word — too much for a one-day schedule. If time remains, history first.

## Project repo (for the record, newly created)

- **Name**: `where-did-i-put-it`
- **Description (GitHub, max 350 chars)**:
  `On-device "Where did I put it?" voice assistant for finding items on Jetson Orin — YOLO11 INT8 + MediaPipe log item locations and events; Whisper → Gemma 4 multimodal → Piper answer spoken questions (fully offline)`
- **Visibility**: public (submission is based on the GitHub repo, so the instructor must be able to see it)
- Create: `gh repo create KRJinyoung/where-did-i-put-it --public --description "<text above>" --clone` → location `C:\Users\jinyoung\OneDrive\Documents\gachon\3-2\ondeviceAI\where-did-i-put-it`
- Initial commit contents:
  - `PLAN.md` = copy of this plan file (to continue in the next session)
  - `README.md` = project intro (placeholders for problem, pipeline, how to run), progress checklist
  - `.gitignore` = `*.gguf *.engine *.onnx *.pt *.task *.wav src/output/ src/json/ .venv/ .piper_venv/ __pycache__/`
- Code at the repo root: `vision_tracker.py`, `voice_assistant.py`. Before the deadline, copy into `Project03/` of the submission repo `KRjinyoung/Vision-LLM-202434616` (+ README, presentation PDF). This repo is for the development record.
- On the Jetson, clone under `~/vision-llm/` and run with `~/vision-llm` as cwd (to keep the `src/models/...` relative paths).

## Continuity
- Save a project memo in memory: concept (option B + improvements 1 and 2), repo name and path, progress, deadline 11/8.
- Update the repo `README.md` checklist (implementation order 1–4) at each step and commit → the next session only needs `PLAN.md` + the checklist to continue.

## Where to work
- Code runs on the Jetson in `~/vision-llm/` (models, whisper.cpp and .piper_venv live there). If written in this Windows folder, move it to your submission repo and pull on the Jetson.
- Writing and understanding the implementation yourself directly affects the score (w) → keep each function short enough to explain why it exists.

## Verification (on the Jetson)
1. Run only `vision_tracker.py` → put a phone and cup on the desk → after 2 s check the `placed` event and location string in the JSON → remove the phone → after 3 s check `lost`.
2. Run `voice_assistant.py` in a separate terminal → say "내 핸드폰 어디 있어?" ("Where is my phone?") → check the STT text is printed → check a spoken answer like "I last saw it on the desk on the left of the screen about 1 minute ago".
3. Ask about a missing item ("지갑 어디 있어?", "Where is my wallet?") → check the "never seen it" answer (hallucination suppression).
4. Improvement 1: pick up the phone by hand → `picked up by hand`; cover it with a book → `occluded / out of view`, check they are distinguished.
5. Improvement 2: check the answer includes an image-based description of the surroundings (nearby objects, colors), measure total response time.
6. Check memory with `tegrastats` while running everything (YOLO TRT + MediaPipe + Gemma + mmproj). If short, reduce `n_ctx` or increase the hand-detection frame interval.
