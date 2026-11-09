# where-did-i-put-it

An on-device "Where did I put it?" voice assistant for finding items, running on Jetson Orin.
A body-worn (first-person) camera remembers the last scene in which each item was seen, and when asked, it answers by voice through Whisper → Gemma 4 multimodal → Piper. Runs fully locally with no internet.

The assistant listens and speaks in Korean, so item names, Gemma prompts and spoken answers in the code stay in Korean.

Final project (Project03) for the Gachon University course "Vision-LLM Fusion Audio-Visual Multimodal Systems for Physical AI".

## Pipeline

```
[vision_tracker.py]  camera → YOLO11n INT8 → per-item 'seen/lost' + nearby items and furniture → output/item_memory.json
                                                                                                + output/last_frame_<id>.jpg
[voice_assistant.py] Enter → STT(Whisper) → Gemma(pick the target item) → Gemma multimodal(memory + last-scene photo) → TTS(Piper)
```

- Two target items: cell phone and mouse. Visible means `seen`; not visible for 3+ seconds means `lost` (brief occlusion or flicker keeps the state)
- Detections below 0.3 confidence are ignored (first-person video is blurry with steep angles, so a higher threshold misses the phone)
- Cups, keyboards, laptops, desks, etc. are not search targets; they only describe "what it was near"
- With a first-person camera, screen coordinates change whenever the head turns, so location is remembered by nearby items/furniture and the last frame instead of coordinates
- Items not in memory are answered with "I haven't seen it" without calling Gemma, so it never makes up a location

## Files

| File | Role |
|---|---|
| `vision_tracker.py` | camera → YOLO → item state and event detection → saves memory JSON and last-scene photo |
| `voice_assistant.py` | question (voice or typed) → memory JSON + photo → Gemma → spoken answer |

The two programs are connected only through the `item_memory.json` file (the tracker keeps writing, the assistant reads only when asked).

## Main functions

### vision_tracker.py

| Function | Input → Output | What it does |
|---|---|---|
| `detect(yolo, frame)` | frame → target items `{class_id: box}`, surrounding objects `[(name, box)]` | Runs YOLO on only the phone, mouse and 10 surrounding classes (confidence ≥ 0.3). Keeps only the most confident box per target class. Coordinates normalized to 0–1 |
| `nearby_names(box, others)` | item box, other objects in the same frame → `["키보드", ...]` | Up to 3 closest by box-center distance, duplicate names merged. First-person view, so location is remembered as "next to what" instead of coordinates |
| `update_items(items, detections, landmarks, now)` | this frame's detections → new events `[(class_id, "seen" or "lost")]` | When visible, overwrites `last_seen`, `box`, `near` with the latest, and if it was `lost`, switches to `seen` and emits 'seen'. If `seen` but not visible for over 3 seconds, switches to `lost` and emits 'lost'. Because it keeps overwriting, the last scene remains when the item disappears |
| `save_frame(frame, class_id, box)` | → `output/last_frame_<id>.jpg` | Draws a green box around the item and saves it, so Gemma knows which object in the photo is the target |
| `save_json(path, data)` | → `output/item_memory.json` | Writes to `.tmp` and swaps it in with `os.replace`, so the assistant never reads a half-written file |
| `main()` | | For each camera frame: `detect` → keep the last frame per item → `update_items` → save photo and JSON on events. Reconnects if the camera drops, `q` to quit |

### voice_assistant.py

| Function | Input → Output | What it does |
|---|---|---|
| `speech_to_text()` | → `"마우스 어디 뒀지?"` ("Where did I put the mouse?") | Records 5 s with `arecord` (PulseAudio paused via `pasuspender`) → transcribes with `whisper-cli -l ko` |
| `text_to_speech(text)` | answer → speaker | Generates a wav with Piper → plays it with `aplay` |
| `load_items()` | → `{"마우스": {...}}` | Converts the memory JSON into a dict keyed by item name (empty dict if the file is missing) |
| `item_to_text(item)` | memory → `"마우스: 5분 전 키보드 근처에서 마지막으로 봄."` ("Mouse: last seen near the keyboard 5 minutes ago.") | Turns the last-seen time into "N minutes ago" and builds a one-line sentence for Gemma |
| `image_to_data_url(path)` | jpg → base64 data URL | Converts to Gemma's multimodal input format |
| `ask_gemma(llm, system_prompt, user_content, max_tokens)` | prompt → answer string | Shared Gemma call (`temperature=0.0`) |
| `answer_question(llm, question)` | question → answer | ① Gemma (text) picks the target from the memory list as one word (maps synonyms like "핸드폰" to "휴대폰") ② if not in memory, answers "I haven't seen it" without Gemma ③ otherwise feeds the memory sentence + last photo to Gemma (multimodal) to describe the location |
| `main()` | | Loads Gemma + mmproj → one warm-up call (first question 32 s → 4 s) → Enter for voice, typed text is used as the question → re-asks if only bracket tokens (`[BLANK_AUDIO]` etc.) → prints and plays the answer, `q` to quit |

## How to run

All paths are relative to this folder, so it runs from anywhere as long as the files below are inside this folder. They are large and not in git (`.gitignore`).

```
where-did-i-put-it/
├── vision_tracker.py
├── voice_assistant.py
├── .venv/                       # ultralytics, llama-cpp-python (the course .venv)
├── .piper_venv/                 # piper-tts
├── src/models/
│   ├── YOLO/yolo11n_int8.engine
│   ├── Gemma4/google_gemma-4-E2B-it-Q4_K_M.gguf
│   ├── Gemma4/mmproj-google_gemma-4-E2B-it-f16.gguf
│   └── Piper/ko_KR-kss-medium.onnx (+ .onnx.json)
├── whisper.cpp/build-cpu/bin/   # whisper-cli + libwhisper / libggml .so
├── whisper.cpp/models/ggml-base.bin
└── output/                      # created on run (memory JSON, photos, audio)
```

Copy them from the course environment (`~/vision-llm`):

```bash
S=~/vision-llm
mkdir -p src/models whisper.cpp/build-cpu whisper.cpp/models
cp -a $S/src/models/YOLO $S/src/models/Gemma4 $S/src/models/Piper src/models/
cp -a $S/whisper.cpp/build-cpu/bin whisper.cpp/build-cpu/
cp -a $S/whisper.cpp/models/ggml-base.bin whisper.cpp/models/
cp -a $S/.venv $S/.piper_venv .
```

Run (from this folder, two terminals):

```bash
# Terminal 1: camera tracking (q to quit)
source .venv/bin/activate
python vision_tracker.py

# Terminal 2: questions (after 'Ready', press Enter → speak for 5 s, typing also works, q to quit)
source .venv/bin/activate
python voice_assistant.py
```

You can also run without activating the venv, e.g. `.venv/bin/python vision_tracker.py`.
`.piper_venv` is invoked by path from the code, so it does not need to be activated.

Demo order
1. In terminal 1, looking at the phone or mouse prints `[seen]`; turn your head away and after 3 s it prints `[lost]`
2. In terminal 2, press Enter → say "마우스 어디 뒀지?" ("Where did I put the mouse?") → the location is answered by voice
3. "지갑 어디 있어?" ("Where is my wallet?") → "그 물건은 아직 본 적이 없어요" ("I haven't seen that item yet") — items not in memory are never made up

Notes
- Set the microphone by name in `MIC_DEVICE` in `voice_assistant.py` (check with `arecord -l`)
- Gemma + mmproj take about 4.2 GB, so close VS Code and other heavy programs before running

## Limitations

- Only COCO classes are detected (no wallet, keys or earphones)
- Only one item per class is tracked (2 cups = 1 cup)
- False positives at close, oblique angles (e.g. the back of a phone detected as a laptop)
