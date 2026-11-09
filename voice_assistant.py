"""Assistant that takes a spoken question about where an item is and answers by voice.

STT(Whisper) → Gemma(pick the target item) → Gemma multimodal(memory + last-scene photo) → TTS(Piper)
Only reads item_memory.json and last_frame_<class_id>.jpg saved by vision_tracker.py.

Run (from this folder, in a different terminal from vision_tracker.py): .venv/bin/python voice_assistant.py

Prompts and spoken text stay in Korean on purpose: Whisper listens with -l ko and Piper speaks with a Korean voice.
"""

import base64
import json
import os
import re
import subprocess
import time

from llama_cpp import Llama
from llama_cpp.llama_chat_format import Gemma4ChatHandler


# All paths are relative to this file's folder → only this folder's models and outputs are used, wherever it runs from
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

GEMMA_MODEL_PATH = os.path.join(BASE_DIR, "src/models/Gemma4/google_gemma-4-E2B-it-Q4_K_M.gguf")
MMPROJ_PATH = os.path.join(BASE_DIR, "src/models/Gemma4/mmproj-google_gemma-4-E2B-it-f16.gguf")

# Same output/ folder as vision_tracker.py (memory JSON, photos, recorded and response audio)
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MEMORY_PATH = os.path.join(OUTPUT_DIR, "item_memory.json")

MIC_DEVICE = "plughw:CARD=Microphone,DEV=0"  # card number changes every boot, so select by name (the name before [ ] in arecord -l)
AUDIO_FILE = os.path.join(OUTPUT_DIR, "input.wav")
WHISPER_DIR = os.path.join(BASE_DIR, "whisper.cpp/build-cpu/bin")
WHISPER_MODEL = os.path.join(BASE_DIR, "whisper.cpp/models/ggml-base.bin")
RECORD_SECONDS = 5

PIPER_PYTHON = os.path.join(BASE_DIR, ".piper_venv/bin/python")
PIPER_MODEL = os.path.join(BASE_DIR, "src/models/Piper/ko_KR-kss-medium.onnx")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "response.wav")

CONTEXT_WINDOW = 2048
MAX_TOKENS = 100

# Step 1: pick the target item from the question (the course's traffic-light 'one word' prompt pattern)
# Instruction: pick which item in the list the user is looking for.
# Constraint: answer with exactly one name from the list; if not in the list, answer 없음 (none); no extra explanation.
# Output Format: one word.
FIND_PROMPT = """
Instruction:
사용자의 질문에서 찾는 물건이 목록 중 무엇인지 고르시오.

Constraint:
반드시 목록에 있는 이름 하나만 그대로 답하시오.
목록에 없는 물건이면 없음 이라고 답하시오.
다른 설명을 추가하지 마시오.

Output Format:
한 단어.
"""

# Step 2: describe the location from memory + last-scene photo
# Instruction: tell where the item is from the memory and last photo; the green box is the item.
# Constraint: state the memory's time as is; describe the location by things clearly visible around the box
#             (on the desk, next to the box, in hand, etc.); don't mention photo/box/green; don't guess;
#             no special symbols or emoji.
# Output Format: at most two Korean sentences, natural to read aloud.
ANSWER_PROMPT = """
Instruction:
기억 정보와 마지막으로 본 사진을 바탕으로 사용자가 찾는 물건이 어디 있는지 알려주시오.
사진의 초록 박스가 찾는 물건이다.

Constraint:
기억 정보의 시간을 그대로 말하시오.
사진에서 박스 주변에 확실히 보이는 물건이나 장소(책상 위, 상자 옆, 손에 들고 있음 등)로 위치를 설명하시오.
사진, 박스, 초록색이라는 말은 하지 마시오.
확인되지 않는 내용은 추측하지 마시오.
특수기호나 이모지를 쓰지 마시오.

Output Format:
한국어 두 문장 이내. 소리 내어 읽기 자연스러운 말투.
"""


def speech_to_text():
    subprocess.run(
        [
            "pasuspender", "--",
            "arecord",
            "-D", MIC_DEVICE,
            "-f", "S16_LE",
            "-r", "16000",
            "-c", "1",
            "-d", str(RECORD_SECONDS),
            AUDIO_FILE,
        ],
        check=True,
    )

    result = subprocess.run(
        [
            os.path.join(WHISPER_DIR, "whisper-cli"),
            "-m", WHISPER_MODEL,
            "-f", AUDIO_FILE,
            "-l", "ko",
            "--no-timestamps",
        ],
        text=True,
        capture_output=True,
        check=True,
        # whisper-cli must use libwhisper.so etc. from this folder (not from the original build location)
        env=dict(os.environ, LD_LIBRARY_PATH=WHISPER_DIR),
    )

    return result.stdout.strip()


def text_to_speech(text):
    subprocess.run([PIPER_PYTHON, "-m", "piper", "-m", PIPER_MODEL, "-f", OUTPUT_FILE, "--", text], check=True)
    subprocess.run(["aplay", OUTPUT_FILE], check=True)


def load_items():
    """Memory JSON → {item name: state}."""
    if not os.path.exists(MEMORY_PATH):
        return {}

    with open(MEMORY_PATH, "r", encoding="utf-8") as file:
        memory = json.load(file)

    return {item["name"]: item for item in memory["items"]}


def ago_text(seconds):
    """Seconds → Korean 'just now' / 'N minutes ago' / 'N hours ago'."""
    if seconds < 60:
        return "방금 전"
    if seconds < 3600:
        return f"{int(seconds // 60)}분 전"
    return f"{int(seconds // 3600)}시간 전"


def item_to_text(item):
    """One item's memory as a one-line sentence for Gemma. e.g. '휴대폰: 2분 전 키보드 근처에서 마지막으로 봄.' (Phone: last seen near the keyboard 2 minutes ago.)"""
    near = ", ".join(item["near"])

    if item["status"] == "seen":
        return f"{item['name']}: 지금 보고 있음" + (f" ({near} 근처)." if near else ".")

    where = f" {near} 근처에서" if near else ""
    return f"{item['name']}: {ago_text(time.time() - item['last_seen'])}{where} 마지막으로 봄."


def image_to_data_url(path):
    with open(path, "rb") as file:
        return "data:image/jpeg;base64," + base64.b64encode(file.read()).decode("utf-8")


def ask_gemma(llm, system_prompt, user_content, max_tokens):
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        max_tokens=max_tokens,
        temperature=0.0,
    )

    return response["choices"][0]["message"]["content"].strip()


def answer_question(llm, question):
    items = load_items()

    # 1. Pick the target among remembered items (Gemma maps other wordings like 핸드폰/폰 to '휴대폰')
    target = ask_gemma(llm, FIND_PROMPT, f"목록: {', '.join(items)}\n질문: {question}", 10).strip(" .")
    print(f"[target] {target}")

    # Not in memory → fixed answer without asking Gemma, so it cannot make up a location
    if target not in items:
        return "그 물건은 아직 본 적이 없어요."  # "I haven't seen that item yet."

    item = items[target]
    memory_text = item_to_text(item)
    print(f"[memory] {memory_text}")

    # 2. Describe the location from the memory sentence + last-scene photo (the tracker saves the photo before the JSON, so it always exists)
    frame_path = os.path.join(OUTPUT_DIR, f"last_frame_{item['class_id']}.jpg")
    content = [
        {"type": "text", "text": f"Context:\n{memory_text}\n\nQuestion:\n{question}"},
        {"type": "image_url", "image_url": {"url": image_to_data_url(frame_path)}},
    ]

    return ask_gemma(llm, ANSWER_PROMPT, content, MAX_TOKENS)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    llm = Llama(
        model_path=GEMMA_MODEL_PATH,
        chat_handler=Gemma4ChatHandler(clip_model_path=MMPROJ_PATH, verbose=False),
        n_gpu_layers=-1,
        n_ctx=CONTEXT_WINDOW,
        n_batch=32,
        n_ubatch=32,
        verbose=False,
    )

    # The first call takes almost 30 s for GPU warm-up, so run one call before taking questions
    ask_gemma(llm, "한 단어로 답하시오.", "안녕", 1)
    print("Ready")

    while True:
        command = input("\nEnter: ask by voice / type text: ask as typed / q: quit > ").strip()

        if command == "q":
            break

        if command:  # fallback when the mic fails: use the typed text as the question
            question = command
        else:
            print(f"Speak for {RECORD_SECONDS} seconds.")
            question = speech_to_text()

        print(f"[question] {question}")

        # On silence/noise Whisper returns only bracket tokens like [BLANK_AUDIO] → if nothing remains after removing brackets, ask again
        if not re.sub(r"\[.*?\]|\(.*?\)", "", question).strip():
            text_to_speech("잘 못 들었어요. 다시 말씀해 주세요.")  # "I didn't catch that. Please say it again."
            continue

        start = time.time()
        answer = answer_question(llm, question)
        print(f"[answer] {answer}  ({time.time() - start:.1f}s)")

        text_to_speech(answer)


if __name__ == "__main__":
    main()
