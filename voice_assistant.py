"""음성으로 물건 위치를 묻고 음성으로 답하는 도우미.

STT(Whisper) → Gemma(찾는 물건 고르기) → Gemma 멀티모달(기억 + 마지막 장면 사진) → TTS(Piper)
실행 (~/vision-llm 에서, vision_tracker.py와 다른 터미널): python where-did-i-put-it/voice_assistant.py
"""

import base64
import json
import os
import subprocess
import sys
import time

from llama_cpp import Llama
from llama_cpp.llama_chat_format import Gemma4ChatHandler

from item_memory import ITEM_NAMES, item_to_text


GEMMA_MODEL_PATH = "src/models/Gemma4/google_gemma-4-E2B-it-Q4_K_M.gguf"
MMPROJ_PATH = "src/models/Gemma4/mmproj-google_gemma-4-E2B-it-f16.gguf"
MEMORY_PATH = "src/json/item_memory.json"
FRAME_DIR = "src/output"

MIC_DEVICE = "plughw:CARD=Microphone,DEV=0"  # card 번호는 부팅마다 바뀌어서 이름으로 지정 (arecord -l 의 [ ] 앞 이름)
AUDIO_FILE = "src/audio/input.wav"
WHISPER_PATH = "whisper.cpp/build-cpu/bin/whisper-cli"
WHISPER_MODEL = "whisper.cpp/models/ggml-base.bin"
RECORD_SECONDS = 5

PIPER_PYTHON = ".piper_venv/bin/python"
PIPER_MODEL = "src/models/Piper/ko_KR-kss-medium.onnx"
OUTPUT_FILE = "src/audio/response.wav"

CONTEXT_WINDOW = 2048
MAX_TOKENS = 100

# 1단계: 질문에서 찾는 물건 고르기 (수업의 신호등 '한 단어' 프롬프트 방식)
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

# 2단계: 기억 + 마지막 장면 사진으로 위치 설명
ANSWER_PROMPT = """
Instruction:
기억 정보와 마지막으로 본 사진을 바탕으로 사용자가 찾는 물건이 어디 있는지 알려주시오.
사진의 초록 박스가 찾는 물건이다.

Constraint:
기억 정보의 시간을 그대로 말하시오.
사진에서 박스 주변에 확실히 보이는 물건이나 장소로 위치를 설명하시오.
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
            WHISPER_PATH,
            "-m", WHISPER_MODEL,
            "-f", AUDIO_FILE,
            "-l", "ko",
            "--no-timestamps",
        ],
        text=True,
        capture_output=True,
        check=True,
    )

    return result.stdout.strip()


def text_to_speech(text):
    subprocess.run([PIPER_PYTHON, "-m", "piper", "-m", PIPER_MODEL, "-f", OUTPUT_FILE, "--", text], check=True)
    subprocess.run(["aplay", OUTPUT_FILE], check=True)


def load_items():
    """기억 JSON → {물건 이름: 상태}."""
    if not os.path.exists(MEMORY_PATH):
        return {}

    with open(MEMORY_PATH, "r", encoding="utf-8") as file:
        memory = json.load(file)

    return {item["name"]: item for item in memory["items"]}


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
    # 1. 찾는 물건 고르기 (핸드폰/폰/휴대폰 같은 다른 표현도 Gemma가 맞춰줌)
    names = ", ".join(ITEM_NAMES.values())
    target = ask_gemma(llm, FIND_PROMPT, f"목록: {names}\n질문: {question}", 10).strip(" .")
    print(f"[찾는 물건] {target}")

    # 기억에 없으면 Gemma에 묻지 않고 고정 답변 → 지어낸 위치를 말할 수 없음
    item = load_items().get(target)

    if item is None:
        return "그 물건은 아직 본 적이 없어요."

    memory_text = item_to_text(item, time.time())
    print(f"[기억] {memory_text}")

    # 2. 기억 문장 + 마지막 장면 사진을 같이 넣어 위치 설명
    content = [{"type": "text", "text": f"Context:\n{memory_text}\n\nQuestion:\n{question}"}]
    frame_path = os.path.join(FRAME_DIR, f"last_frame_{item['class_id']}.jpg")

    if os.path.exists(frame_path):
        content.append({"type": "image_url", "image_url": {"url": image_to_data_url(frame_path)}})

    return ask_gemma(llm, ANSWER_PROMPT, content, MAX_TOKENS)


def main():
    os.makedirs(os.path.dirname(AUDIO_FILE), exist_ok=True)

    llm = Llama(
        model_path=GEMMA_MODEL_PATH,
        chat_handler=Gemma4ChatHandler(clip_model_path=MMPROJ_PATH),
        n_gpu_layers=-1,
        n_ctx=CONTEXT_WINDOW,
        n_batch=32,
        n_ubatch=32,
        verbose=False,
    )

    # Windows SSH 터미널은 한글을 UTF-8이 아닌 인코딩으로 보낼 수 있어서, 깨진 글자 때문에 죽지 않게
    sys.stdin.reconfigure(errors="replace")

    while True:
        command = input("\nEnter: 말로 질문 / 글자 입력: 그대로 질문 / q: 종료 > ").strip()

        if command == "q":
            break

        if command:  # 마이크가 안 될 때 대비: 입력한 글자를 그대로 질문으로
            question = command
        else:
            print(f"{RECORD_SECONDS}초 동안 말씀해 주세요.")
            question = speech_to_text()

        print(f"[질문] {question}")

        # 무음이면 Whisper가 [BLANK_AUDIO] 같은 토큰만 돌려줌
        if not question or "BLANK_AUDIO" in question:
            text_to_speech("잘 못 들었어요. 다시 말씀해 주세요.")
            continue

        start = time.time()
        answer = answer_question(llm, question)
        print(f"[답변] {answer}  ({time.time() - start:.1f}초)")

        text_to_speech(answer)


if __name__ == "__main__":
    main()
