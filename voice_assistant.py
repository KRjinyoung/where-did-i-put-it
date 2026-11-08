"""음성으로 물건 위치를 묻고 음성으로 답하는 도우미.

STT(Whisper) → 기억 JSON → Gemma → TTS(Piper)
실행 (~/vision-llm 에서, vision_tracker.py와 다른 터미널): python where-did-i-put-it/voice_assistant.py
"""

import json
import os
import subprocess
import time

from llama_cpp import Llama

from item_memory import memory_to_text


GEMMA_MODEL_PATH = "src/models/Gemma4/google_gemma-4-E2B-it-Q4_K_M.gguf"
MEMORY_PATH = "src/json/item_memory.json"

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

SYSTEM_PROMPT = """
Instruction:
사용자가 찾는 물건이 어디에 있는지 기억 목록을 바탕으로 알려주시오.

Constraint:
기억 목록에 있는 정보만 사용하시오.
목록에 없는 물건은 본 적이 없다고 답하시오.
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


def load_memory_text():
    if not os.path.exists(MEMORY_PATH):
        return "아직 기억한 물건이 없습니다."

    with open(MEMORY_PATH, "r", encoding="utf-8") as file:
        memory = json.load(file)

    return memory_to_text(memory, time.time())


def main():
    os.makedirs(os.path.dirname(AUDIO_FILE), exist_ok=True)

    llm = Llama(
        model_path=GEMMA_MODEL_PATH,
        n_gpu_layers=-1,
        n_ctx=CONTEXT_WINDOW,
        n_batch=32,
        n_ubatch=32,
        verbose=False,
    )

    while True:
        command = input("\nEnter: 질문하기 / q: 종료 > ")

        if command.strip() == "q":
            break

        print(f"{RECORD_SECONDS}초 동안 말씀해 주세요.")
        question = speech_to_text()
        print(f"[질문] {question}")

        # 무음이면 Whisper가 [BLANK_AUDIO] 같은 토큰만 돌려줌
        if not question or "BLANK_AUDIO" in question:
            text_to_speech("잘 못 들었어요. 다시 말씀해 주세요.")
            continue

        memory_text = load_memory_text()
        print(f"[기억]\n{memory_text}")

        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{memory_text}\n\nQuestion:\n{question}"},
            ],
            max_tokens=MAX_TOKENS,
            temperature=0.3,
        )

        answer = response["choices"][0]["message"]["content"].strip()
        print(f"[답변] {answer}")

        text_to_speech(answer)


if __name__ == "__main__":
    main()
