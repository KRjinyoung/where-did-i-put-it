"""음성으로 물건 위치를 묻고 음성으로 답하는 도우미.

STT(Whisper) → Gemma(찾는 물건 고르기) → Gemma 멀티모달(기억 + 마지막 장면 사진) → TTS(Piper)
vision_tracker.py가 저장한 item_memory.json과 last_frame_<class_id>.jpg를 읽기만 한다.

실행 (이 폴더에서, vision_tracker.py와 다른 터미널): .venv/bin/python voice_assistant.py
"""

import base64
import json
import os
import re
import subprocess
import time

from llama_cpp import Llama
from llama_cpp.llama_chat_format import Gemma4ChatHandler


# 모든 경로는 이 파일이 있는 폴더 기준 → 어디서 실행해도 이 폴더 안의 모델·결과만 사용
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

GEMMA_MODEL_PATH = os.path.join(BASE_DIR, "src/models/Gemma4/google_gemma-4-E2B-it-Q4_K_M.gguf")
MMPROJ_PATH = os.path.join(BASE_DIR, "src/models/Gemma4/mmproj-google_gemma-4-E2B-it-f16.gguf")

# vision_tracker.py와 같은 output/ 폴더 (기억 JSON, 사진, 녹음·응답 음성)
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MEMORY_PATH = os.path.join(OUTPUT_DIR, "item_memory.json")

MIC_DEVICE = "plughw:CARD=Microphone,DEV=0"  # card 번호는 부팅마다 바뀌어서 이름으로 지정 (arecord -l 의 [ ] 앞 이름)
AUDIO_FILE = os.path.join(OUTPUT_DIR, "input.wav")
WHISPER_DIR = os.path.join(BASE_DIR, "whisper.cpp/build-cpu/bin")
WHISPER_MODEL = os.path.join(BASE_DIR, "whisper.cpp/models/ggml-base.bin")
RECORD_SECONDS = 5

PIPER_PYTHON = os.path.join(BASE_DIR, ".piper_venv/bin/python")
PIPER_MODEL = os.path.join(BASE_DIR, "src/models/Piper/ko_KR-kss-medium.onnx")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "response.wav")

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
        # whisper-cli는 같은 폴더의 libwhisper.so 등을 써야 함 (빌드한 원래 위치가 아니라 이 폴더 것으로)
        env=dict(os.environ, LD_LIBRARY_PATH=WHISPER_DIR),
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


def ago_text(seconds):
    if seconds < 60:
        return "방금 전"
    if seconds < 3600:
        return f"{int(seconds // 60)}분 전"
    return f"{int(seconds // 3600)}시간 전"


def item_to_text(item):
    """물건 하나의 기억을 Gemma에게 줄 한 줄 문장으로. 예: '휴대폰: 2분 전 키보드 근처에서 마지막으로 봄.'"""
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

    # 1. 기억한 물건 중 찾는 물건 고르기 (핸드폰/폰 같은 다른 표현도 Gemma가 '휴대폰'으로 맞춰줌)
    target = ask_gemma(llm, FIND_PROMPT, f"목록: {', '.join(items)}\n질문: {question}", 10).strip(" .")
    print(f"[찾는 물건] {target}")

    # 기억에 없으면 Gemma에 묻지 않고 고정 답변 → 위치를 지어낼 수 없음
    if target not in items:
        return "그 물건은 아직 본 적이 없어요."

    item = items[target]
    memory_text = item_to_text(item)
    print(f"[기억] {memory_text}")

    # 2. 기억 문장 + 마지막 장면 사진을 같이 넣어 위치 설명 (트래커가 사진을 JSON보다 먼저 저장하므로 항상 있음)
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

    # 첫 호출은 GPU 준비 때문에 30초 가까이 걸려서, 질문 받기 전에 한 번 미리 돌려둠
    ask_gemma(llm, "한 단어로 답하시오.", "안녕", 1)
    print("준비 완료")

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

        # 무음·잡음이면 Whisper가 [BLANK_AUDIO], [끝] 같은 괄호 토큰만 돌려줌 → 괄호를 빼고 남는 말이 없으면 다시 묻기
        if not re.sub(r"\[.*?\]|\(.*?\)", "", question).strip():
            text_to_speech("잘 못 들었어요. 다시 말씀해 주세요.")
            continue

        start = time.time()
        answer = answer_question(llm, question)
        print(f"[답변] {answer}  ({time.time() - start:.1f}초)")

        text_to_speech(answer)


if __name__ == "__main__":
    main()
