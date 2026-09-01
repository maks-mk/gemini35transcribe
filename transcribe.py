import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai

MODEL = "gemini-3.5-transcribe"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Транскрибация аудио с помощью Gemini 3.5 Transcribe."
    )
    parser.add_argument("audio_file", type=Path, help="Путь к аудиофайлу")
    parser.add_argument(
        "--mode",
        choices=("verbatim", "smart"),
        default="verbatim",
        help="Режим: verbatim сохраняет речь, smart убирает слова-паразиты и форматирует текст.",
    )
    parser.add_argument(
        "--language",
        action="append",
        dest="languages",
        metavar="BCP-47",
        help="Язык в формате BCP-47, например ru-RU. Можно указать несколько раз.",
    )
    parser.add_argument(
        "--speaker-diarization",
        action="store_true",
        help="Добавить идентификаторы говорящих (до 8 говорящих).",
    )
    parser.add_argument(
        "--timestamps",
        action="store_true",
        help="Запросить таймстемпы каждого слова и сохранить полный ответ в JSON.",
    )
    parser.add_argument(
        "--vocabulary",
        action="append",
        dest="vocabulary",
        metavar="TERM",
        help="Термин, имя или аббревиатура для custom vocabulary. Можно указать до 1000 раз.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Путь к TXT-файлу результата. По умолчанию рядом с аудио.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Сохранить полный JSON-ответ API, включая аннотации.",
    )
    return parser.parse_args()


def extract_word_annotations(interaction):
    annotations = []
    for step in getattr(interaction, "steps", []) or []:
        for content in getattr(step, "content", []) or []:
            for annotation in getattr(content, "annotations", []) or []:
                if getattr(annotation, "type", None) == "word_info":
                    annotations.append(annotation)
    return annotations


def format_annotated_transcript(interaction, include_speakers, include_timestamps):
    annotations = extract_word_annotations(interaction)
    if not annotations:
        return interaction.output_text or ""

    lines = []
    current_speaker = None
    current_words = []
    current_start = None
    current_end = None

    def flush_segment():
        if not current_words:
            return
        prefix = []
        if include_speakers and current_speaker:
            prefix.append(f"[{current_speaker}]")
        if include_timestamps and current_start and current_end:
            prefix.append(f"({current_start} - {current_end})")
        lines.append(f"{' '.join(prefix)} {' '.join(current_words)}".strip())

    for annotation in annotations:
        speaker = getattr(annotation, "speaker", None)
        if current_words and speaker != current_speaker:
            flush_segment()
            current_words = []
            current_start = None
            current_end = None

        if not current_words:
            current_speaker = speaker
            current_start = getattr(annotation, "start_offset", None)
        current_words.append(getattr(annotation, "text", ""))
        current_end = getattr(annotation, "end_offset", None)

    flush_segment()
    return "\n".join(lines)


def main():
    args = parse_args()
    audio_path = args.audio_file

    if not audio_path.exists():
        print(f"Файл не найден: {audio_path}")
        sys.exit(1)

    if len(args.vocabulary or []) > 1000:
        print("Можно указать не более 1000 терминов в --vocabulary")
        sys.exit(1)

    if (args.speaker_diarization or args.timestamps) and args.mode == "smart":
        print("--speaker-diarization и --timestamps требуют режима --mode verbatim")
        sys.exit(1)

    load_dotenv(Path(__file__).with_name(".env"))
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        print("Ключ GEMINI_API_KEY не найден в .env")
        sys.exit(1)

    client = genai.Client(api_key=api_key)
    output_path = args.output or audio_path.with_suffix(".txt")

    transcription_config = {}
    if args.languages:
        transcription_config["language_codes"] = args.languages
    if args.vocabulary:
        transcription_config["custom_vocabulary"] = args.vocabulary

    mode = {"type": args.mode}
    if args.speaker_diarization:
        mode["diarization_mode"] = "speaker"
    if args.timestamps:
        mode["timestamp_granularities"] = ["word"]
    transcription_config["mode"] = mode

    print(f"Файл: {audio_path}")
    print("Загрузка...")

    start = time.perf_counter()
    audio_file = client.files.upload(file=str(audio_path))

    print("Транскрипция...")
    transcription_start = time.perf_counter()

    interaction = client.interactions.create(
        model=MODEL,
        input=[
            {
                "type": "audio",
                "uri": audio_file.uri,
                "mime_type": audio_file.mime_type,
            }
        ],
        generation_config={"transcription_config": transcription_config},
    )

    transcription_elapsed = time.perf_counter() - transcription_start
    elapsed = time.perf_counter() - start

    text = format_annotated_transcript(
        interaction,
        include_speakers=args.speaker_diarization,
        include_timestamps=args.timestamps,
    )
    output_path.write_text(text, encoding="utf-8")

    if args.json_output:
        args.json_output.write_text(
            json.dumps(interaction.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print()
    print("Готово.")
    print(f"Транскрипция: {transcription_elapsed:.2f} сек.")
    print(f"Общее время (загрузка + транскрипция): {elapsed:.2f} сек.")
    print(f"Результат: {output_path}")
    if args.json_output:
        print(f"JSON: {args.json_output}")
    print()
    print(text)


if __name__ == "__main__":
    main()
