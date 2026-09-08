import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import yt_dlp
from dotenv import load_dotenv
from google import genai
from google.genai import types
from yt_dlp.utils import DownloadCancelled

MODEL = "gemini-3.5-transcribe"
REQUEST_TIMEOUT_MS = 600_000
CLEANUP_TIMEOUT_MS = 15_000


class TranscriptionCancelled(Exception):
    """Raised when the user requests cancellation."""


def download_audio(url, temp_dir, cookies_file=None, stop_event=None, progress_callback=None):
    output_template = str(Path(temp_dir) / "%(id)s.%(ext)s")

    def progress_hook(data):
        if stop_event and stop_event.is_set():
            raise DownloadCancelled("Остановлено пользователем")
        if progress_callback and data.get("status") == "downloading":
            downloaded = data.get("downloaded_bytes") or 0
            total = data.get("total_bytes") or data.get("total_bytes_estimate")
            progress_callback(downloaded, total)

    ydl_opts = {
        "format": "bestaudio[ext=m4a]/bestaudio/best",
        "outtmpl": output_template,
        "noplaylist": True,
        "socket_timeout": 60,
        "retries": 10,
        "fragment_retries": 10,
        "file_access_retries": 3,
        "continuedl": True,
        "overwrites": True,
        "progress_hooks": [progress_hook],
        "postprocessor_hooks": [progress_hook],
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "m4a",
            }
        ],
    }
    if cookies_file:
        ydl_opts["cookiefile"] = str(cookies_file)

    node_path = shutil.which("node")
    if node_path:
        # Setting the option replaces the yt-dlp default, so keep Deno enabled too.
        ydl_opts["js_runtimes"] = {"deno": {}, "node": {"path": node_path}}

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            if stop_event and stop_event.is_set():
                raise TranscriptionCancelled()
            audio_path = _downloaded_audio_path(ydl, info, temp_dir)
    except DownloadCancelled as error:
        raise TranscriptionCancelled() from error
    except yt_dlp.utils.DownloadError as error:
        if stop_event and stop_event.is_set():
            raise TranscriptionCancelled() from error
        raise

    if stop_event and stop_event.is_set():
        raise TranscriptionCancelled()
    return audio_path


def _downloaded_audio_path(ydl, info, temp_dir):
    """Return the audio file yt-dlp reported after postprocessing."""
    for download in info.get("requested_downloads") or ():
        filepath = download.get("filepath")
        if filepath and Path(filepath).is_file():
            return Path(filepath)

    audio_path = Path(ydl.prepare_filename(info)).with_suffix(".m4a")
    if audio_path.is_file():
        return audio_path

    audio_files = list(Path(temp_dir).glob("*.m4a"))
    if len(audio_files) != 1:
        raise RuntimeError("yt-dlp не создал аудиофайл в формате M4A")
    return audio_files[0]


def delete_uploaded_file(client, audio_file):
    """Remove the uploaded audio from the Gemini Files API; cleanup must not fail a run."""
    name = getattr(audio_file, "name", None)
    if not name:
        return
    try:
        client.files.delete(
            name=name,
            config=types.DeleteFileConfig(
                http_options=types.HttpOptions(timeout=CLEANUP_TIMEOUT_MS)
            ),
        )
    except Exception as error:
        print(f"Не удалось удалить файл {name} из Gemini: {error}", file=sys.stderr)


def format_error_message(error, phase=None):
    """Convert common dependency errors into actionable Russian messages."""
    message = str(error).strip()
    lower = message.lower()
    context = f" на этапе «{phase}»" if phase else ""
    error_code = getattr(error, "code", None)

    if isinstance(error, TranscriptionCancelled):
        return "Операция остановлена пользователем."
    if "sign in to confirm" in lower or "not a bot" in lower or "captcha" in lower:
        return (
            "YouTube заблокировал загрузку как запрос от бота. "
            "Экспортируйте cookies в формате Netscape и укажите путь к файлу "
            "через YTDLP_COOKIES_FILE в .env."
        )
    if "javascript" in lower or "n challenge" in lower or "yt-dlp-ejs" in lower:
        return (
            "Для этой ссылки yt-dlp требует JavaScript runtime и yt-dlp-ejs. "
            "Установите Node.js и зависимость yt-dlp-ejs, затем повторите попытку."
        )
    if "ffmpeg" in lower or "postprocessor" in lower:
        return "Не найден или не запустился FFmpeg. Установите FFmpeg и добавьте его в PATH."
    if (
        error_code == 401
        or "http error 401" in lower
        or "api key" in lower
        or "api_key" in lower
        or "unauthenticated" in lower
    ):
        if phase in ("отправка файла в Gemini", "распознавание речи"):
            return "Ошибка API-ключа Gemini. Проверьте GEMINI_API_KEY в файле .env."
        return "Источник требует авторизацию. Проверьте cookies и доступность ссылки."
    if error_code == 403 or "http error 403" in lower or "permission denied" in lower:
        if phase in ("отправка файла в Gemini", "распознавание речи"):
            return "Gemini отклонил запрос. Проверьте права доступа к API."
        return "Доступ к источнику запрещён. Проверьте cookies и доступность ссылки."
    if error_code == 429 or "http error 429" in lower or "quota" in lower or "rate limit" in lower or "resource_exhausted" in lower:
        return "Превышена квота или лимит запросов Gemini. Проверьте тариф и повторите позже."
    if "authentication" in lower:
        return "Ошибка авторизации Gemini. Проверьте GEMINI_API_KEY и права доступа к API."
    if isinstance(error_code, int) and error_code >= 500:
        return f"Сервис Gemini временно недоступен{context}. Повторите попытку позже."
    if any(term in lower for term in ("timed out", "timeout", "timedout", "read timed out")):
        return f"Истёк тайм-аут{context}. Проверьте интернет-соединение и повторите попытку."
    if any(term in lower for term in ("connection reset", "connection aborted", "connection refused", "temporary failure", "network is unreachable")):
        return f"Сетевая ошибка{context}. Проверьте интернет-соединение и повторите попытку."
    if "no video formats" in lower or "requested format is not available" in lower:
        return "Для этой ссылки не найден доступный аудиоформат. Ссылка может быть недоступна или требовать авторизацию."
    if "private video" in lower or "video unavailable" in lower or "not available" in lower:
        return "Ссылка недоступна: видео удалено, скрыто или требует авторизацию."
    if any(term in lower for term in ("unsupported audio", "unsupported format", "invalid audio", "corrupt", "mimetype")):
        return "Аудиофайл повреждён или имеет неподдерживаемый формат. Выберите другой файл."
    if isinstance(error, FileNotFoundError):
        return f"Файл не найден{context}. Проверьте путь к аудиофайлу."
    if isinstance(error, PermissionError):
        return f"Нет доступа к файлу{context}. Проверьте права и что файл не занят другой программой."
    if isinstance(error, (ValueError, OSError)) and phase == "подготовка":
        return f"Не удалось подготовить файлы: {message or 'неизвестная ошибка'}."
    if message:
        return f"Ошибка{context}: {message}"
    return f"Неизвестная ошибка{context}."


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


def _response_value(value, name, default=None):
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _text_blocks(interaction):
    """Yield (text, word annotations) for every text block of the model output."""
    for step in _response_value(interaction, "steps", []) or []:
        if _response_value(step, "type") != "model_output":
            continue
        for content in _response_value(step, "content", []) or []:
            if _response_value(content, "type") != "text":
                continue
            text = _response_value(content, "text", "")
            annotations = [
                annotation
                for annotation in _response_value(content, "annotations", []) or []
                if _response_value(annotation, "type") == "word_info"
            ]
            yield (text if isinstance(text, str) else ""), annotations


def extract_word_annotations(interaction):
    annotations = []
    for _text, block_annotations in _text_blocks(interaction):
        annotations.extend(block_annotations)
    return annotations


def extract_output_text(interaction):
    output_text = _response_value(interaction, "output_text", "")
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    return "".join(text for text, _annotations in _text_blocks(interaction))


def _byte_slice(raw, start, end):
    """Decode raw[start:end] when the annotation offsets describe a valid UTF-8 span."""
    if not isinstance(start, int) or not isinstance(end, int):
        return None
    if not 0 <= start < end <= len(raw):
        return None
    try:
        return raw[start:end].decode("utf-8")
    except UnicodeDecodeError:
        return None


def _annotation_span(raw, first, last, fallback_words):
    text = _byte_slice(
        raw,
        _response_value(first, "start_index"),
        _response_value(last, "end_index"),
    )
    if text is None:
        text = " ".join(word for word in fallback_words if word)
    return text.strip()


def _word_text(raw, annotation):
    text = _byte_slice(
        raw,
        _response_value(annotation, "start_index"),
        _response_value(annotation, "end_index"),
    )
    if text is None:
        text = _response_value(annotation, "text", "")
    return text if isinstance(text, str) else ""


def _ends_sentence(word):
    stripped = word.rstrip().rstrip("\"'»”’)]")
    return bool(stripped) and stripped[-1] in ".!?…"


def _format_annotated_block(text, annotations, include_speakers, include_timestamps):
    """Split one text block into speaker turns and sentences with the requested prefixes."""
    raw = text.encode("utf-8")
    lines = []
    segment = []
    words = []

    def flush():
        if not segment:
            return
        body = _annotation_span(raw, segment[0], segment[-1], words)
        if body:
            prefix = []
            if include_speakers:
                speaker = _response_value(segment[0], "speaker")
                if speaker:
                    prefix.append(f"[{speaker}]")
            if include_timestamps:
                start = _response_value(segment[0], "start_offset")
                end = _response_value(segment[-1], "end_offset")
                if start and end:
                    prefix.append(f"({start} - {end})")
            lines.append(" ".join([*prefix, body]))
        segment.clear()
        words.clear()

    for annotation in annotations:
        if segment and _response_value(annotation, "speaker") != _response_value(segment[0], "speaker"):
            flush()
        word = _word_text(raw, annotation)
        segment.append(annotation)
        words.append(word)
        if include_timestamps and _ends_sentence(word):
            flush()
    flush()
    return lines


def format_annotated_transcript(interaction, include_speakers, include_timestamps):
    if not include_speakers and not include_timestamps:
        return extract_output_text(interaction)

    lines = []
    for text, annotations in _text_blocks(interaction):
        if annotations:
            lines.extend(
                _format_annotated_block(text, annotations, include_speakers, include_timestamps)
            )
        elif text.strip():
            lines.append(text.strip())
    if not lines:
        return extract_output_text(interaction)
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

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS),
    )
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

    try:
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
    finally:
        delete_uploaded_file(client, audio_file)

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
