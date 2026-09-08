import asyncio
import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

if sys.platform == "win32":
    import ctypes

import qtawesome as qta
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PySide6.QtCore import QObject, QThread, Qt, Signal, Slot
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QFont
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QButtonGroup,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSplitter,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from transcribe import (
    MODEL,
    REQUEST_TIMEOUT_MS,
    TranscriptionCancelled,
    delete_uploaded_file,
    download_audio,
    format_annotated_transcript,
    format_error_message,
)


def _load_api_key():
    """Load the API key from .env located beside the source or packaged EXE."""
    if getattr(sys, "frozen", False):
        env_path = Path(sys.executable).resolve().with_name(".env")
    else:
        env_path = Path(__file__).resolve().with_name(".env")

    load_dotenv(env_path)
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    return api_key, env_path


class TranscriptionWorker(QObject):
    finished = Signal(str, str, float, float)
    failed = Signal(str)
    stopped = Signal()
    status = Signal(str)
    progress = Signal(int)

    def __init__(self, source, source_is_url, output_path, json_path, config):
        super().__init__()
        self.source = source
        self.source_is_url = source_is_url
        self.output_path = output_path
        self.json_path = json_path
        self.config = config
        self.stop_event = threading.Event()
        self.phase = "подготовка"
        self._loop = None
        self._task = None
        self._uploaded_file = None

    def request_stop(self):
        self.stop_event.set()
        loop = self._loop
        task = self._task
        if loop and task and not task.done():
            try:
                loop.call_soon_threadsafe(task.cancel)
            except RuntimeError:
                pass

    def _check_stopped(self):
        if self.stop_event.is_set():
            raise TranscriptionCancelled()

    def _download_progress(self, downloaded, total):
        if total:
            percent = round(min(downloaded, total) * 100 / total)
            self.progress.emit(percent)

    def _download_audio(self, temp_dir, cookies_file=None):
        self.phase = "загрузка аудиодорожки"
        self.status.emit("Загрузка аудиодорожки…")
        self.progress.emit(-1)
        return download_audio(
            self.source,
            temp_dir,
            cookies_file=cookies_file,
            stop_event=self.stop_event,
            progress_callback=self._download_progress,
        )

    async def _run_gemini(self, client, audio_path):
        self._loop = asyncio.get_running_loop()
        self._task = asyncio.current_task()
        try:
            self._check_stopped()
            self._uploaded_file = await client.aio.files.upload(file=str(audio_path))
            self._check_stopped()

            self.phase = "распознавание речи"
            self.status.emit("Распознавание речи…")
            transcription_started = time.perf_counter()
            self._check_stopped()
            interaction = await client.aio.interactions.create(
                model=MODEL,
                input=[
                    {
                        "type": "audio",
                        "uri": self._uploaded_file.uri,
                        "mime_type": self._uploaded_file.mime_type,
                    }
                ],
                generation_config={"transcription_config": self.config},
            )
            self._check_stopped()
            return interaction, time.perf_counter() - transcription_started
        finally:
            try:
                await client.aio.aclose()
            except Exception as error:
                print(f"Не удалось закрыть async-клиент Gemini: {error}", file=sys.stderr)
            self._task = None
            self._loop = None

    def _write_outputs(self, text, interaction):
        """Publish results atomically so a cancelled run leaves no partial files."""
        payloads = [(self.output_path, text)]
        if self.json_path:
            payloads.append((
                self.json_path,
                json.dumps(interaction.model_dump(mode="json"), ensure_ascii=False, indent=2),
            ))

        staged = []
        try:
            for path, data in payloads:
                temp_path = path.with_name(f"{path.name}.part")
                temp_path.write_text(data, encoding="utf-8")
                staged.append((temp_path, path))
            self._check_stopped()
            while staged:
                temp_path, path = staged.pop()
                os.replace(temp_path, path)
        finally:
            for temp_path, _path in staged:
                temp_path.unlink(missing_ok=True)

    @Slot()
    def run(self):
        temp_dir = None
        client = None
        try:
            self.phase = "подготовка"
            api_key, env_path = _load_api_key()
            if not api_key:
                raise RuntimeError(f"Ключ GEMINI_API_KEY не найден в файле {env_path}")
            self._check_stopped()

            started = time.perf_counter()
            audio_path = Path(self.source)
            if self.source_is_url:
                temp_dir = tempfile.TemporaryDirectory(prefix="gemini-transcribe-")
                cookies_file = os.getenv("YTDLP_COOKIES_FILE", "").strip()
                if cookies_file:
                    cookies_path = Path(cookies_file)
                    if not cookies_path.is_absolute():
                        cookies_path = env_path.parent / cookies_file
                    if not cookies_path.is_file():
                        raise RuntimeError(f"Файл cookies не найден: {cookies_path}")
                else:
                    cookies_path = None
                audio_path = self._download_audio(temp_dir.name, cookies_path)
            else:
                self._check_stopped()
                self.status.emit("Загрузка аудиофайла…")

            self._check_stopped()
            self.phase = "отправка файла в Gemini"
            self.status.emit("Отправка аудиофайла в Gemini…")
            self.progress.emit(-1)
            client = genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS),
            )
            interaction, transcription_elapsed = asyncio.run(
                self._run_gemini(client, audio_path)
            )

            elapsed = time.perf_counter() - started
            include_speakers = self.config["mode"].get("diarization_mode") == "speaker"
            include_timestamps = bool(self.config["mode"].get("timestamp_granularities"))
            text = format_annotated_transcript(
                interaction,
                include_speakers=include_speakers,
                include_timestamps=include_timestamps,
            )
            if not text.strip():
                raise RuntimeError("Gemini вернул пустую расшифровку без текста в ответе")
            self._check_stopped()
            self._write_outputs(text, interaction)
            self.finished.emit(text, str(self.output_path), transcription_elapsed, elapsed)
        except (TranscriptionCancelled, asyncio.CancelledError):
            self.stopped.emit()
        except Exception as error:  # API and yt-dlp errors vary by version.
            if self.stop_event.is_set():
                self.stopped.emit()
            else:
                print(f"Ошибка на этапе {self.phase}: {error}", file=sys.stderr)
                self.failed.emit(format_error_message(error, self.phase))
        finally:
            if client:
                delete_uploaded_file(client, self._uploaded_file)
                try:
                    client.close()
                except Exception as error:
                    print(f"Не удалось закрыть клиент Gemini: {error}", file=sys.stderr)
            if temp_dir:
                temp_dir.cleanup()


class AudioDropEdit(QLineEdit):
    file_dropped = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls and urls[0].isLocalFile():
            self.file_dropped.emit(urls[0].toLocalFile())
            event.acceptProposedAction()
        else:
            event.ignore()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.thread = None
        self.worker = None
        self._cancel_requested = False
        self._close_pending = False
        self.setWindowTitle("Gemini Transcribe")
        self.setMinimumSize(640, 460)
        self.setSizeIncrement(1, 1)
        available = QApplication.primaryScreen().availableGeometry()
        self.resize(
            min(1040, max(640, available.width() - 40)),
            min(760, max(460, available.height() - 60)),
        )
        self.move(
            available.left() + (available.width() - self.width()) // 2,
            self.y(),
        )
        self._build_ui()
        self._apply_theme()

    def _build_ui(self):
        content = QWidget()
        root = QVBoxLayout(content)
        root.setContentsMargins(14, 12, 14, 12)
        root.setSpacing(8)

        header = QHBoxLayout()
        title_box = QVBoxLayout()
        title = QLabel("Транскрибация аудио")
        title.setObjectName("title")
        subtitle = QLabel("Gemini 3.5 Transcribe · текст из аудиозаписей")
        subtitle.setObjectName("subtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header.addLayout(title_box)
        header.addStretch()
        self.badge = QLabel("ГОТОВО")
        self.badge.setObjectName("badge")
        header.addWidget(self.badge, alignment=Qt.AlignTop)
        root.addLayout(header)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(8)

        settings = QWidget()
        settings.setObjectName("settingsPane")
        settings.setMinimumWidth(0)
        settings_layout = QVBoxLayout(settings)
        settings_layout.setContentsMargins(0, 0, 6, 0)
        settings_layout.setSpacing(6)

        file_card = self._card("Источник", "Файл или ссылка на видеохостинг")
        source_mode_row = QHBoxLayout()
        source_mode_row.setContentsMargins(0, 0, 0, 0)
        source_mode_row.setSpacing(12)
        self.file_source_check = QCheckBox("Локальный файл")
        self.url_source_check = QCheckBox("Ссылка")
        self.file_source_check.setChecked(True)
        self.file_source_check.setToolTip("Транскрибировать аудиофайл с компьютера")
        self.url_source_check.setToolTip("Скачать аудиодорожку по ссылке через yt-dlp")
        self.source_group = QButtonGroup(self)
        self.source_group.setExclusive(True)
        self.source_group.addButton(self.file_source_check)
        self.source_group.addButton(self.url_source_check)
        self.file_source_check.toggled.connect(self._source_mode_changed)
        source_mode_row.addWidget(self.file_source_check)
        source_mode_row.addWidget(self.url_source_check)
        source_mode_row.addStretch()
        file_card.layout().addLayout(source_mode_row)

        file_row = QHBoxLayout()
        file_row.setSpacing(6)
        self.audio_edit = AudioDropEdit()
        self.audio_edit.setPlaceholderText("meeting.mp3")
        self.audio_edit.setMinimumWidth(0)
        self.audio_edit.setToolTip("Аудиофайл для транскрибации. Поддерживаются MP3, WAV, M4A, FLAC, OGG и AAC")
        self.audio_edit.file_dropped.connect(self._set_audio_path)
        self.browse_source = self._button("fa6s.folder-open", "Обзор", "secondary")
        self.browse_source.clicked.connect(self._source_action)
        file_row.addWidget(self.audio_edit)
        file_row.addWidget(self.browse_source)
        file_card.layout().addLayout(file_row)
        settings_layout.addWidget(file_card)

        options_card = self._card("Параметры", "Формат и точность результата")
        form = QFormLayout()
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(4)
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Дословно", "verbatim")
        self.mode_combo.addItem("Умное форматирование", "smart")
        self.mode_combo.setMinimumWidth(0)
        self.mode_combo.setToolTip("Выберите режим: дословная расшифровка или умное форматирование текста")
        self.mode_combo.currentIndexChanged.connect(self._mode_changed)
        self.language_edit = QLineEdit()
        self.language_edit.setPlaceholderText("Авто (ru-RU, en-US)")
        self.language_edit.setMinimumWidth(0)
        self.language_edit.setToolTip("BCP-47 коды языков через запятую, например: ru-RU, en-US. Пусто — определить автоматически")
        self.speaker_check = QCheckBox("Говорящие")
        self.speaker_check.setToolTip("Разделять расшифровку по говорящим")
        self.timestamp_check = QCheckBox("Таймстемпы")
        self.timestamp_check.setToolTip("Добавлять временные отметки в расшифровку")
        self.vocabulary_edit = QLineEdit()
        self.vocabulary_edit.setPlaceholderText("Например: Иван Иванов, API, Kubernetes, специальные термины")
        self.vocabulary_edit.setMinimumWidth(0)
        self.vocabulary_edit.setToolTip("Имена, аббревиатуры и специальные термины через запятую")
        form.addRow("Режим", self.mode_combo)
        form.addRow("Язык", self.language_edit)
        options_row = QWidget()
        options_row.setObjectName("optionsRow")
        options_layout = QHBoxLayout(options_row)
        options_layout.setContentsMargins(0, 0, 0, 0)
        options_layout.setSpacing(8)
        options_layout.addWidget(self.speaker_check)
        options_layout.addWidget(self.timestamp_check)
        options_layout.addStretch()
        form.addRow("Опции", options_row)
        form.addRow("Словарь", self.vocabulary_edit)
        options_card.layout().addLayout(form)
        settings_layout.addWidget(options_card)

        output_card = self._card("Сохранение", "TXT обязателен, JSON — дополнительно")
        output_form = QFormLayout()
        output_form.setHorizontalSpacing(8)
        output_form.setVerticalSpacing(7)
        self.output_edit = QLineEdit()
        self.output_edit.setMinimumWidth(0)
        self.output_edit.setToolTip("Путь и имя обязательного TXT-файла с результатом")
        self.json_edit = QLineEdit()
        self.json_edit.setMinimumWidth(0)
        self.json_edit.setToolTip("Необязательный путь и имя JSON-файла с результатом")
        output_txt = self._button("fa6s.file-lines", "…", "compact")
        output_json = self._button("fa6s.file-code", "…", "compact")
        output_txt.clicked.connect(self._browse_output)
        output_json.clicked.connect(self._browse_json)
        output_form.addRow("TXT", self._with_button(self.output_edit, output_txt))
        output_form.addRow("JSON", self._with_button(self.json_edit, output_json))
        output_card.layout().addLayout(output_form)
        settings_layout.addWidget(output_card)

        action_card = QFrame()
        action_card.setObjectName("actionCard")
        action_layout = QHBoxLayout(action_card)
        action_layout.setContentsMargins(12, 10, 12, 10)
        action_layout.setSpacing(8)
        self.start_button = self._button("fa6s.play", "Начать", "primary")
        self.start_button.clicked.connect(self._start_or_stop)
        action_layout.addWidget(self.start_button)
        self.status_label = QLabel("Выберите аудиофайл")
        self.status_label.setObjectName("status")
        self.status_label.setWordWrap(True)
        action_layout.addWidget(self.status_label, 1)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setFixedWidth(70)
        self.progress.hide()
        action_layout.addWidget(self.progress)
        settings_layout.addWidget(action_card)
        settings_layout.addStretch()

        result_panel = QWidget()
        result_panel.setObjectName("resultPanel")
        result_panel.setMinimumWidth(0)
        result_layout = QVBoxLayout(result_panel)
        result_layout.setContentsMargins(6, 0, 0, 0)
        result_layout.setSpacing(8)
        result_header = QHBoxLayout()
        result_title = QLabel("Расшифровка")
        result_title.setObjectName("sectionTitle")
        result_header.addWidget(result_title)
        result_header.addStretch()
        result_hint = QLabel("Можно копировать")
        result_hint.setObjectName("hint")
        result_header.addWidget(result_hint)
        result_layout.addLayout(result_header)
        self.result_edit = QPlainTextEdit()
        self.result_edit.setReadOnly(True)
        self.result_edit.setPlaceholderText("Текст транскрипции появится здесь…")
        result_layout.addWidget(self.result_edit, 1)

        splitter.addWidget(settings)
        splitter.addWidget(result_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([450, 650])
        root.addWidget(splitter, 1)

        self.setCentralWidget(content)

    def _card(self, title, description):
        card = QFrame()
        card.setObjectName("card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 9, 12, 11)
        layout.setSpacing(6)
        heading = QHBoxLayout()
        heading.setSpacing(8)
        label = QLabel(title)
        label.setObjectName("sectionTitle")
        hint = QLabel(description)
        hint.setObjectName("hint")
        hint.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        heading.addWidget(label)
        heading.addWidget(hint, 1)
        layout.addLayout(heading)
        return card

    @staticmethod
    def _with_button(widget, button):
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(widget)
        layout.addWidget(button)
        return row

    def _button(self, icon_name, text, kind):
        button = QPushButton(qta.icon(icon_name, color="#b8c0cc"), text)
        button.setProperty("kind", kind)
        return button

    def _apply_theme(self):
        self.setStyleSheet("""
            QWidget { background: #17191d; color: #e2e5e9; font-size: 13px; }
            QMainWindow { background: #17191d; }
            QScrollArea, QScrollArea > QWidget > QWidget { background: #17191d; border: 0; }
            QScrollBar:vertical { background: #1d2025; width: 10px; margin: 2px; }
            QScrollBar::handle:vertical { background: #454d58; border-radius: 5px; min-height: 30px; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
            QLabel { background: transparent; }
            QWidget#optionsRow { background: transparent; }
            QLabel#title { color: #f1f3f5; font-size: 22px; font-weight: 600; }
            QLabel#subtitle, QLabel#hint { color: #8d96a3; }
            QLabel#sectionTitle { color: #eef0f2; font-size: 14px; font-weight: 600; }
            QLabel#status { color: #9fa8b4; }
            QLabel#badge { background: #29332f; color: #9cc4aa; border-radius: 10px; padding: 5px 10px; font-size: 11px; font-weight: 600; }
            QFrame#card, QFrame#actionCard { background: #202328; border: 1px solid #30353d; border-radius: 8px; }
            QLineEdit, QComboBox, QPlainTextEdit { background: #191b1f; border: 1px solid #3a414b; border-radius: 5px; padding: 5px 7px; color: #e2e5e9; selection-background-color: #566b83; }
            QLineEdit, QComboBox { min-height: 26px; }
            QLineEdit:focus, QComboBox:focus, QPlainTextEdit:focus { border: 1px solid #71859a; }
            QComboBox::drop-down { border: 0; width: 28px; }
            QComboBox QAbstractItemView { background: #252930; border: 1px solid #454c56; selection-background-color: #3d4b5b; }
            QCheckBox { background: transparent; spacing: 8px; color: #cbd0d6; }
            QCheckBox::indicator { width: 16px; height: 16px; border: 1px solid #58616e; border-radius: 3px; background: #191b1f; }
            QCheckBox::indicator:checked { background: #71859a; border-color: #8b9eb0; }
            QPushButton { min-height: 29px; padding: 0 11px; border-radius: 5px; border: 1px solid #424953; background: #2a2e35; color: #e4e7eb; }
            QPushButton:hover { background: #343a43; }
            QPushButton:disabled { color: #69717c; background: #25282e; }
            QPushButton[kind="primary"] { background: #52677c; border-color: #687f95; font-weight: 600; min-width: 205px; }
            QPushButton[kind="primary"]:hover { background: #60778e; }
            QPushButton[kind="compact"] { min-height: 31px; padding: 0 10px; }
            QProgressBar { border: 0; background: #2c3138; border-radius: 3px; max-width: 180px; height: 6px; text-align: center; }
            QProgressBar::chunk { background: #71859a; border-radius: 3px; }
        """)

    def _source_mode_changed(self, file_mode):
        if file_mode:
            self.audio_edit.setPlaceholderText("meeting.mp3")
            self.audio_edit.setToolTip("Аудиофайл для транскрибации. Поддерживаются MP3, WAV, M4A, FLAC, OGG и AAC")
            self.browse_source.setText("Обзор")
            self.browse_source.setIcon(qta.icon("fa6s.folder-open", color="#b8c0cc"))
        else:
            self.audio_edit.setPlaceholderText("https://www.youtube.com/watch?v=…")
            self.audio_edit.setToolTip("Ссылка на видео или аудио с YouTube, Rutube, VK и других поддерживаемых yt-dlp сайтов")
            self.browse_source.setText("Вставить")
            self.browse_source.setIcon(qta.icon("fa6s.link", color="#b8c0cc"))

    def _source_action(self):
        if self.url_source_check.isChecked():
            self._paste_url()
        else:
            self._browse_audio()

    def _set_audio_path(self, path):
        if self.url_source_check.isChecked():
            return
        path = Path(path)
        self.audio_edit.setText(str(path))
        if not self.output_edit.text().strip():
            self.output_edit.setText(str(path.with_suffix(".txt")))

    def _paste_url(self):
        self.audio_edit.setFocus()
        self.audio_edit.paste()

    def _browse_audio(self):
        path, _ = QFileDialog.getOpenFileName(self, "Выберите аудиофайл", "", "Аудио (*.mp3 *.wav *.m4a *.flac *.ogg *.aac);;Все файлы (*)")
        if path:
            self._set_audio_path(path)

    def _browse_output(self):
        path, _ = QFileDialog.getSaveFileName(self, "Куда сохранить TXT", self.output_edit.text(), "Текст (*.txt)")
        if path:
            self.output_edit.setText(path)

    def _browse_json(self):
        path, _ = QFileDialog.getSaveFileName(self, "Куда сохранить JSON", self.json_edit.text(), "JSON (*.json)")
        if path:
            self.json_edit.setText(path)

    def _mode_changed(self):
        smart = self.mode_combo.currentData() == "smart"
        self.speaker_check.setEnabled(not smart)
        self.timestamp_check.setEnabled(not smart)
        if smart:
            self.speaker_check.setChecked(False)
            self.timestamp_check.setChecked(False)

    def _start_or_stop(self):
        if self.thread and self.thread.isRunning():
            self._stop()
        else:
            self._start()

    @staticmethod
    def _ensure_writable(path):
        """Fail before the API call if the result cannot be written afterwards."""
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_dir():
            raise IsADirectoryError(f"путь занят каталогом: {path}")
        if path.exists():
            with path.open("a", encoding="utf-8"):
                pass
        else:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".write-check-"):
                pass

    def _start(self):
        source = self.audio_edit.text().strip()
        source_is_url = self.url_source_check.isChecked()
        if not source:
            QMessageBox.warning(self, "Источник не указан", "Укажите путь к аудиофайлу или ссылку на видео.")
            return

        if source_is_url:
            parsed_url = urlparse(source)
            if parsed_url.scheme not in ("http", "https") or not parsed_url.netloc:
                QMessageBox.warning(self, "Некорректная ссылка", "Укажите полную ссылку, начинающуюся с http:// или https://.")
                return
            default_output = Path.cwd() / "transcription.txt"
        else:
            audio_path = Path(source)
            if not audio_path.is_file():
                QMessageBox.warning(self, "Файл не найден", "Выберите существующий аудиофайл.")
                return
            default_output = audio_path.with_suffix(".txt")

        output_path = Path(self.output_edit.text().strip() or default_output)
        json_path = Path(self.json_edit.text().strip()) if self.json_edit.text().strip() else None
        try:
            self._ensure_writable(output_path)
            if json_path:
                self._ensure_writable(json_path)
        except OSError as error:
            QMessageBox.warning(
                self,
                "Не удалось подготовить сохранение",
                format_error_message(error, "подготовка"),
            )
            return


        config = {"mode": {"type": self.mode_combo.currentData()}}
        languages = [item.strip() for item in self.language_edit.text().split(",") if item.strip()]
        vocabulary = [item.strip() for item in self.vocabulary_edit.text().split(",") if item.strip()]
        if languages:
            config["language_codes"] = languages
        if vocabulary:
            if len(vocabulary) > 1000:
                QMessageBox.warning(self, "Слишком большой словарь", "Можно указать не более 1000 терминов.")
                return
            config["custom_vocabulary"] = vocabulary
        if self.speaker_check.isChecked():
            config["mode"]["diarization_mode"] = "speaker"
        if self.timestamp_check.isChecked():
            config["mode"]["timestamp_granularities"] = ["word"]

        self._cancel_requested = False
        self.start_button.setText("Остановить")
        self.start_button.setIcon(qta.icon("fa6s.stop", color="#b8c0cc"))
        self.start_button.setEnabled(True)
        self.progress.setRange(0, 0)
        self.progress.show()
        self.badge.setText("В РАБОТЕ")
        self.status_label.setText("Подготовка…")
        self.result_edit.clear()
        self.thread = QThread(self)
        self.worker = TranscriptionWorker(source, source_is_url, output_path, json_path, config)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.status.connect(self._set_status)
        self.worker.progress.connect(self._set_progress)
        self.worker.finished.connect(self._success)
        self.worker.failed.connect(self._failure)
        self.worker.stopped.connect(self._stopped)
        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.worker.stopped.connect(self.thread.quit)
        self.thread.finished.connect(self._thread_finished)
        self.thread.start()


    @Slot(str)
    def _set_status(self, message):
        if not self._cancel_requested:
            self.status_label.setText(message)

    @Slot(int)
    def _set_progress(self, percent):
        if self._cancel_requested:
            return
        if percent >= 0:
            self.progress.setRange(0, 100)
            self.progress.setValue(percent)
        else:
            self.progress.setRange(0, 0)

    @Slot()
    def _stop(self):
        if not self.worker or not self.thread or not self.thread.isRunning():
            return
        self._cancel_requested = True
        self.start_button.setEnabled(False)
        self.status_label.setText("Остановка операции…")
        self.badge.setText("ОСТАНОВКА")
        self.worker.request_stop()

    @Slot(str, str, float, float)
    def _success(self, text, output, transcription_time, total_time):
        if self._cancel_requested:
            return
        self.result_edit.setPlainText(text)
        self.status_label.setText(f"Готово · распознавание {transcription_time:.1f} с · всего {total_time:.1f} с")
        self.badge.setText("ГОТОВО")
        self.progress.hide()

    @Slot(str)
    def _failure(self, message):
        if self._cancel_requested:
            return
        self.status_label.setText("Ошибка обработки")
        self.badge.setText("ОШИБКА")
        self.progress.hide()
        QMessageBox.critical(self, "Ошибка транскрибации", message)

    @Slot()
    def _stopped(self):
        self._cancel_requested = True
        self.status_label.setText("Остановлено пользователем")
        self.badge.setText("ОСТАНОВЛЕНО")
        self.progress.hide()


    def _thread_finished(self):
        if self.worker:
            self.worker.deleteLater()
        if self.thread:
            self.thread.deleteLater()
        self.worker = None
        self.thread = None
        self._cancel_requested = False
        self.start_button.setText("Начать")
        self.start_button.setIcon(qta.icon("fa6s.play", color="#b8c0cc"))
        self.start_button.setEnabled(True)
        if self._close_pending:
            self.close()

    def closeEvent(self, event):
        if self.thread and self.thread.isRunning():
            self._close_pending = True
            self._stop()
            event.ignore()
            return
        event.accept()



def main():
    if sys.platform == "win32":
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "gemini-transcribe.desktop"
        )

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setFont(QFont("Segoe UI", 10))
    app_icon = qta.icon("fa6s.wave-square", color="#9cc4aa")
    app.setWindowIcon(app_icon)

    api_key, env_path = _load_api_key()
    if not api_key:
        QMessageBox.critical(
            None,
            "Не найден API-ключ",
            f"Укажите GEMINI_API_KEY в файле:\n{env_path}",
        )
        sys.exit(1)

    window = MainWindow()
    window.setWindowIcon(app_icon)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
