from __future__ import annotations

import os
import random
import re
import subprocess
import tempfile
import threading
import time
import uuid
import wave
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import httpx
import numpy as np
from dotenv import load_dotenv
from fastapi import Body, FastAPI, HTTPException, Response
from pydantic import BaseModel, Field
from sklearn_wakeword import DEFAULT_THRESHOLD as DEFAULT_CUSTOM_WAKE_THRESHOLD, SklearnWakeWordModel
from skills_engine import SkillsEngine


app = FastAPI(title="ESP32 Live Assistant Server")
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
load_dotenv(ROOT_DIR / ".env")


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw else default


def env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw else default


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower()).strip()


def looks_like_junk_transcript(text: str) -> bool:
    normalized = normalize_text(text)
    if not normalized:
        return True
    junk_markers = [
        "transcription by castingwords",
        "castingwords",
        "ready say hey dobby",
        "ready and listening for hey dobby",
        "what can i help with",
    ]
    if any(marker in normalized for marker in junk_markers):
        return True

    tokens = normalized.split()
    if not tokens:
        return True

    # Single-token filler/noise utterances are usually non-user speech.
    filler = {"uh", "um", "hmm", "mm", "ah", "er", "huh"}
    if len(tokens) == 1 and tokens[0] in filler:
        return True

    alpha_chars = sum(1 for ch in normalized if ch.isalpha())
    if alpha_chars < 3:
        return True
    return False


def parse_wake_phrases(value: str) -> list[str]:
    phrases = [normalize_text(part) for part in value.split("|")]
    return [phrase for phrase in phrases if phrase]


def openwakeword_model_key(value: str) -> str:
    candidate = Path(value)
    if candidate.exists():
        return candidate.stem
    return value.strip()


def looks_like_task_request(text: str) -> bool:
    normalized = normalize_text(text)
    task_patterns = [
        "open ",
        "close ",
        "launch ",
        "run ",
        "create ",
        "edit ",
        "write ",
        "delete ",
        "rename ",
        "move ",
        "copy ",
        "install ",
        "uninstall ",
        "search my",
        "find file",
        "show me the file",
        "in the folder",
        "on my computer",
        "use codex",
        "use the terminal",
        "run a command",
        "open browser",
        "open chrome",
        "click ",
        "go to ",
    ]
    return any(pattern in normalized for pattern in task_patterns)


def looks_like_time_request(text: str) -> bool:
    normalized = normalize_text(text)
    return any(
        phrase in normalized
        for phrase in [
            "what time is it",
            "whats the time",
            "what is the time",
            "current time",
            "time right now",
            "what day is it",
            "what is today's date",
            "what is todays date",
            "what's today's date",
            "whats todays date",
            "what date is it",
            "tell me the time",
            "tell me the date",
            "what day is today",
        ]
    )


def looks_like_nonlocal_time_request(text: str) -> bool:
    normalized = normalize_text(text)
    location_markers = [
        " in ",
        " at ",
        " for ",
        " tokyo",
        " london",
        " chicago",
        " new york",
        " los angeles",
        " paris",
        " miami",
        " seattle",
        " denver",
        " california",
        " texas",
        " florida",
        " europe",
        " asia",
    ]
    return looks_like_time_request(text) and any(marker in normalized for marker in location_markers)


def looks_like_weather_request(text: str) -> bool:
    normalized = normalize_text(text)
    return any(
        phrase in normalized
        for phrase in [
            "weather",
            "temperature",
            "forecast",
            "is it raining",
            "is it snowing",
            "how hot",
            "how cold",
        ]
    )


def looks_like_end_conversation(text: str) -> bool:
    normalized = normalize_text(text)
    return any(
        phrase in normalized
        for phrase in [
            "end conversation",
            "stop listening",
            "goodbye dobby",
            "bye dobby",
            "thanks dobby",
            "thank you dobby",
            "never mind",
            "nevermind",
            "thats all",
            "that's all",
            "we are done",
        ]
    )


def get_local_time_answer() -> str:
    now = datetime.now()
    return now.strftime("It is %I:%M %p on %A, %B %-d.") if os.name != "nt" else now.strftime("It is %I:%M %p on %A, %B %#d.")


def _format_time_answer(now: datetime, where: str) -> str:
    if os.name != "nt":
        return now.strftime(f"It is %I:%M %p on %A, %B %-d in {where}.")
    return now.strftime(f"It is %I:%M %p on %A, %B %#d in {where}.")


TIMEZONE_HINTS: list[tuple[str, str, str]] = [
    ("san jose california", "San Jose, California", "America/Los_Angeles"),
    ("san jose", "San Jose, California", "America/Los_Angeles"),
    ("california", "California", "America/Los_Angeles"),
    ("los angeles", "Los Angeles", "America/Los_Angeles"),
    ("seattle", "Seattle", "America/Los_Angeles"),
    ("denver", "Denver", "America/Denver"),
    ("chicago", "Chicago", "America/Chicago"),
    ("miami", "Miami", "America/New_York"),
    ("new york", "New York", "America/New_York"),
    ("paris", "Paris", "Europe/Paris"),
    ("london", "London", "Europe/London"),
    ("tokyo", "Tokyo", "Asia/Tokyo"),
]


def get_time_answer(text: str) -> str:
    normalized = normalize_text(text)
    if not looks_like_nonlocal_time_request(text):
        return get_local_time_answer()

    for hint, label, tz_name in sorted(TIMEZONE_HINTS, key=lambda row: len(row[0]), reverse=True):
        if hint in normalized:
            now = datetime.now(ZoneInfo(tz_name))
            return _format_time_answer(now, label)

    return "I could not determine that location's time zone. Please say the city and state."


def extract_location_phrase(text: str) -> Optional[str]:
    lowered = text.strip().lower()
    for marker in [" in ", " at ", " for "]:
        idx = lowered.find(marker)
        if idx >= 0:
            location = text[idx + len(marker) :].strip(" .?!,")
            for tail in ["right now", "now", "today", "currently", "please"]:
                if location.lower().endswith(" " + tail):
                    location = location[: -len(tail) - 1].strip(" .?!,")
            if location:
                return location
    return None


def weather_code_to_text(code: int) -> str:
    weather_map = {
        0: "clear sky",
        1: "mainly clear",
        2: "partly cloudy",
        3: "overcast",
        45: "foggy",
        48: "depositing rime fog",
        51: "light drizzle",
        53: "moderate drizzle",
        55: "dense drizzle",
        56: "light freezing drizzle",
        57: "dense freezing drizzle",
        61: "slight rain",
        63: "moderate rain",
        65: "heavy rain",
        66: "light freezing rain",
        67: "heavy freezing rain",
        71: "slight snow",
        73: "moderate snow",
        75: "heavy snow",
        77: "snow grains",
        80: "slight rain showers",
        81: "moderate rain showers",
        82: "violent rain showers",
        85: "slight snow showers",
        86: "heavy snow showers",
        95: "thunderstorm",
        96: "thunderstorm with slight hail",
        99: "thunderstorm with heavy hail",
    }
    return weather_map.get(code, "unknown conditions")


class LiveFacts:
    def __init__(self, timeout_seconds: float = 8.0) -> None:
        self._client = httpx.Client(timeout=timeout_seconds)

    def _geocode(self, location_query: str) -> Optional[dict[str, object]]:
        response = self._client.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": location_query, "count": 1, "language": "en", "format": "json"},
        )
        response.raise_for_status()
        results = response.json().get("results", [])
        if not results:
            return None
        return results[0]

    def _format_location_label(self, place: dict[str, object]) -> str:
        name = str(place.get("name", "")).strip()
        admin1 = str(place.get("admin1", "")).strip()
        country = str(place.get("country", "")).strip()
        parts = [part for part in [name, admin1, country] if part]
        return ", ".join(parts) if parts else "that location"

    def time_answer(self, text: str) -> str:
        if not looks_like_nonlocal_time_request(text):
            return get_local_time_answer()

        location_query = extract_location_phrase(text)
        if not location_query:
            return "Please say the city and state for the time you want."

        try:
            place = self._geocode(location_query)
            if not place:
                return "I could not find that place. Please say the city and state."
            timezone_name = str(place.get("timezone", "")).strip()
            if not timezone_name:
                return "I could not determine that location's time zone. Please try another location."
            now = datetime.now(ZoneInfo(timezone_name))
            return _format_time_answer(now, self._format_location_label(place))
        except Exception:
            return "I could not fetch live time right now. Please try again."

    def weather_answer(self, text: str) -> str:
        location_query = extract_location_phrase(text)
        if not location_query:
            return "Please say the city and state for the weather, like weather in San Jose California."

        try:
            place = self._geocode(location_query)
            if not place:
                return "I could not find that place. Please say the city and state."

            latitude = place.get("latitude")
            longitude = place.get("longitude")
            if latitude is None or longitude is None:
                return "I could not resolve that location for weather."

            response = self._client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": latitude,
                    "longitude": longitude,
                    "current": "temperature_2m,apparent_temperature,weather_code,wind_speed_10m",
                    "temperature_unit": "fahrenheit",
                    "wind_speed_unit": "mph",
                    "timezone": "auto",
                },
            )
            response.raise_for_status()
            payload = response.json()
            current = payload.get("current", {})
            temp_f = current.get("temperature_2m")
            feels_f = current.get("apparent_temperature")
            wind_mph = current.get("wind_speed_10m")
            weather_code = int(current.get("weather_code", -1))
            condition = weather_code_to_text(weather_code)
            label = self._format_location_label(place)
            return (
                f"Current weather in {label}: {condition}, {temp_f} degrees Fahrenheit, "
                f"feels like {feels_f}, with wind around {wind_mph} miles per hour."
            )
        except Exception:
            return "I could not fetch live weather right now. Please try again."


def pcm_energy(pcm_bytes: bytes) -> float:
    if not pcm_bytes:
        return 0.0
    samples = np.frombuffer(pcm_bytes, dtype=np.int16)
    if samples.size == 0:
        return 0.0
    return float(np.mean(np.abs(samples)))


def write_wav_file(path: Path, pcm_bytes: bytes, sample_rate: int, channels: int) -> None:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_bytes)


def resample_pcm16_mono(raw_bytes: bytes, src_rate: int, dst_rate: int) -> bytes:
    if src_rate == dst_rate or not raw_bytes:
        return raw_bytes

    samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32)
    if samples.size == 0:
        return raw_bytes

    duration = samples.size / src_rate
    dst_size = max(int(duration * dst_rate), 1)
    src_x = np.linspace(0.0, 1.0, num=samples.size, endpoint=False)
    dst_x = np.linspace(0.0, 1.0, num=dst_size, endpoint=False)
    resampled = np.interp(dst_x, src_x, samples)
    clipped = np.clip(resampled, -32768, 32767).astype(np.int16)
    return clipped.tobytes()


def normalize_pcm16(raw_bytes: bytes, target_peak: int = 30000) -> bytes:
    if not raw_bytes:
        return raw_bytes

    samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.int32)
    if samples.size == 0:
        return raw_bytes

    peak = int(np.max(np.abs(samples)))
    if peak <= 0:
        return raw_bytes

    gain = min(target_peak / peak, 8.0)
    boosted = np.clip(samples * gain, -32768, 32767).astype(np.int16)
    return boosted.tobytes()


def preprocess_command_pcm16(
    raw_bytes: bytes,
    sample_rate: int,
    noise_gate: float,
    trim_padding_seconds: float,
) -> bytes:
    if not raw_bytes:
        return raw_bytes

    samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.int32)
    if samples.size == 0:
        return raw_bytes

    gate = max(int(noise_gate), 0)
    if gate > 0:
        mask = np.abs(samples) >= gate
        samples = np.where(mask, samples, 0)

    nonzero = np.flatnonzero(samples)
    if nonzero.size == 0:
        return b""

    pad = max(int(sample_rate * max(trim_padding_seconds, 0.0)), 1)
    start = max(int(nonzero[0]) - pad, 0)
    end = min(int(nonzero[-1]) + pad + 1, samples.size)
    trimmed = samples[start:end]
    if trimmed.size == 0:
        return b""

    return np.clip(trimmed, -32768, 32767).astype(np.int16).tobytes()


@dataclass(frozen=True)
class Settings:
    wake_word: str = os.getenv("ASSISTANT_WAKE_WORD", "hey dobby")
    wake_phrases_raw: str = os.getenv("ASSISTANT_WAKE_PHRASES", "hey dobby|hey bobby")
    workspace_dir: Path = Path(os.getenv("ASSISTANT_WORKSPACE", str(Path.home())))
    codex_cmd: str = os.getenv(
        "CODEX_CMD", str(Path.home() / "AppData" / "Roaming" / "npm" / "codex.cmd")
    )
    codex_model: Optional[str] = os.getenv("CODEX_MODEL")
    codex_timeout_seconds: int = env_int("CODEX_TIMEOUT_SECONDS", 180)
    pcm_sample_rate: int = env_int("PCM_SAMPLE_RATE", 16000)
    pcm_channels: int = env_int("PCM_CHANNELS", 1)
    wake_window_seconds: float = env_float("WAKE_WINDOW_SECONDS", 2.5)
    wake_check_interval_seconds: float = env_float("WAKE_CHECK_INTERVAL_SECONDS", 1.0)
    command_max_seconds: float = env_float("COMMAND_MAX_SECONDS", 12.0)
    command_min_seconds: float = env_float("COMMAND_MIN_SECONDS", 1.0)
    command_silence_seconds: float = env_float("COMMAND_SILENCE_SECONDS", 1.2)
    vad_energy_threshold: float = env_float("VAD_ENERGY_THRESHOLD", 900.0)
    vad_adaptive_enabled: bool = env_bool("VAD_ADAPTIVE_ENABLED", True)
    vad_noise_floor_alpha: float = env_float("VAD_NOISE_FLOOR_ALPHA", 0.08)
    vad_noise_multiplier: float = env_float("VAD_NOISE_MULTIPLIER", 2.2)
    vad_min_floor: float = env_float("VAD_MIN_FLOOR", 120.0)
    vad_max_floor: float = env_float("VAD_MAX_FLOOR", 2800.0)
    command_noise_gate: float = env_float("COMMAND_NOISE_GATE", 500.0)
    command_trim_padding_seconds: float = env_float("COMMAND_TRIM_PADDING_SECONDS", 0.12)
    wake_transcribe_model: str = os.getenv("WAKE_TRANSCRIBE_MODEL", "tiny.en")
    wake_transcribe_fallback_enabled: bool = env_bool("WAKE_TRANSCRIBE_FALLBACK_ENABLED", False)
    command_transcribe_model: str = os.getenv("COMMAND_TRANSCRIBE_MODEL", "base.en")
    command_stt_backend: str = os.getenv("COMMAND_STT_BACKEND", "groq").strip().lower()
    command_stt_prompt: str = os.getenv(
        "COMMAND_STT_PROMPT",
        "Transcribe English speech for a voice assistant named Dobby. Return only spoken words.",
    )
    whisper_compute_type: str = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
    piper_cmd: Optional[str] = os.getenv("PIPER_CMD")
    piper_model: Optional[str] = os.getenv("PIPER_MODEL")
    piper_config: Optional[str] = os.getenv("PIPER_CONFIG")
    piper_sample_rate: int = env_int("PIPER_SAMPLE_RATE", 22050)
    custom_sklearn_wake_model: Optional[str] = os.getenv("CUSTOM_WAKEWORD_MODEL_PATH")
    custom_sklearn_wake_threshold: float = env_float("CUSTOM_WAKEWORD_THRESHOLD", DEFAULT_CUSTOM_WAKE_THRESHOLD)
    custom_wake_model: Optional[str] = os.getenv("OPENWAKEWORD_MODEL_PATH")
    custom_wake_verifier: Optional[str] = os.getenv("OPENWAKEWORD_VERIFIER_PATH")
    custom_wake_threshold: float = env_float("OPENWAKEWORD_THRESHOLD", 0.6)
    custom_wake_verifier_trigger_threshold: float = env_float("OPENWAKEWORD_VERIFIER_TRIGGER_THRESHOLD", 0.1)
    groq_api_key: Optional[str] = os.getenv("GROQ_API_KEY")
    groq_base_url: str = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    groq_stt_model: str = os.getenv("GROQ_STT_MODEL", "whisper-large-v3-turbo")
    groq_chat_model: str = os.getenv("GROQ_CHAT_MODEL", "groq/compound-mini")
    groq_timeout_seconds: int = env_int("GROQ_TIMEOUT_SECONDS", 60)
    groq_max_retries: int = env_int("GROQ_MAX_RETRIES", 4)
    groq_retry_base_seconds: float = env_float("GROQ_RETRY_BASE_SECONDS", 0.75)
    groq_stt_min_interval_seconds: float = env_float("GROQ_STT_MIN_INTERVAL_SECONDS", 1.0)
    groq_chat_min_interval_seconds: float = env_float("GROQ_CHAT_MIN_INTERVAL_SECONDS", 0.5)
    assistant_persona: str = os.getenv(
        "ASSISTANT_PERSONA",
        "You are Dobby, a fast voice assistant. Be concise, accurate, and natural to hear aloud. "
        "Use live web knowledge when useful. Do not mention hidden tools unless asked.",
    )
    wake_prompt_text: str = os.getenv("WAKE_PROMPT_TEXT", "What can I help with?")
    post_wake_listen_delay_seconds: float = env_float("POST_WAKE_LISTEN_DELAY_SECONDS", 2.0)
    followup_timeout_seconds: float = env_float("FOLLOWUP_TIMEOUT_SECONDS", 600.0)
    conversation_turn_limit: int = env_int("CONVERSATION_TURN_LIMIT", 8)
    followup_enabled: bool = env_bool("FOLLOWUP_ENABLED", False)
    primary_agent_mode: str = os.getenv("PRIMARY_AGENT_MODE", "codex_first").strip().lower()
    primary_agent_fallback_to_groq: bool = env_bool("PRIMARY_AGENT_FALLBACK_TO_GROQ", True)
    skills_enabled: bool = env_bool("SKILLS_ENABLED", True)
    skills_timeout_seconds: float = env_float("SKILLS_TIMEOUT_SECONDS", 8.0)
    prewarm_models: bool = env_bool("PREWARM_MODELS", True)


SETTINGS = Settings()


class ReplyStatus(BaseModel):
    ready: bool
    transcript: Optional[str] = None
    response_text: Optional[str] = None
    state: str


class TextCommandRequest(BaseModel):
    text: str = Field(min_length=1)
    route_to_codex: bool = True


class AudioFrameResult(BaseModel):
    bytes_received: int
    buffered_frames: int
    state: str


class CodexResult(BaseModel):
    success: bool
    text: str
    return_code: int


class FasterWhisperTranscriber:
    def __init__(self, model_name: str, compute_type: str) -> None:
        self.model_name = model_name
        self.compute_type = compute_type
        self._model = None
        self._lock = threading.Lock()

    def _ensure_model(self):
        with self._lock:
            if self._model is None:
                from faster_whisper import WhisperModel

                self._model = WhisperModel(self.model_name, compute_type=self.compute_type)
            return self._model

    def warmup(self) -> None:
        self._ensure_model()

    def transcribe_pcm(self, pcm_bytes: bytes, sample_rate: int, channels: int) -> str:
        if not pcm_bytes:
            return ""

        model = self._ensure_model()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = Path(temp_file.name)

        try:
            write_wav_file(temp_path, pcm_bytes, sample_rate, channels)
            segments, _ = model.transcribe(
                str(temp_path),
                language="en",
                task="transcribe",
                beam_size=1,
                vad_filter=True,
                condition_on_previous_text=False,
            )
            return " ".join(segment.text.strip() for segment in segments).strip()
        finally:
            temp_path.unlink(missing_ok=True)


class PiperSynthesizer:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self.enabled = bool(settings.piper_cmd and settings.piper_model)
        self._source_sample_rate = settings.piper_sample_rate
        self._config_arg = settings.piper_config
        if settings.piper_model:
            json_path = (
                Path(settings.piper_config)
                if settings.piper_config
                else Path(settings.piper_model).with_suffix(".onnx.json")
            )
            if json_path.exists():
                self._config_arg = str(json_path)
                try:
                    import json

                    config = json.loads(json_path.read_text(encoding="utf-8"))
                    self._source_sample_rate = int(
                        config.get("audio", {}).get("sample_rate", self._source_sample_rate)
                    )
                except Exception:
                    pass

    def synthesize(self, text: str) -> bytes:
        if not self.enabled or not text.strip():
            return b""

        command = [self._settings.piper_cmd, "--model", self._settings.piper_model, "--output_raw"]
        if self._config_arg:
            command.extend(["--config", self._config_arg])

        completed = subprocess.run(
            command,
            input=text.encode("utf-8"),
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.decode("utf-8", errors="replace"))

        pcm = resample_pcm16_mono(
            completed.stdout,
            src_rate=self._source_sample_rate,
            dst_rate=self._settings.pcm_sample_rate,
        )
        return normalize_pcm16(pcm)


class OpenWakeWordDetector:
    def __init__(self, settings: Settings) -> None:
        self.enabled = False
        self.threshold = settings.custom_wake_threshold
        self.model_name = "custom"
        self.verifier_enabled = False
        self._model = None

        if not settings.custom_wake_model:
            return

        try:
            from openwakeword.model import Model

            model_name = openwakeword_model_key(settings.custom_wake_model)
            verifier_models = {}
            if settings.custom_wake_verifier:
                verifier_models[model_name] = settings.custom_wake_verifier

            self._model = Model(
                wakeword_models=[settings.custom_wake_model],
                custom_verifier_models=verifier_models,
                custom_verifier_threshold=settings.custom_wake_verifier_trigger_threshold,
            )
            self.enabled = True
            self.model_name = model_name
            self.verifier_enabled = bool(verifier_models)
        except Exception:
            self.enabled = False

    def triggered(self, pcm_bytes: bytes) -> bool:
        if not self.enabled or not pcm_bytes:
            return False

        frame = np.frombuffer(pcm_bytes, dtype=np.int16)
        if frame.size == 0:
            return False

        prediction = self._model.predict(frame)
        for score in prediction.values():
            if score >= self.threshold:
                return True
        return False


class SklearnWakeWordDetector:
    def __init__(self, settings: Settings) -> None:
        self.enabled = False
        self.threshold = settings.custom_sklearn_wake_threshold
        self._model: Optional[SklearnWakeWordModel] = None

        if not settings.custom_sklearn_wake_model:
            return

        try:
            self._model = SklearnWakeWordModel(settings.custom_sklearn_wake_model)
            self.enabled = True
        except Exception:
            self.enabled = False

    def triggered(self, pcm_bytes: bytes) -> bool:
        if not self.enabled or not self._model or not pcm_bytes:
            return False
        return self._model.score(pcm_bytes) >= self.threshold


class CodexExecutor:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def run(self, prompt: str) -> CodexResult:
        output_path = DATA_DIR / f"codex-last-message-{uuid.uuid4().hex}.txt"
        command = ["cmd", "/c", self._settings.codex_cmd]
        if self._settings.codex_model:
            command.extend(["--model", self._settings.codex_model])
        command.extend(
            [
                "exec",
                "--skip-git-repo-check",
                "--sandbox",
                "workspace-write",
                "--output-last-message",
                str(output_path),
                "-C",
                str(self._settings.workspace_dir),
                prompt,
            ]
        )

        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=self._settings.codex_timeout_seconds,
            check=False,
        )

        last_message = ""
        if output_path.exists():
            last_message = output_path.read_text(encoding="utf-8", errors="replace").strip()
            output_path.unlink(missing_ok=True)

        if not last_message:
            last_message = completed.stdout.strip() or completed.stderr.strip()

        return CodexResult(
            success=completed.returncode == 0,
            text=last_message,
            return_code=completed.returncode,
        )


class GroqClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self.enabled = bool(settings.groq_api_key)
        self._client = httpx.Client(
            base_url=settings.groq_base_url,
            timeout=settings.groq_timeout_seconds,
            headers={"Authorization": f"Bearer {settings.groq_api_key}"} if settings.groq_api_key else {},
        )
        self._rate_lock = threading.Lock()
        self._last_stt_call_at = 0.0
        self._last_chat_call_at = 0.0

    def _respect_min_interval(self, call_type: str) -> None:
        now = time.monotonic()
        with self._rate_lock:
            if call_type == "stt":
                minimum_gap = self._settings.groq_stt_min_interval_seconds
                elapsed = now - self._last_stt_call_at
                if elapsed < minimum_gap:
                    time.sleep(minimum_gap - elapsed)
                self._last_stt_call_at = time.monotonic()
            else:
                minimum_gap = self._settings.groq_chat_min_interval_seconds
                elapsed = now - self._last_chat_call_at
                if elapsed < minimum_gap:
                    time.sleep(minimum_gap - elapsed)
                self._last_chat_call_at = time.monotonic()

    def _post_with_retry(self, path: str, *, call_type: str, data=None, files=None, json=None) -> httpx.Response:
        attempts = max(0, self._settings.groq_max_retries)
        for attempt in range(attempts + 1):
            self._respect_min_interval(call_type)
            response = self._client.post(path, data=data, files=files, json=json)
            if response.status_code != 429:
                response.raise_for_status()
                return response

            if attempt >= attempts:
                response.raise_for_status()

            retry_after = response.headers.get("retry-after", "").strip()
            if retry_after.isdigit():
                delay_seconds = float(retry_after)
            else:
                delay_seconds = min(10.0, self._settings.groq_retry_base_seconds * (2 ** attempt))
            delay_seconds += random.uniform(0.0, 0.25)
            print(f"[groq] rate-limited on {call_type}, retrying in {delay_seconds:.2f}s (attempt {attempt + 1}/{attempts})")
            time.sleep(delay_seconds)

        raise RuntimeError("Groq request retry loop ended unexpectedly")

    def transcribe_pcm(self, pcm_bytes: bytes, sample_rate: int, channels: int, prompt: str = "") -> str:
        if not self.enabled or not pcm_bytes:
            return ""

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = Path(temp_file.name)

        try:
            write_wav_file(temp_path, pcm_bytes, sample_rate, channels)
            with temp_path.open("rb") as audio_file:
                files = {
                    "file": (temp_path.name, audio_file, "audio/wav"),
                }
                data = {
                    "model": self._settings.groq_stt_model,
                    "language": "en",
                    "response_format": "verbose_json",
                }
                if prompt:
                    data["prompt"] = prompt
                response = self._post_with_retry(
                    "/audio/transcriptions",
                    call_type="stt",
                    data=data,
                    files=files,
                )
                payload = response.json()
                return str(payload.get("text", "")).strip()
        finally:
            temp_path.unlink(missing_ok=True)

    def chat(self, user_text: str, history: list[dict[str, str]] | None = None) -> str:
        if not self.enabled:
            raise RuntimeError("Groq API key is not configured")

        messages = [{"role": "system", "content": self._settings.assistant_persona}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_text})

        payload = {
            "model": self._settings.groq_chat_model,
            "messages": messages,
            "temperature": 0.3,
            "compound_custom": {
                "tools": {
                    "enabled_tools": ["web_search"]
                }
            },
        }
        response = self._post_with_retry(
            "/chat/completions",
            call_type="chat",
            json=payload,
        )
        data = response.json()
        return (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )


class AssistantPipeline:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.state = "idle"
        self.buffered_frames: deque[bytes] = deque()
        self.buffered_bytes = 0
        self.max_buffer_bytes = int(
            settings.pcm_sample_rate * settings.pcm_channels * 2 * max(settings.command_max_seconds, settings.wake_window_seconds + 1.0)
        )
        self.last_transcript: Optional[str] = None
        self.last_response_text: Optional[str] = None
        self.reply_pcm: bytes = b""
        self.lock = threading.Lock()
        self.codex = CodexExecutor(settings)
        self.groq = GroqClient(settings)
        self.live_facts = LiveFacts()
        self.skills = SkillsEngine(ROOT_DIR / "skills", timeout_seconds=settings.skills_timeout_seconds) if settings.skills_enabled else None
        self.wake_transcriber = FasterWhisperTranscriber(
            settings.wake_transcribe_model, settings.whisper_compute_type
        )
        self.command_transcriber = FasterWhisperTranscriber(
            settings.command_transcribe_model, settings.whisper_compute_type
        )
        self.synthesizer = PiperSynthesizer(settings)
        self._wake_prompt_pcm: bytes = b""
        self.sklearn_wake = SklearnWakeWordDetector(settings)
        self.openwake = OpenWakeWordDetector(settings)
        self.wake_phrases = parse_wake_phrases(settings.wake_phrases_raw)
        self.normalized_wake_word = self.wake_phrases[0] if self.wake_phrases else normalize_text(settings.wake_word)
        self.conversation_history: deque[dict[str, str]] = deque(maxlen=settings.conversation_turn_limit * 2)
        self.command_audio = bytearray()
        self.capture_requires_wake = True
        self.command_started_at = 0.0
        self.last_voice_at = 0.0
        self.last_wake_check_at = 0.0
        self.last_wake_detection_text = ""
        self.waiting_for_followup_command = False
        self.followup_listen_after = 0.0
        self.conversation_active = False
        self.followup_until = 0.0
        self.noise_floor_energy = max(settings.vad_min_floor, settings.vad_energy_threshold * 0.5)
        self.current_vad_threshold = settings.vad_energy_threshold
        self._stop_event = threading.Event()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()
        if settings.prewarm_models:
            threading.Thread(target=self._prewarm_models, daemon=True).start()
        if settings.wake_prompt_text.strip():
            threading.Thread(target=self._prewarm_wake_prompt, daemon=True).start()

    def _prewarm_models(self) -> None:
        try:
            print("[warmup] loading wake transcriber model...")
            self.wake_transcriber.warmup()
            print("[warmup] loading command transcriber model...")
            self.command_transcriber.warmup()
            print("[warmup] whisper models ready")
        except Exception as exc:
            print(f"[warmup] model prewarm failed: {exc}")

    def _prewarm_wake_prompt(self) -> None:
        try:
            prompt = self.settings.wake_prompt_text.strip()
            if not prompt:
                return
            print("[warmup] precomputing wake prompt audio...")
            self._wake_prompt_pcm = self._safely_synthesize(prompt)
            print("[warmup] wake prompt audio ready")
        except Exception as exc:
            print(f"[warmup] wake prompt precompute failed: {exc}")

    def add_audio_frame(self, frame: bytes) -> AudioFrameResult:
        now = time.monotonic()
        energy = pcm_energy(frame)
        with self.lock:
            # Only learn ambient floor during true idle; ignore playback/follow-up phases.
            can_learn_floor = (
                self.state == "idle"
                and not self.conversation_active
                and not self.waiting_for_followup_command
            )
            if can_learn_floor:
                self._update_noise_floor(energy)
            self.buffered_frames.append(frame)
            self.buffered_bytes += len(frame)
            while self.buffered_bytes > self.max_buffer_bytes and self.buffered_frames:
                removed = self.buffered_frames.popleft()
                self.buffered_bytes -= len(removed)

            if self.state == "capturing_command":
                self.command_audio.extend(frame)
                if energy >= self._effective_vad_threshold():
                    self.last_voice_at = now

            return AudioFrameResult(
                bytes_received=len(frame),
                buffered_frames=len(self.buffered_frames),
                state=self.state,
            )

    def _effective_vad_threshold(self) -> float:
        return max(self.settings.vad_energy_threshold, self.current_vad_threshold)

    def _update_noise_floor(self, energy: float) -> None:
        if not self.settings.vad_adaptive_enabled:
            self.current_vad_threshold = self.settings.vad_energy_threshold
            return
        clamped = min(max(energy, self.settings.vad_min_floor), self.settings.vad_max_floor)
        alpha = min(max(self.settings.vad_noise_floor_alpha, 0.01), 0.5)
        self.noise_floor_energy = (1.0 - alpha) * self.noise_floor_energy + alpha * clamped
        adaptive = self.noise_floor_energy * max(self.settings.vad_noise_multiplier, 1.0)
        hard_ceiling = min(self.settings.vad_max_floor * 1.15, self.settings.vad_energy_threshold * 2.2)
        adaptive = min(max(adaptive, self.settings.vad_energy_threshold), hard_ceiling)
        self.current_vad_threshold = adaptive

    def handle_text_command(self, text: str, route_to_codex: bool) -> CodexResult:
        with self.lock:
            self.state = "processing"
            self.last_transcript = text

        result, response_text = self._generate_response(text, route_to_codex)
        reply_pcm = self._safely_synthesize(response_text)

        with self.lock:
            self.last_response_text = response_text
            self.reply_pcm = reply_pcm
            self.state = "reply_ready" if reply_pcm else "idle"

        return result

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                if self.state == "idle":
                    self._maybe_detect_wake_or_followup()
                elif self.state == "capturing_command":
                    self._maybe_finish_command()
            except Exception as exc:
                with self.lock:
                    self.state = "error"
                    self.last_response_text = f"Assistant pipeline error: {exc}"
            time.sleep(0.1)

    def _recent_audio(self, seconds: float) -> bytes:
        target_bytes = int(self.settings.pcm_sample_rate * self.settings.pcm_channels * 2 * seconds)
        parts: list[bytes] = []
        total = 0
        for frame in reversed(self.buffered_frames):
            parts.append(frame)
            total += len(frame)
            if total >= target_bytes:
                break
        return b"".join(reversed(parts))

    def _clear_buffered_audio_locked(self) -> None:
        self.buffered_frames.clear()
        self.buffered_bytes = 0

    def _maybe_detect_wake_or_followup(self) -> None:
        now = time.monotonic()
        if self.conversation_active and now >= self.followup_until:
            print("[followup] conversation timed out")
            with self.lock:
                self.conversation_active = False
                self.followup_until = 0.0

        if self.conversation_active and now >= self.followup_listen_after:
            with self.lock:
                audio_window = self._recent_audio(min(1.5, self.settings.wake_window_seconds))
                vad_threshold = self._effective_vad_threshold()
            if pcm_energy(audio_window) >= vad_threshold:
                print("[followup] continuing conversation without wake word")
                with self.lock:
                    self.state = "capturing_command"
                    self.command_audio = bytearray()
                    self.capture_requires_wake = False
                    self.waiting_for_followup_command = False
                    self.command_started_at = now
                    self.last_voice_at = now
                return

        # During an active conversation, do not re-run wake phrase detection.
        # Follow-up turns are gated by voice activity instead.
        if self.conversation_active:
            return

        now = time.monotonic()
        if now - self.last_wake_check_at < self.settings.wake_check_interval_seconds:
            return
        self.last_wake_check_at = now

        with self.lock:
            audio_window = self._recent_audio(self.settings.wake_window_seconds)
            vad_threshold = self._effective_vad_threshold()

        if pcm_energy(audio_window) < vad_threshold:
            return

        custom_wake_triggered = self.sklearn_wake.triggered(audio_window) or self.openwake.triggered(audio_window)
        if custom_wake_triggered:
            detection_text = self.settings.wake_word
        else:
            if not self.settings.wake_transcribe_fallback_enabled:
                return
            detection_text = self.wake_transcriber.transcribe_pcm(
                audio_window,
                sample_rate=self.settings.pcm_sample_rate,
                channels=self.settings.pcm_channels,
            )

        normalized = normalize_text(detection_text)
        matched_phrase = next((phrase for phrase in self.wake_phrases if phrase in normalized), None)
        if not matched_phrase:
            return
        if normalized == self.last_wake_detection_text and now - self.command_started_at < 2.0:
            return

        self.last_wake_detection_text = normalized
        print(f"[wake] matched='{matched_phrase}' transcript='{detection_text}'")
        response_text = self.settings.wake_prompt_text
        if response_text.strip():
            reply_pcm = self._wake_prompt_pcm or self._safely_synthesize(response_text)
        else:
            reply_pcm = b""
        with self.lock:
            if not self.conversation_active:
                self.conversation_history.clear()
            self._clear_buffered_audio_locked()
            self.state = "reply_ready" if reply_pcm else "idle"
            self.reply_pcm = reply_pcm
            self.command_audio = bytearray()
            self.capture_requires_wake = False
            self.conversation_active = True
            self.waiting_for_followup_command = True
            self.followup_until = now + self.settings.followup_timeout_seconds
            self.followup_listen_after = now + self.settings.post_wake_listen_delay_seconds
            self.command_started_at = 0.0
            self.last_voice_at = 0.0
            self.last_transcript = detection_text
            self.last_response_text = response_text

    def _maybe_finish_command(self) -> None:
        now = time.monotonic()
        with self.lock:
            duration = now - self.command_started_at
            silence = now - self.last_voice_at
            ready = duration >= self.settings.command_max_seconds or (
                duration >= self.settings.command_min_seconds
                and silence >= self.settings.command_silence_seconds
            )
            if not ready:
                return

            command_audio = bytes(self.command_audio)
            capture_requires_wake = self.capture_requires_wake
            self.command_audio = bytearray()
            self.state = "processing"

        transcript_input = preprocess_command_pcm16(
            command_audio,
            sample_rate=self.settings.pcm_sample_rate,
            noise_gate=self.settings.command_noise_gate,
            trim_padding_seconds=self.settings.command_trim_padding_seconds,
        )
        transcript = self._transcribe_command(transcript_input or command_audio)
        cleaned = self._strip_wake_phrase(transcript) if capture_requires_wake else normalize_text(transcript)
        self.last_transcript = cleaned or transcript
        print(f"[command] wake_required={capture_requires_wake} transcript='{transcript}' cleaned='{cleaned}'")

        if looks_like_junk_transcript(cleaned):
            with self.lock:
                self._clear_buffered_audio_locked()
                self.waiting_for_followup_command = False
                self.conversation_active = False
                self.followup_until = 0.0
                self.state = "idle"
                self.reply_pcm = b""
            return
        if not cleaned:
            if self.conversation_active:
                with self.lock:
                    self._clear_buffered_audio_locked()
                    self.waiting_for_followup_command = True
                    self.followup_listen_after = time.monotonic() + max(
                        0.5, min(1.5, self.settings.post_wake_listen_delay_seconds)
                    )
                    self.state = "idle"
                    self.reply_pcm = b""
                return
            response_text = "I missed that. Please try again."
            result = CodexResult(success=False, text=response_text, return_code=1)
        elif looks_like_end_conversation(cleaned):
            response_text = "Okay. I will stop listening until you say the wake word again."
            result = CodexResult(success=True, text=response_text, return_code=0)
            with self.lock:
                self.conversation_active = False
                self.waiting_for_followup_command = False
                self.followup_until = 0.0
        else:
            result, response_text = self._generate_response(
                cleaned, route_to_codex=looks_like_task_request(cleaned)
            )
            print(f"[assistant] response='{response_text}'")

        reply_pcm = self._safely_synthesize(response_text)
        with self.lock:
            self.last_response_text = response_text
            self.reply_pcm = reply_pcm
            self._clear_buffered_audio_locked()
            if self.conversation_active:
                if self.settings.followup_enabled:
                    self.followup_until = time.monotonic() + self.settings.followup_timeout_seconds
                    self.waiting_for_followup_command = True
                    self.followup_listen_after = time.monotonic() + self.settings.post_wake_listen_delay_seconds
                else:
                    self.conversation_active = False
                    self.waiting_for_followup_command = False
                    self.followup_until = 0.0
            self.state = "reply_ready" if reply_pcm else "idle"

    def _transcribe_command(self, command_audio: bytes) -> str:
        if not command_audio:
            return ""

        # Preferred path: cloud STT for better accuracy/latency, then fallback local.
        if self.settings.command_stt_backend in {"groq", "auto"} and self.groq.enabled:
            try:
                transcript = self.groq.transcribe_pcm(
                    command_audio,
                    sample_rate=self.settings.pcm_sample_rate,
                    channels=self.settings.pcm_channels,
                    prompt=self.settings.command_stt_prompt,
                )
                if transcript:
                    return transcript
            except Exception:
                pass

        return self.command_transcriber.transcribe_pcm(
            command_audio,
            sample_rate=self.settings.pcm_sample_rate,
            channels=self.settings.pcm_channels,
        )

    def _strip_wake_phrase(self, transcript: str) -> str:
        normalized = normalize_text(transcript)
        for phrase in self.wake_phrases:
            if phrase in normalized:
                stripped = normalized.split(phrase, 1)[1].strip(" ,.!?")
                return stripped
        return normalized

    def _generate_response(self, text: str, route_to_codex: bool) -> tuple[CodexResult, str]:
        if looks_like_time_request(text):
            response_text = self.live_facts.time_answer(text)
            result = CodexResult(success=True, text=response_text, return_code=0)
            self._append_turn("user", text)
            self._append_turn("assistant", response_text)
            return result, response_text

        if looks_like_weather_request(text):
            response_text = self.live_facts.weather_answer(text)
            result = CodexResult(success=True, text=response_text, return_code=0)
            self._append_turn("user", text)
            self._append_turn("assistant", response_text)
            return result, response_text

        if self.skills:
            matched_skill = self.skills.match(text)
            if matched_skill:
                response_text = self.skills.execute(matched_skill, text)
                result = CodexResult(success=True, text=response_text, return_code=0)
                self._append_turn("user", text)
                self._append_turn("assistant", response_text)
                return result, response_text

        def run_codex() -> CodexResult:
            return self.codex.run(self._build_codex_prompt(text))

        def run_groq() -> CodexResult:
            if not self.groq.enabled:
                return CodexResult(success=False, text="", return_code=1)
            try:
                answer = self.groq.chat(text, self._conversation_history()[-6:])
                return CodexResult(success=bool(answer.strip()), text=answer.strip(), return_code=0)
            except Exception:
                return CodexResult(success=False, text="", return_code=1)

        mode = self.settings.primary_agent_mode
        if route_to_codex:
            result = run_groq() if mode == "groq_first" else run_codex()
            if (not result.success or not result.text.strip()) and self.settings.primary_agent_fallback_to_groq:
                fallback = run_codex() if mode == "groq_first" else run_groq()
                if fallback.text.strip():
                    result = fallback
        else:
            result = run_groq() if self.groq.enabled else CodexResult(success=True, text=get_local_time_answer(), return_code=0)

        response_text = result.text.strip() if result.text.strip() else "I could not complete that request right now."
        if not result.text.strip():
            result = CodexResult(success=False, text=response_text, return_code=result.return_code)
        self._append_turn("user", text)
        self._append_turn("assistant", response_text)
        return result, response_text

    def _append_turn(self, role: str, content: str) -> None:
        with self.lock:
            self.conversation_history.append({"role": role, "content": content})

    def _conversation_history(self) -> list[dict[str, str]]:
        with self.lock:
            return list(self.conversation_history)

    def _safely_synthesize(self, response_text: str) -> bytes:
        try:
            return self.synthesizer.synthesize(response_text)
        except Exception:
            return b""

    def _build_codex_prompt(self, transcript: str) -> str:
        history = self._conversation_history()[-6:]
        history_block = "\n".join(
            f"{turn.get('role', 'user')}: {turn.get('content', '').strip()}" for turn in history if turn.get("content")
        )
        return (
            "You are a desktop voice assistant triggered by the wake phrase "
            f"'{self.settings.wake_word}'. "
            "Interpret the user's spoken request and carry it out if it is an actionable computer task. "
            "If execution is not appropriate, answer directly. "
            "Keep the final response natural to hear aloud, concise, and useful. "
            "Prefer one short sentence; use at most two unless the user asks for detail. "
            "Do not ask follow-up questions unless absolutely required to complete the request. "
            f"Recent conversation:\n{history_block}\n"
            f"Spoken request: {transcript}"
        )

    def consume_reply(self) -> bytes:
        with self.lock:
            pcm = self.reply_pcm
            self.reply_pcm = b""
            self._clear_buffered_audio_locked()
            self.state = "idle"
            return pcm

    def status(self) -> ReplyStatus:
        with self.lock:
            return ReplyStatus(
                ready=bool(self.reply_pcm),
                transcript=self.last_transcript,
                response_text=self.last_response_text,
                state=self.state,
            )

    def snapshot(self) -> dict[str, object]:
        with self.lock:
            return {
                "state": self.state,
                "wake_word": self.settings.wake_word,
                "wake_phrases": self.wake_phrases,
                "buffered_frames": len(self.buffered_frames),
                "buffered_bytes": self.buffered_bytes,
                "last_transcript": self.last_transcript,
                "last_response_text": self.last_response_text,
                "conversation_active": self.conversation_active,
                "waiting_for_followup_command": self.waiting_for_followup_command,
                "followup_until": self.followup_until,
                "conversation_turns": len(self.conversation_history) // 2,
                "workspace_dir": str(self.settings.workspace_dir),
                "codex_cmd": self.settings.codex_cmd,
                "groq_enabled": self.groq.enabled,
                "groq_chat_model": self.settings.groq_chat_model,
                "groq_stt_model": self.settings.groq_stt_model,
                "wake_transcribe_model": self.settings.wake_transcribe_model,
                "command_transcribe_model": self.settings.command_transcribe_model,
                "tts_enabled": self.synthesizer.enabled,
                "custom_wake_enabled": self.sklearn_wake.enabled,
                "openwake_enabled": self.openwake.enabled,
                "primary_agent_mode": self.settings.primary_agent_mode,
                "primary_agent_fallback_to_groq": self.settings.primary_agent_fallback_to_groq,
                "skills_enabled": bool(self.skills),
                "loaded_skills": self.skills.list_skill_ids() if self.skills else [],
                "vad_threshold_current": self.current_vad_threshold,
                "vad_noise_floor_current": self.noise_floor_energy,
            }


pipeline = AssistantPipeline(SETTINGS)


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "wake_word": SETTINGS.wake_word,
        "workspace_dir": str(SETTINGS.workspace_dir),
        "pcm_sample_rate": SETTINGS.pcm_sample_rate,
        "groq_enabled": pipeline.groq.enabled,
        "tts_enabled": pipeline.synthesizer.enabled,
        "custom_wake_enabled": pipeline.sklearn_wake.enabled,
        "openwake_enabled": pipeline.openwake.enabled,
        "skills_enabled": bool(pipeline.skills),
    }


@app.post("/audio-frame", response_model=AudioFrameResult)
async def audio_frame(raw: bytes = Body(..., media_type="application/octet-stream")) -> AudioFrameResult:
    return pipeline.add_audio_frame(raw)


@app.post("/simulate-text", response_model=CodexResult)
def simulate_text(request: TextCommandRequest) -> CodexResult:
    return pipeline.handle_text_command(request.text, request.route_to_codex)


@app.get("/reply-status", response_model=ReplyStatus)
def reply_status() -> ReplyStatus:
    return pipeline.status()


@app.get("/reply-audio")
def reply_audio() -> Response:
    return Response(content=pipeline.consume_reply(), media_type="application/octet-stream")


@app.get("/state")
def state() -> dict[str, object]:
    return pipeline.snapshot()


@app.post("/debug/save-buffer")
def save_buffer() -> dict[str, str]:
    snapshot_path = DATA_DIR / f"audio-buffer-{uuid.uuid4().hex}.pcm"
    with pipeline.lock:
        if not pipeline.buffered_frames:
            raise HTTPException(status_code=400, detail="No buffered audio frames available")
        snapshot_path.write_bytes(b"".join(pipeline.buffered_frames))
    return {"saved_to": str(snapshot_path)}


@app.post("/debug/process-buffer")
def process_buffer() -> dict[str, str]:
    with pipeline.lock:
        audio_window = pipeline._recent_audio(SETTINGS.wake_window_seconds)

    transcript = pipeline.command_transcriber.transcribe_pcm(
        audio_window,
        sample_rate=SETTINGS.pcm_sample_rate,
        channels=SETTINGS.pcm_channels,
    )
    return {"transcript": transcript}


@app.get("/notes")
def notes() -> dict[str, list[str]]:
    return {
        "implemented_now": [
            "Always-on PCM ingestion from ESP32",
            "Wake detection via local Whisper transcription",
            "Optional custom openWakeWord model support",
            "Command capture until silence",
            "Local Whisper command transcription",
            "Groq conversation path for normal questions",
            "Codex CLI task execution for computer-task style requests",
            "Piper TTS synthesis to reply PCM",
        ],
        "next_steps": [
            "Tune VAD_ENERGY_THRESHOLD and command timing in your room",
            "Switch ESP32 uplink to WebSocket if HTTP frame overhead is too high",
            "Optionally train an openWakeWord custom model for more reliable exact wake-word detection",
        ],
    }
