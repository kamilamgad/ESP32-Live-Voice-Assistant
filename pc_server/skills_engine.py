from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import httpx


def normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower()).strip()


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


@dataclass(frozen=True)
class SkillDefinition:
    skill_id: str
    description: str
    handler: str
    triggers: tuple[str, ...]
    enabled: bool = True


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
        location_query = extract_location_phrase(text)
        if not location_query:
            now = datetime.now()
            return now.strftime("It is %I:%M %p on %A, %B %#d.") if now.tzinfo is None else now.strftime(
                "It is %I:%M %p on %A, %B %#d."
            )
        try:
            place = self._geocode(location_query)
            if not place:
                return "I could not find that place. Please say the city and state."
            timezone_name = str(place.get("timezone", "")).strip()
            if not timezone_name:
                return "I could not determine that location's time zone. Please try another location."
            now = datetime.now(ZoneInfo(timezone_name))
            location = self._format_location_label(place)
            return now.strftime(f"It is %I:%M %p on %A, %B %#d in {location}.")
        except Exception:
            return "I could not fetch live time right now. Please try again."

    def weather_answer(self, text: str) -> str:
        location_query = extract_location_phrase(text)
        if not location_query:
            return "Please say the city and state for weather, for example weather in San Jose California."
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
                f"feels like {feels_f}, wind around {wind_mph} miles per hour."
            )
        except Exception:
            return "I could not fetch live weather right now. Please try again."


class SkillsEngine:
    def __init__(self, skills_dir: Path, timeout_seconds: float = 8.0) -> None:
        self.skills_dir = skills_dir
        self._facts = LiveFacts(timeout_seconds=timeout_seconds)
        self.skills: list[SkillDefinition] = self._load_skills()

    def _load_skills(self) -> list[SkillDefinition]:
        loaded: list[SkillDefinition] = []
        if not self.skills_dir.exists():
            return loaded
        for path in sorted(self.skills_dir.glob("*.json")):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                skill = SkillDefinition(
                    skill_id=str(raw.get("id", path.stem)).strip(),
                    description=str(raw.get("description", "")).strip(),
                    handler=str(raw.get("handler", "")).strip().lower(),
                    triggers=tuple(normalize_text(str(v)) for v in raw.get("triggers", []) if str(v).strip()),
                    enabled=bool(raw.get("enabled", True)),
                )
                if skill.enabled and skill.handler and skill.triggers:
                    loaded.append(skill)
            except Exception:
                continue
        return loaded

    def list_skill_ids(self) -> list[str]:
        return [skill.skill_id for skill in self.skills]

    def match(self, text: str) -> Optional[SkillDefinition]:
        normalized = normalize_text(text)
        if not normalized:
            return None
        for skill in self.skills:
            if any(trigger and trigger in normalized for trigger in skill.triggers):
                return skill
        return None

    def execute(self, skill: SkillDefinition, text: str) -> str:
        if skill.handler == "time":
            return self._facts.time_answer(text)
        if skill.handler == "weather":
            return self._facts.weather_answer(text)
        if skill.handler == "research":
            return self._research(text)
        return f"Skill '{skill.skill_id}' is configured with an unknown handler."

    def _research(self, text: str) -> str:
        query = text.strip()
        if not query:
            return "Please say what topic you want me to research."
        try:
            response = httpx.get(
                "https://api.duckduckgo.com/",
                params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
                timeout=8.0,
            )
            response.raise_for_status()
            payload = response.json()
            abstract = str(payload.get("AbstractText", "")).strip()
            heading = str(payload.get("Heading", "")).strip()
            if abstract:
                title = f"{heading}: " if heading else ""
                return f"{title}{abstract}"

            related = payload.get("RelatedTopics", []) or []
            snippets: list[str] = []
            for item in related:
                if isinstance(item, dict) and item.get("Text"):
                    snippets.append(str(item["Text"]).strip())
                if len(snippets) >= 2:
                    break
            if snippets:
                return " ".join(snippets)
            return "I could not find a strong research result for that query."
        except Exception:
            return "I could not complete that research request right now."
