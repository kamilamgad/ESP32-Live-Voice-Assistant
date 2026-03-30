# ESP32 Live Voice Assistant

`ESP32 Live Voice Assistant` is a hybrid voice assistant project that uses an ESP32 as the always-on audio front end and a PC server for wake detection, transcription, task execution, and spoken replies.

The goal is practical voice control, not just speech-to-text. The device streams microphone audio from ESP32 hardware to a local server, waits for a wake phrase, captures the spoken request, routes the request into Codex or utility skills, and returns assistant speech back to the ESP32 for playback.

## What It Does

- streams live microphone PCM from ESP32 hardware to a PC server
- listens for a wake phrase such as `hey dobby`
- captures the user command after wake detection
- supports command routing to `codex exec` for local tasks
- supports local skills such as time, weather, and lightweight research
- generates assistant voice responses with Piper
- sends reply audio back to the ESP32 for playback

## Architecture

This project is split into two parts:

- `firmware/assistant_client/`
  ESP32 firmware for microphone capture, streaming, polling, and audio playback
- `pc_server/`
  FastAPI backend for wake detection, STT, skills, Codex routing, and TTS

## Stack

- ESP32 firmware with I2S microphone and speaker output
- FastAPI backend
- `faster-whisper` for transcription
- optional `openWakeWord` / custom sklearn wakeword model support
- Groq API support for STT/chat fallback
- Piper for text-to-speech
- Codex CLI for actionable local task execution

## Hardware

The current setup is built around:

- ESP32
- INMP441 microphone
- MAX98357A amplifier

Shared-clock wiring used in this project:

- `GPIO14` -> mic `SCK` and amp `BCLK`
- `GPIO15` -> mic `WS` and amp `LRC`
- `GPIO32` <- mic `SD`
- `GPIO22` -> amp `DIN`

## Repo Layout

- `firmware/assistant_client/assistant_client.ino`
- `firmware/assistant_client/config.example.h`
- `pc_server/server.py`
- `pc_server/skills_engine.py`
- `pc_server/sklearn_wakeword.py`
- `pc_server/train_wakeword.py`
- `pc_server/requirements.txt`
- `pc_server/.env.example`
- `run_server.bat`

## Quick Start

### 1. Set up the PC server

```powershell
cd pc_server
python -m venv ..\.venv
..\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Then edit `.env` and configure your local values for:

- `ASSISTANT_WORKSPACE`
- `CODEX_CMD`
- `PIPER_CMD`
- `PIPER_MODEL`
- `PIPER_CONFIG`
- optionally `GROQ_API_KEY`

Run the backend:

```powershell
cd pc_server
..\.venv\Scripts\python.exe -m uvicorn server:app --host 0.0.0.0 --port 8000
```

Or use:

```powershell
run_server.bat
```

### 2. Set up the ESP32 firmware

Copy:

- `firmware/assistant_client/config.example.h`

to:

- `firmware/assistant_client/config.h`

Then fill in:

- Wi-Fi name
- Wi-Fi password
- PC server URL

Open `firmware/assistant_client/assistant_client.ino` in Arduino IDE and upload it to the ESP32.

## Runtime Flow

1. ESP32 continuously captures mono `16 kHz` PCM
2. ESP32 streams PCM frames to the PC server
3. PC listens for the wake phrase
4. After wake detection, the server captures the user command
5. The command is handled by:
   - Codex for local tasks, or
   - built-in skills / direct assistant handling
6. Piper synthesizes the spoken reply
7. ESP32 fetches and plays only the assistant response audio

## Notes

- This is designed as a local-first assistant architecture
- The user's speech is not intended to be played back to the speaker in the final workflow
- Wakeword reliability can be improved further with a stronger dedicated wakeword model
- WebSocket streaming would be a cleaner future transport than the current HTTP chunk flow

## Privacy

- local config, secrets, trained wakeword artifacts, temp audio, voice models, and Piper binaries are intentionally excluded from the repo
- use `.env.example` and `config.example.h` as templates for local setup

## Current Status

This repo is best understood as a practical voice-assistant prototype focused on real hardware integration, local task execution, and voice-driven workflows rather than a generic chatbot demo.
