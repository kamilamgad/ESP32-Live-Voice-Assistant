#pragma once

// Copy this file to config.h and fill in your local values.

#define WIFI_SSID "YOUR_WIFI_NAME"
#define WIFI_PASSWORD "YOUR_WIFI_PASSWORD"

// Example: "http://192.168.1.50:8000"
#define SERVER_BASE_URL "http://YOUR_PC_IP:8000"

#define I2S_BCLK_PIN 14
#define I2S_WS_PIN 15
#define I2S_MIC_DATA_PIN 32
#define I2S_SPK_DATA_PIN 22

#define AUDIO_SAMPLE_RATE 16000
#define AUDIO_BUFFER_SAMPLES 256
#define AUDIO_UPLOAD_CHUNK_SAMPLES 4096
