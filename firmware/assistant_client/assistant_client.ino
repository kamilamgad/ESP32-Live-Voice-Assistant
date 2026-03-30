#include <Arduino.h>
#include <HTTPClient.h>
#include <WiFi.h>
#include <driver/i2s.h>

#include "config.h"

namespace audio {
constexpr i2s_port_t kI2SPort = I2S_NUM_0;
constexpr int kMicShift = 8;
constexpr int kTxShift = 16;
constexpr size_t kReplyChunkBytes = 1024;
constexpr int kReplyGain = 6;
constexpr int kReplyVolumeNumerator = 3;
constexpr int kReplyVolumeDenominator = 4;
constexpr uint16_t kHttpTimeoutMs = 5000;
constexpr int kMaxConsecutiveUploadFailures = 8;
constexpr unsigned long kReplyPollIntervalMs = 1000;
}  // namespace audio

enum class AssistantState {
  kConnectingWifi,
  kIdleStreaming,
  kFetchingReply,
  kPlayingReply,
  kError,
};

AssistantState g_state = AssistantState::kConnectingWifi;
int32_t g_rxBuffer[AUDIO_BUFFER_SAMPLES];
int32_t g_txBuffer[AUDIO_BUFFER_SAMPLES];
int16_t g_uploadBuffer[AUDIO_UPLOAD_CHUNK_SAMPLES];
size_t g_uploadFill = 0;
unsigned long g_lastLogMs = 0;
unsigned long g_lastReplyPollMs = 0;
unsigned long g_replyPollSuspendUntilMs = 0;
unsigned long g_lastRecoverAttemptMs = 0;
int g_consecutiveUploadFailures = 0;

void logState(const char* message) {
  Serial.printf("[state=%d] %s\n", static_cast<int>(g_state), message);
}

void configureI2S() {
  const i2s_config_t i2sConfig = {
      .mode = static_cast<i2s_mode_t>(I2S_MODE_MASTER | I2S_MODE_RX | I2S_MODE_TX),
      .sample_rate = AUDIO_SAMPLE_RATE,
      .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
      .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
      .communication_format = I2S_COMM_FORMAT_I2S,
      .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
      .dma_buf_count = 8,
      .dma_buf_len = AUDIO_BUFFER_SAMPLES,
      .use_apll = false,
      .tx_desc_auto_clear = true,
      .fixed_mclk = 0,
  };

  const i2s_pin_config_t pinConfig = {
      .bck_io_num = I2S_BCLK_PIN,
      .ws_io_num = I2S_WS_PIN,
      .data_out_num = I2S_SPK_DATA_PIN,
      .data_in_num = I2S_MIC_DATA_PIN,
  };

  ESP_ERROR_CHECK(i2s_driver_install(audio::kI2SPort, &i2sConfig, 0, nullptr));
  ESP_ERROR_CHECK(i2s_set_pin(audio::kI2SPort, &pinConfig));
  ESP_ERROR_CHECK(i2s_zero_dma_buffer(audio::kI2SPort));
}

void connectWifi() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  Serial.printf("Connecting to WiFi SSID '%s'\n", WIFI_SSID);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();
  Serial.print("WiFi connected. ESP32 IP: ");
  Serial.println(WiFi.localIP());
}

void writeSilenceFrame(size_t samples) {
  memset(g_txBuffer, 0, samples * sizeof(int32_t));
  size_t bytesWritten = 0;
  i2s_write(audio::kI2SPort, g_txBuffer, samples * sizeof(int32_t), &bytesWritten, portMAX_DELAY);
}

bool streamAudioFrame(const int16_t* samples, size_t sampleCount) {
  const String url = String(SERVER_BASE_URL) + "/audio-frame";

  int lastStatus = 0;
  String lastError = "";
  for (int attempt = 0; attempt < 2; ++attempt) {
    HTTPClient http;
    if (!http.begin(url)) {
      lastStatus = -1;
      lastError = "HTTP begin failed for /audio-frame";
      continue;
    }

    http.setTimeout(audio::kHttpTimeoutMs);
    http.setReuse(false);
    http.addHeader("Content-Type", "application/octet-stream");
    http.addHeader("Connection", "close");
    const int status = http.POST(reinterpret_cast<uint8_t*>(const_cast<int16_t*>(samples)),
                                 sampleCount * sizeof(int16_t));
    if (status >= 200 && status < 300) {
      http.end();
      return true;
    }

    lastStatus = status;
    lastError = http.errorToString(status);
    http.end();

    const bool retryable =
        status == HTTPC_ERROR_READ_TIMEOUT || status == HTTPC_ERROR_SEND_PAYLOAD_FAILED;
    if (!retryable) {
      break;
    }
    delay(30);
  }

  Serial.printf("/audio-frame POST failed: status=%d error=%s\n",
                lastStatus,
                lastError.c_str());
  return false;
}

bool appendAndMaybeStream(const int16_t* samples, size_t sampleCount) {
  size_t offset = 0;
  while (offset < sampleCount) {
    const size_t remaining = AUDIO_UPLOAD_CHUNK_SAMPLES - g_uploadFill;
    const size_t copyCount = (sampleCount - offset) < remaining ? (sampleCount - offset) : remaining;

    memcpy(&g_uploadBuffer[g_uploadFill], &samples[offset], copyCount * sizeof(int16_t));
    g_uploadFill += copyCount;
    offset += copyCount;

    if (g_uploadFill == AUDIO_UPLOAD_CHUNK_SAMPLES) {
      if (!streamAudioFrame(g_uploadBuffer, g_uploadFill)) {
        return false;
      }
      g_uploadFill = 0;
    }
  }

  return true;
}

bool replyReady() {
  HTTPClient http;
  const String url = String(SERVER_BASE_URL) + "/reply-status";
  if (!http.begin(url)) {
    Serial.println("HTTP begin failed for /reply-status");
    return false;
  }

  http.setTimeout(audio::kHttpTimeoutMs);
  http.setReuse(false);
  http.addHeader("Connection", "close");
  const int status = http.GET();
  if (status != 200) {
    Serial.printf("/reply-status GET failed: status=%d error=%s\n",
                  status,
                  http.errorToString(status).c_str());
    http.end();
    return false;
  }

  const String body = http.getString();
  http.end();
  return body.indexOf("\"ready\":true") >= 0;
}

void playReplyAudio(const uint8_t* bytes, size_t length) {
  const int16_t* pcm16 = reinterpret_cast<const int16_t*>(bytes);
  const size_t sampleCount = length / sizeof(int16_t);

  for (size_t i = 0; i < sampleCount; ++i) {
    int32_t boosted = static_cast<int32_t>(pcm16[i]) * audio::kReplyGain;
    boosted = (boosted * audio::kReplyVolumeNumerator) / audio::kReplyVolumeDenominator;
    if (boosted > 32767) {
      boosted = 32767;
    } else if (boosted < -32768) {
      boosted = -32768;
    }

    int32_t sample32 = boosted << audio::kTxShift;
    g_txBuffer[i % AUDIO_BUFFER_SAMPLES] = sample32;

    if ((i % AUDIO_BUFFER_SAMPLES) == AUDIO_BUFFER_SAMPLES - 1) {
      size_t bytesWritten = 0;
      i2s_write(audio::kI2SPort,
                g_txBuffer,
                AUDIO_BUFFER_SAMPLES * sizeof(int32_t),
                &bytesWritten,
                portMAX_DELAY);
    }
  }

  const size_t remainder = sampleCount % AUDIO_BUFFER_SAMPLES;
  if (remainder > 0) {
    for (size_t i = remainder; i < AUDIO_BUFFER_SAMPLES; ++i) {
      g_txBuffer[i] = 0;
    }
    size_t bytesWritten = 0;
    i2s_write(audio::kI2SPort,
              g_txBuffer,
              AUDIO_BUFFER_SAMPLES * sizeof(int32_t),
              &bytesWritten,
              portMAX_DELAY);
  }
}

bool fetchAndPlayReply() {
  HTTPClient http;
  const String url = String(SERVER_BASE_URL) + "/reply-audio";
  if (!http.begin(url)) {
    Serial.println("HTTP begin failed for /reply-audio");
    return false;
  }

  http.setTimeout(audio::kHttpTimeoutMs);
  http.setReuse(false);
  http.addHeader("Connection", "close");
  const int status = http.GET();
  if (status != 200) {
    Serial.printf("/reply-audio GET failed: status=%d error=%s\n",
                  status,
                  http.errorToString(status).c_str());
    http.end();
    return false;
  }

  WiFiClient* stream = http.getStreamPtr();
  uint8_t chunk[audio::kReplyChunkBytes];

  g_state = AssistantState::kPlayingReply;
  logState("playing reply audio");

  while (http.connected()) {
    const int available = stream->available();
    if (available <= 0) {
      delay(10);
      continue;
    }

    const int toRead =
        available > static_cast<int>(sizeof(chunk)) ? static_cast<int>(sizeof(chunk)) : available;
    const int bytesRead = stream->readBytes(chunk, toRead);
    if (bytesRead <= 0) {
      break;
    }
    playReplyAudio(chunk, static_cast<size_t>(bytesRead));
  }

  http.end();
  g_state = AssistantState::kIdleStreaming;
  // After speaking the wake prompt, prioritize command uplink traffic.
  g_replyPollSuspendUntilMs = millis() + 4000;
  return true;
}

void setup() {
  Serial.begin(115200);
  delay(1200);

  Serial.println();
  Serial.println("ESP32 live assistant client");
  Serial.println("This device streams mic audio and only plays assistant replies.");

  configureI2S();
  connectWifi();
  g_state = AssistantState::kIdleStreaming;
  logState("ready to stream audio");
}

void loop() {
  if (g_state == AssistantState::kError) {
    const unsigned long now = millis();
    if (now - g_lastRecoverAttemptMs >= 2000) {
      g_lastRecoverAttemptMs = now;
      if (WiFi.status() != WL_CONNECTED) {
        Serial.println("WiFi disconnected, reconnecting...");
        connectWifi();
      }
      g_uploadFill = 0;
      g_consecutiveUploadFailures = 0;
      g_state = AssistantState::kIdleStreaming;
      logState("recovered from error, resuming stream");
    }
    delay(20);
    return;
  }

  size_t bytesRead = 0;
  const esp_err_t readErr =
      i2s_read(audio::kI2SPort, g_rxBuffer, sizeof(g_rxBuffer), &bytesRead, portMAX_DELAY);

  if (readErr != ESP_OK) {
    g_state = AssistantState::kError;
    logState("i2s_read failed");
    delay(250);
    return;
  }

  const int sampleCount = bytesRead / sizeof(int32_t);
  static int16_t pcm16[AUDIO_BUFFER_SAMPLES];

  int32_t peak = 0;
  for (int i = 0; i < sampleCount; ++i) {
    int32_t sample24 = g_rxBuffer[i] >> audio::kMicShift;
    if (sample24 > 32767) {
      sample24 = 32767;
    } else if (sample24 < -32768) {
      sample24 = -32768;
    }

    pcm16[i] = static_cast<int16_t>(sample24);
    const int32_t magnitude = sample24 >= 0 ? sample24 : -sample24;
    if (magnitude > peak) {
      peak = magnitude;
    }
  }

  if (g_state == AssistantState::kIdleStreaming) {
    if (!appendAndMaybeStream(pcm16, sampleCount)) {
      ++g_consecutiveUploadFailures;
      Serial.printf("audio upload failed (%d/%d)\n",
                    g_consecutiveUploadFailures,
                    audio::kMaxConsecutiveUploadFailures);
      if (g_consecutiveUploadFailures >= audio::kMaxConsecutiveUploadFailures) {
        g_state = AssistantState::kError;
        logState("too many consecutive audio upload failures");
        delay(300);
        return;
      }
    } else {
      g_consecutiveUploadFailures = 0;
    }
  }

  const unsigned long now = millis();
  if (g_state == AssistantState::kIdleStreaming &&
      g_consecutiveUploadFailures == 0 &&
      now >= g_replyPollSuspendUntilMs &&
      now - g_lastReplyPollMs >= audio::kReplyPollIntervalMs) {
    g_lastReplyPollMs = now;
    if (replyReady()) {
      g_state = AssistantState::kFetchingReply;
      logState("reply available");
      if (!fetchAndPlayReply()) {
        g_state = AssistantState::kIdleStreaming;
        logState("reply fetch failed, continuing stream");
      }
    }
  }

  writeSilenceFrame(sampleCount);

  if (now - g_lastLogMs >= 1000) {
    Serial.printf("streaming samples=%d peak=%ld wifi=%d state=%d\n",
                  sampleCount,
                  static_cast<long>(peak),
                  WiFi.status(),
                  static_cast<int>(g_state));
    g_lastLogMs = now;
  }
}

