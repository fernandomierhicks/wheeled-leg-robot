#include <Arduino.h>
#include <Adafruit_GFX.h>
#include <Adafruit_ST7789.h>
#include <SPI.h>
#include <FastLED.h>
#include <WiFi.h>
#include <string.h>
#include "config.h"
#include "CommLink.h"
#include "comm_protocol.h"
#include "udp_stream.h"
#include "wifi_config.h"

// ── NeoPixel ──────────────────────────────────────────────────────────────────

#define PIN_NEO    13
#define NUM_LEDS   30
#define BRIGHTNESS 200

// Robot state values — must match RobotStateEnum in teensy/src/robot_state.h
enum : uint8_t {
    RS_STARTUP     = 0,
    RS_CALIBRATION = 1,
    RS_STANDBY     = 2,
    RS_RUNNING     = 3,
    RS_ESTOP       = 4,
    RS_MANUAL      = 5,
    RS_CMD_REJECT  = 6,
};

static CRGB leds[NUM_LEDS];
static volatile uint8_t  g_robot_state      = RS_STARTUP;
static volatile bool     g_teensy_ever_heard = false;
static volatile uint32_t g_last_teensy_ms   = 0;
static volatile bool     g_display_dirty    = false;

static void neo_task(void*) {
    FastLED.addLeds<WS2812B, PIN_NEO, GRB>(leds, NUM_LEDS);
    FastLED.setBrightness(BRIGHTNESS);

    uint32_t tick = 0;

    for (;;) {
        uint8_t state  = g_robot_state;
        bool    linked = g_teensy_ever_heard && ((millis() - g_last_teensy_ms) < 1000);

        CRGB base;
        bool blink_fast = false;   // CMD_REJECT: fast orange blink
        bool blink_slow = false;   // ESTOP: slow red blink

        if (!linked) {
            // No data from Teensy — dim white breathe
            base = CRGB(80, 80, 80);
        } else {
            switch (state) {
                case RS_STARTUP:     base = CRGB(255, 255, 255); break;
                case RS_CALIBRATION: base = CRGB(0,   0,   255); break;
                case RS_STANDBY:     base = CRGB(255, 200,   0); break;
                case RS_RUNNING:     base = CRGB(0,   255,   0); break;
                case RS_ESTOP:       base = CRGB(255,   0,   0); blink_slow = true; break;
                case RS_MANUAL:      base = CRGB(0,   200, 255); break;
                case RS_CMD_REJECT:  base = CRGB(255,  80,   0); blink_fast = true; break;
                default:             base = CRGB(255, 255, 255); break;
            }
        }

        if (blink_slow) {
            // ~2 Hz: 250 ms on, 250 ms off  (tick period = 20 ms → 25 ticks/cycle)
            bool on = (tick % 25) < 12;
            fill_solid(leds, NUM_LEDS, on ? base : CRGB::Black);
        } else if (blink_fast) {
            // ~8 Hz: 60 ms on, 60 ms off  (tick period = 20 ms → 6 ticks/cycle)
            bool on = (tick % 6) < 3;
            fill_solid(leds, NUM_LEDS, on ? base : CRGB::Black);
        } else {
            uint8_t phase = (uint8_t)((tick % 100) * 256 / 100);
            uint8_t bri   = sin8(phase);
            bri = map(bri, 0, 255, 30, 255);
            CRGB c = base;
            c.nscale8(bri);
            fill_solid(leds, NUM_LEDS, c);
        }

        FastLED.show();
        tick++;
        vTaskDelay(pdMS_TO_TICKS(20));
    }
}

// ── CommLink instances ────────────────────────────────────────────────────────

static CommLink g_teensy(Serial2, COMM_SRC_ESP32);  // Teensy UART bridge
static CommLink g_usb(Serial,    COMM_SRC_ESP32);   // USB serial to PC

// ── Telemetry state for display ───────────────────────────────────────────────

static volatile float    g_telem_test_val = 0.0f;
static volatile uint8_t  g_fault_code     = FAULT_NONE;

// ── WiFi / TCP / UDP ──────────────────────────────────────────────────────────

static WiFiServer  g_tcp_server(CMD_TCP_PORT);
static WiFiClient  g_tcp_client;
static CommLink*   g_comm_tcp   = nullptr;   // allocated per connection
static UDPStream   g_udp_stream;
static CommLink    g_telem_udp(g_udp_stream, COMM_SRC_ESP32);
static bool        g_wifi_inited = false;

// ── Command routing ───────────────────────────────────────────────────────────

static const char* mode_name(uint8_t state);  // defined in Display section below

// Rolling log of the last few decoded commands, newest first.
#define CMD_LOG_SIZE 3
static char     g_cmd_log[CMD_LOG_SIZE][24];
static uint32_t g_cmd_log_ms[CMD_LOG_SIZE];
static uint8_t  g_cmd_log_count = 0;
static volatile bool g_cmd_log_dirty = false;

// Decode a CommandPayload into a short human-readable string.
static void decode_command(const uint8_t* payload, uint16_t len, char* out, size_t outlen) {
    if (len < 1) {
        snprintf(out, outlen, "CMD ?");
        return;
    }
    uint8_t cmd_id = payload[0];
    switch (cmd_id) {
        case CMD_ID_SET_MODE:
            if (len >= 2)
                snprintf(out, outlen, "MODE -> %s", mode_name(payload[1]));
            else
                snprintf(out, outlen, "SET MODE");
            break;
        case CMD_ID_HIP:
            if (len >= 3) {
                const char* motor = (payload[1] == HIP_MOTOR_L) ? "L"
                                   : (payload[1] == HIP_MOTOR_R) ? "R" : "BOTH";
                const char* sub;
                switch (payload[2]) {
                    case HIP_SUB_DISABLE: sub = "DISABLE"; break;
                    case HIP_SUB_ENABLE:  sub = "ENABLE";  break;
                    case HIP_SUB_ZERO:    sub = "ZERO";    break;
                    case HIP_SUB_MIT:     sub = "MIT";     break;
                    default:              sub = "?";       break;
                }
                snprintf(out, outlen, "HIP %s %s", motor, sub);
            } else {
                snprintf(out, outlen, "HIP");
            }
            break;
        case CMD_ID_REBOOT:
            snprintf(out, outlen, "REBOOT");
            break;
        case CMD_ID_PARAM_SET:
            if (len >= 3) {
                uint16_t pid;
                memcpy(&pid, payload + 1, 2);
                snprintf(out, outlen, "PSET 0x%04X", pid);
            } else {
                snprintf(out, outlen, "PARAM SET");
            }
            break;
        case CMD_ID_PARAM_GET:
            if (len >= 3) {
                uint16_t pid;
                memcpy(&pid, payload + 1, 2);
                if (pid == 0xFFFF)
                    snprintf(out, outlen, "PARAM DUMP");
                else
                    snprintf(out, outlen, "PGET 0x%04X", pid);
            } else {
                snprintf(out, outlen, "PARAM GET");
            }
            break;
        default:
            snprintf(out, outlen, "CMD 0x%02X", cmd_id);
            break;
    }
}

// Push a newly decoded command onto the front of the rolling log.
static void log_command(const uint8_t* payload, uint16_t len) {
    for (int i = CMD_LOG_SIZE - 1; i > 0; i--) {
        memcpy(g_cmd_log[i], g_cmd_log[i - 1], sizeof(g_cmd_log[i]));
        g_cmd_log_ms[i] = g_cmd_log_ms[i - 1];
    }
    decode_command(payload, len, g_cmd_log[0], sizeof(g_cmd_log[0]));
    g_cmd_log_ms[0] = millis();
    if (g_cmd_log_count < CMD_LOG_SIZE)
        g_cmd_log_count++;
    g_cmd_log_dirty = true;
}

// Forward any COMMAND packet (from USB or WiFi TCP) to the Teensy UART.
static void forward_to_teensy(uint8_t type, uint8_t version, uint8_t /*source*/,
                               const uint8_t* payload, uint16_t len) {
    if (type == COMM_TYPE_COMMAND) {
        g_teensy.send(type, version, payload, len);
        log_command(payload, len);
    }
}

// ── Teensy → all outputs ──────────────────────────────────────────────────────

static void on_teensy_packet(uint8_t type, uint8_t version, uint8_t /*source*/,
                              const uint8_t* payload, uint16_t len) {
    // Forward to USB only when the TX buffer has room — avoids blocking if
    // the host port is closed and nobody is draining the UART TX buffer.
    if (Serial.availableForWrite() >= (int)(len + 9))
        g_usb.send(type, version, payload, len);

    // Forward to any connected WiFi TCP client (commands + telemetry both ways)
    if (g_comm_tcp && g_tcp_client.connected())
        g_comm_tcp->send(type, version, payload, len);

    // Broadcast telemetry-only over UDP for lightweight monitoring clients
    if (type == COMM_TYPE_TELEMETRY && g_wifi_inited)
        g_telem_udp.send(type, version, payload, len);

    // Extract robot state and dummy value; mark UART as active.
    // Use memcpy to read floats from the byte buffer — ESP32's Xtensa FPU
    // requires 4-byte alignment and _rx_buf may not satisfy that.
    if (type == COMM_TYPE_TELEMETRY && version == TELEM_PAYLOAD_V2
            && len >= sizeof(TelemetryPayload)) {
        float   test_val;
        uint8_t new_state = payload[offsetof(TelemetryPayload, robot_state)];
        uint8_t new_fault = payload[offsetof(TelemetryPayload, fault_code)];
        if (new_state != g_robot_state || new_fault != g_fault_code)
            g_display_dirty = true;
        g_robot_state    = new_state;
        g_fault_code     = new_fault;
        memcpy(&test_val, payload + offsetof(TelemetryPayload, test_val), sizeof(float));
        g_telem_test_val = test_val;
    }
    if (!g_teensy_ever_heard) g_display_dirty = true;  // first packet: show link active
    g_last_teensy_ms     = millis();
    g_teensy_ever_heard  = true;
}

// ── Display ───────────────────────────────────────────────────────────────────

static Adafruit_ST7789 tft(TFT_CS, TFT_DC, TFT_RST);

static uint16_t mode_color_565(uint8_t state) {
    switch (state) {
        case RS_STARTUP:     return ST77XX_WHITE;
        case RS_CALIBRATION: return ST77XX_BLUE;
        case RS_STANDBY:     return ST77XX_YELLOW;
        case RS_RUNNING:     return ST77XX_GREEN;
        case RS_ESTOP:       return ST77XX_RED;
        case RS_MANUAL:      return ST77XX_CYAN;
        case RS_CMD_REJECT:  return tft.color565(255, 100, 0);  // orange
        default:             return ST77XX_WHITE;
    }
}

static const char* mode_name(uint8_t state) {
    switch (state) {
        case RS_STARTUP:     return "STARTUP";
        case RS_CALIBRATION: return "CALIBRATION";
        case RS_STANDBY:     return "STANDBY";
        case RS_RUNNING:     return "RUNNING";
        case RS_ESTOP:       return "ESTOP";
        case RS_MANUAL:      return "MANUAL";
        case RS_CMD_REJECT:  return "CMD REJECT";
        default:             return "UNKNOWN";
    }
}

// Verbose fault descriptions — MIRROR: software/gui/flash_monitor.py _FAULT_DESCRIPTIONS
static const char* fault_description(uint8_t code) {
    switch (code) {
        case FAULT_NONE:              return "";
        case FAULT_IMU_ERROR:         return "IMU reported ERROR during startup";
        case FAULT_HIP_INIT_TIMEOUT:  return "No CAN reply from hip motors within 2s of boot";
        case FAULT_HIP_FEEDBACK_LOST: return "Hip CAN feedback timed out during operation";
        case FAULT_HIP_LARGE_POS_CMD: return "Commanded hip position jump too large";
        case FAULT_CALIBRATION_TIMEOUT: return "Hardstop not found during calibration";
        case FAULT_HUMAN_ESTOP:       return "Human triggered ESTOP";
        default:                      return "Unknown fault";
    }
}

// Redraws only fields that changed; call from loop() at ~10 Hz.
static void update_display() {
    static uint8_t prev_state  = 0xFF;
    static uint8_t prev_fault  = 0xFF;
    static bool    prev_active = false;
    static float   prev_val    = -9999.0f;

    uint8_t state  = g_robot_state;
    uint8_t fault  = g_fault_code;
    bool    active = g_teensy_ever_heard && ((millis() - g_last_teensy_ms) < 1000);
    float   val    = g_telem_test_val;

    if (state != prev_state || fault != prev_fault || active != prev_active) {
        // "Mode:" label (static, but repaint on first draw)
        tft.fillRect(0, 5, 320, 80, ST77XX_BLACK);
        tft.setTextSize(2);
        tft.setTextColor(ST77XX_WHITE);
        tft.setCursor(10, 10);
        tft.print("Mode:");
        tft.fillRect(0, 35, 320, 40, ST77XX_BLACK);
        tft.setTextSize(4);
        if (active) {
            tft.setTextColor(mode_color_565(state));
            tft.setCursor(10, 38);
            tft.print(mode_name(state));
        } else {
            tft.setTextColor(tft.color565(80, 80, 80));  // dark gray
            tft.setCursor(10, 38);
            tft.print("--");
        }

        // Verbose fault description, shown only while latched in ESTOP
        tft.fillRect(0, 85, 320, 12, ST77XX_BLACK);
        if (state == RS_ESTOP && fault != FAULT_NONE) {
            tft.setTextSize(1);
            tft.setTextColor(ST77XX_YELLOW);
            tft.setCursor(10, 88);
            tft.print(fault_description(fault));
        }
        prev_state = state;
        prev_fault = fault;
    }

    if (active != prev_active) {
        tft.fillRect(0, 100, 320, 28, ST77XX_BLACK);
        tft.setTextSize(2);
        if (active) {
            tft.setTextColor(ST77XX_GREEN);
            tft.setCursor(10, 105);
            tft.print("UART: Active");
        } else {
            tft.setTextColor(ST77XX_YELLOW);
            tft.setCursor(10, 105);
            tft.print("UART: Waiting for Teensy");
        }
        prev_active = active;
    }

    if (fabsf(val - prev_val) > 0.0005f) {
        tft.fillRect(0, 145, 320, 28, ST77XX_BLACK);
        tft.setTextSize(2);
        tft.setTextColor(ST77XX_WHITE);
        tft.setCursor(10, 150);
        tft.print("Dummy: ");
        tft.print(val, 3);
        prev_val = val;
    }

    // Redraw on a new command, or once a second so the "ago" times stay current.
    static uint32_t last_log_redraw_ms = 0;
    if (g_cmd_log_dirty || (millis() - last_log_redraw_ms >= 1000)) {
        tft.fillRect(0, 178, 320, 62, ST77XX_BLACK);
        tft.setTextSize(1);
        tft.setTextColor(ST77XX_WHITE);
        tft.setCursor(10, 180);
        tft.print("Last commands:");
        tft.setTextColor(ST77XX_CYAN);
        for (uint8_t i = 0; i < g_cmd_log_count; i++) {
            uint32_t age_s = (millis() - g_cmd_log_ms[i]) / 1000;
            tft.setCursor(10, 195 + i * 14);
            tft.printf("%3lus  %s", (unsigned long)age_s, g_cmd_log[i]);
        }
        g_cmd_log_dirty   = false;
        last_log_redraw_ms = millis();
    }
}

// ── Setup / loop ──────────────────────────────────────────────────────────────

void setup() {
    Serial.begin(115200);
    Serial2.begin(TEENSY_UART_BAUD, SERIAL_8N1, TEENSY_UART_RX, TEENSY_UART_TX);

    g_teensy.onPacket(on_teensy_packet);
    g_usb.onPacket(forward_to_teensy);

    WiFi.setSleep(false);
    WiFi.begin(WIFI_SSID, WIFI_PASS);

    pinMode(TFT_BLK, OUTPUT);
    digitalWrite(TFT_BLK, HIGH);
    tft.init(240, 320);
    tft.setRotation(1);
    tft.fillScreen(ST77XX_BLACK);
    // Initial render — update_display() will fill in live values on first loop tick

    xTaskCreatePinnedToCore(neo_task, "neo", 2048, nullptr, 1, nullptr, 0);

    Serial.println("[ESP32] ready");
}

void loop() {
    // Drive all CommLink parsers
    g_teensy.update();
    g_usb.update();

    if (g_comm_tcp && g_tcp_client.connected())
        g_comm_tcp->update();

    // One-time WiFi init: start TCP server + UDP once WiFi comes up
    if (WiFi.status() == WL_CONNECTED && !g_wifi_inited) {
        g_tcp_server.begin();
        g_udp_stream.beginSend("255.255.255.255", TELEM_UDP_PORT);
        g_wifi_inited = true;
        Serial.print("[WiFi] IP: ");
        Serial.println(WiFi.localIP());
    }

    // Reconnect WiFi if it drops
    static uint32_t last_reconnect_ms = 0;
    if (WiFi.status() != WL_CONNECTED) {
        g_wifi_inited = false;
        if (millis() - last_reconnect_ms > 5000) {
            WiFi.begin(WIFI_SSID, WIFI_PASS);
            last_reconnect_ms = millis();
        }
    }

    // Refresh display: immediately on state change, otherwise at ~10 Hz
    static uint32_t last_disp_ms = 0;
    if (g_display_dirty || (millis() - last_disp_ms >= 100)) {
        g_display_dirty = false;
        update_display();
        last_disp_ms = millis();
    }

    // Accept a new TCP client (one at a time; replaces any dropped connection)
    if (g_wifi_inited && g_tcp_server.hasClient()) {
        WiFiClient c = g_tcp_server.available();
        if (c) {
            g_tcp_client = c;
            delete g_comm_tcp;
            g_comm_tcp = new CommLink(g_tcp_client, COMM_SRC_ESP32);
            g_comm_tcp->onPacket(forward_to_teensy);
            Serial.print("[TCP] client: ");
            Serial.println(g_tcp_client.remoteIP());
        }
    }
}
