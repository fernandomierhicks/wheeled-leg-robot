#include <Arduino.h>
#include <Adafruit_GFX.h>
#include <Adafruit_ST7789.h>
#include <SPI.h>
#include <FastLED.h>
#include <WiFi.h>
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
};

static CRGB leds[NUM_LEDS];
static volatile uint8_t g_robot_state = RS_STARTUP;

static void neo_task(void*) {
    FastLED.addLeds<WS2812B, PIN_NEO, GRB>(leds, NUM_LEDS);
    FastLED.setBrightness(BRIGHTNESS);

    uint32_t tick = 0;

    for (;;) {
        uint8_t state = g_robot_state;

        CRGB base;
        bool blink_mode = false;
        switch (state) {
            case RS_STARTUP:     base = CRGB(255, 255, 255); break;
            case RS_CALIBRATION: base = CRGB(0,   0,   255); break;
            case RS_STANDBY:     base = CRGB(255, 200,   0); break;
            case RS_RUNNING:     base = CRGB(0,   255,   0); break;
            case RS_ESTOP:       base = CRGB(255,   0,   0); blink_mode = true; break;
            case RS_MANUAL:      base = CRGB(0,   200, 255); break;
            default:             base = CRGB(255, 255, 255); break;
        }

        if (blink_mode) {
            bool on = (tick % 10) < 5;
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

static volatile uint32_t g_last_teensy_ms = 0;
static volatile float    g_telem_test_val = 0.0f;

// ── WiFi / TCP / UDP ──────────────────────────────────────────────────────────

static WiFiServer  g_tcp_server(CMD_TCP_PORT);
static WiFiClient  g_tcp_client;
static CommLink*   g_comm_tcp   = nullptr;   // allocated per connection
static UDPStream   g_udp_stream;
static CommLink    g_telem_udp(g_udp_stream, COMM_SRC_ESP32);
static bool        g_wifi_inited = false;

// ── Command routing ───────────────────────────────────────────────────────────

// Forward any COMMAND packet (from USB or WiFi TCP) to the Teensy UART.
static void forward_to_teensy(uint8_t type, uint8_t version, uint8_t /*source*/,
                               const uint8_t* payload, uint16_t len) {
    if (type == COMM_TYPE_COMMAND)
        g_teensy.send(type, version, payload, len);
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

    // Extract robot state and dummy value; mark UART as active
    if (type == COMM_TYPE_TELEMETRY && version == TELEM_PAYLOAD_V1
            && len >= sizeof(TelemetryPayload)) {
        const TelemetryPayload* t = reinterpret_cast<const TelemetryPayload*>(payload);
        g_robot_state    = t->robot_state;
        g_telem_test_val = t->test_val;
    }
    g_last_teensy_ms = millis();
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
        default:             return "UNKNOWN";
    }
}

// Redraws only fields that changed; call from loop() at ~10 Hz.
static void update_display() {
    static uint8_t prev_state  = 0xFF;
    static bool    prev_active = false;
    static float   prev_val    = -9999.0f;

    uint8_t state  = g_robot_state;
    bool    active = (millis() - g_last_teensy_ms) < 1000;
    float   val    = g_telem_test_val;

    if (state != prev_state) {
        // "Mode:" label (static, but repaint on first draw)
        tft.fillRect(0, 5, 320, 80, ST77XX_BLACK);
        tft.setTextSize(2);
        tft.setTextColor(ST77XX_WHITE);
        tft.setCursor(10, 10);
        tft.print("Mode:");
        tft.fillRect(0, 35, 320, 40, ST77XX_BLACK);
        tft.setTextSize(4);
        tft.setTextColor(mode_color_565(state));
        tft.setCursor(10, 38);
        tft.print(mode_name(state));
        prev_state = state;
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

    // Refresh display at ~10 Hz
    static uint32_t last_disp_ms = 0;
    if (millis() - last_disp_ms >= 100) {
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
