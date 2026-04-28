// CAN ↔ USB-Serial bridge (SLCAN protocol)
// Implements the SLCAN subset that python-can's 'slcan' interface expects.
//
// Wire ODrive CAN to UNO R4 WiFi: CANTX → D4, CANRX → D5, common GND.
// CAN baud fixed at 1 Mbit/s (matching ODrive default).
//
// Python usage:
//   import can
//   bus = can.Bus(interface='slcan', channel='COMx', bitrate=1000000, sleep_after_open=0.1)
//
// SLCAN command subset implemented:
//   O\r          open CAN channel
//   C\r          close CAN channel
//   Sx\r         set bitrate (accepted, ignored — always 1M)
//   V\r          firmware version query
//   N\r          serial number query
//   tIIIDXX\r    send standard 11-bit frame   → z\r on success
//   TXXXXXXXXDXX\r send extended 29-bit frame → Z\r on success
//   Received frames are printed in same t/T format.

#include <Arduino.h>
#include <Arduino_CAN.h>

static bool g_open = false;

static char rx_buf[32];
static uint8_t rx_len = 0;

static int hexval(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static void send_frame(bool extended, uint32_t id, uint8_t dlc, const uint8_t* data) {
    CanMsg msg(extended ? CanExtendedId(id) : CanStandardId(id), dlc, data);
    if (CAN.write(msg) > 0) {
        Serial.print(extended ? "Z\r" : "z\r");
    } else {
        Serial.print("\a");
    }
}

static void process(const char* cmd, uint8_t len) {
    if (len == 0) return;

    switch (cmd[0]) {

    case 'O':
        if (!g_open) {
            if (CAN.begin(CanBitRate::BR_1000k)) {
                g_open = true;
                Serial.print("\r");
            } else {
                Serial.print("\a");
            }
        } else {
            Serial.print("\r");
        }
        break;

    case 'C':
        if (g_open) { CAN.end(); g_open = false; }
        Serial.print("\r");
        break;

    case 'S':   // bitrate select — ignored, always 1M
        Serial.print("\r");
        break;

    case 'V':
        Serial.print("V0101\r");
        break;

    case 'N':
        Serial.print("NFFFF\r");
        break;

    case 't': {  // standard 11-bit frame: tIIIDXX...
        if (!g_open || len < 5) { Serial.print("\a"); return; }
        int id = (hexval(cmd[1]) << 8) | (hexval(cmd[2]) << 4) | hexval(cmd[3]);
        int dlc = hexval(cmd[4]);
        if (id < 0 || dlc < 0 || dlc > 8 || len < (uint8_t)(5 + dlc * 2)) {
            Serial.print("\a"); return;
        }
        uint8_t data[8];
        for (int i = 0; i < dlc; i++) {
            int hi = hexval(cmd[5 + i * 2]);
            int lo = hexval(cmd[6 + i * 2]);
            if (hi < 0 || lo < 0) { Serial.print("\a"); return; }
            data[i] = (uint8_t)((hi << 4) | lo);
        }
        send_frame(false, (uint32_t)id, (uint8_t)dlc, data);
        break;
    }

    case 'T': {  // extended 29-bit frame: TXXXXXXXXDXX...
        if (!g_open || len < 10) { Serial.print("\a"); return; }
        uint32_t id = 0;
        for (int i = 1; i <= 8; i++) {
            int v = hexval(cmd[i]);
            if (v < 0) { Serial.print("\a"); return; }
            id = (id << 4) | (uint32_t)v;
        }
        int dlc = hexval(cmd[9]);
        if (dlc < 0 || dlc > 8 || len < (uint8_t)(10 + dlc * 2)) {
            Serial.print("\a"); return;
        }
        uint8_t data[8];
        for (int i = 0; i < dlc; i++) {
            int hi = hexval(cmd[10 + i * 2]);
            int lo = hexval(cmd[11 + i * 2]);
            if (hi < 0 || lo < 0) { Serial.print("\a"); return; }
            data[i] = (uint8_t)((hi << 4) | lo);
        }
        send_frame(true, id, (uint8_t)dlc, data);
        break;
    }

    default:
        Serial.print("\a");
        break;
    }
}

void setup() {
    Serial.begin(1000000);
    while (!Serial);
    // CAN starts on first 'O' command from host
}

void loop() {
    // USB → CAN: accumulate until \r or \n
    while (Serial.available()) {
        char c = (char)Serial.read();
        if (c == '\r' || c == '\n') {
            process(rx_buf, rx_len);
            rx_len = 0;
        } else if (rx_len < sizeof(rx_buf) - 1) {
            rx_buf[rx_len++] = c;
        }
    }

    // CAN → USB
    if (g_open && CAN.available()) {
        CanMsg const msg = CAN.read();
        char line[32];
        int n;
        if (msg.isExtendedId()) {
            n = sprintf(line, "T%08lX%d", (unsigned long)msg.id, msg.data_length);
        } else {
            n = sprintf(line, "t%03lX%d", (unsigned long)msg.id, msg.data_length);
        }
        for (uint8_t i = 0; i < msg.data_length; i++) {
            sprintf(line + n, "%02X", msg.data[i]);
            n += 2;
        }
        line[n++] = '\r';
        line[n] = '\0';
        Serial.print(line);
    }
}
