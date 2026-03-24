#pragma once
#include <SPI.h>
#include <Adafruit_BNO08x.h>

#ifndef BNO_CS_PIN
#define BNO_CS_PIN  10
#endif
#ifndef BNO_INT_PIN
#define BNO_INT_PIN 2
#endif
#ifndef BNO_RST_PIN
#define BNO_RST_PIN 3
#endif

Adafruit_BNO08x bno08x(BNO_RST_PIN);
sh2_SensorValue_t sensorValue;

// ---------- INT-pin fast-check via direct port register ----------
// digitalRead() costs ~3us on RA4M1 due to pin table lookup.
// Direct register read is ~0.1us.  D2 = P104 = Port 1, Bit 4.
// RA4M1 PORT1 base = 0x40040020, PIDR offset = 0x06
#define PORT1_PIDR  (*(volatile const uint16_t *)0x40040026)
static inline bool int_asserted() {
    return (PORT1_PIDR & (1U << 4)) == 0;  // H_INTN active-low
}

bool init_bno086() {
    for (int attempt = 1; attempt <= 3; attempt++) {
        if (bno08x.begin_SPI(BNO_CS_PIN, BNO_INT_PIN)) {
            Serial.print("[INFO] BNO086 init OK (Adafruit lib) on attempt ");
            Serial.println(attempt);
            delay(100);
            if (bno08x.wasReset()) {
                Serial.println("[INFO] Post-init reset consumed");
                bno08x.enableReport(SH2_GAME_ROTATION_VECTOR, 100000);  // dummy to ack reset
            }
            // Poll a few times to let SH2 handshake complete
            for (int i = 0; i < 10; i++) {
                bno08x.getSensorEvent(&sensorValue);
                delay(10);
            }
            return true;
        }
        Serial.print("[ERROR] BNO086 not detected (attempt ");
        Serial.print(attempt); Serial.println("/3)");
        Serial.println("[ERROR]   CS=D10, INT=D2, RST=D3, SCK=D13, MOSI=D11, MISO=D12");
        Serial.println("[ERROR]   Solder jumpers PS0+PS1 must be bridged for SPI mode");
        delay(3000);
    }
    Serial.println("[FATAL] BNO086 init failed after 3 attempts");
    return false;
}

// Re-enable reports after a BNO reset (call from loop if wasReset() fires)
typedef void (*ReportEnableFn)();
ReportEnableFn activeReportEnabler = nullptr;

void check_and_handle_reset() {
    if (bno08x.wasReset()) {
        Serial.println("[WARN] BNO086 reset detected — re-enabling reports");
        if (activeReportEnabler) activeReportEnabler();
    }
}

void print_menu() {
    Serial.println();
    Serial.println("=== IMU SPEED CHARACTERIZATION (Adafruit BNO08x) ===");
    Serial.println("1 - Game Rotation Vector max datarate (10s)");
    Serial.println("2 - Raw gyro-Y max datarate (10s)");
    Serial.println("3 - Calibrated gyro-Y (ARVR Gyro) max datarate (10s)");
    Serial.println("A - All three tests sequentially");
    Serial.println("Select test [1-3, A]: ");
}
