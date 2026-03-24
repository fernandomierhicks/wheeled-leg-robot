#pragma once
#include <SPI.h>
#include <SparkFun_BNO08x_Arduino_Library.h>

#ifndef BNO_CS_PIN
#define BNO_CS_PIN  10
#endif
#ifndef BNO_INT_PIN
#define BNO_INT_PIN 2
#endif
#ifndef BNO_RST_PIN
#define BNO_RST_PIN 3
#endif
#ifndef BNO_SPI_SPEED
#define BNO_SPI_SPEED 3000000  // 3MHz max
#endif

BNO08x imu;

bool init_bno086() {
    for (int attempt = 1; attempt <= 3; attempt++) {
        if (imu.beginSPI(BNO_CS_PIN, BNO_INT_PIN, BNO_RST_PIN, BNO_SPI_SPEED, SPI)) {
            Serial.print("[INFO] BNO086 init OK on attempt ");
            Serial.println(attempt);
            // Consume the post-init reset flag and let the sensor settle
            delay(100);
            if (imu.wasReset()) {
                Serial.println("[INFO] Post-init reset consumed");
            }
            // Poll a few times to let SH2 handshake complete
            for (int i = 0; i < 10; i++) {
                imu.getSensorEvent();
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

void print_menu() {
    Serial.println();
    Serial.println("=== IMU CHARACTERIZATION ===");
    Serial.println("1 - GRV max datarate — fast (timestamps only, 10s)");
    Serial.println("F - GRV datarate — full quat output (serial-limited, 10s)");
    Serial.println("2 - Raw gyro/accel max datarate (10s)");
    Serial.println("3 - Static GRV drift (5 min)");
    Serial.println("4 - SPI transaction latency (5s)");
    Serial.println("5 - Noise density (60s)");
    Serial.println("6 - End-to-end latency (5s)");
    Serial.println("Select test [1-6]: ");
}
