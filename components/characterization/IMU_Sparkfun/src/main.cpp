#include <Arduino.h>
#include <WiFiS3.h>
#include <ArduinoOTA.h>
#include "bno086_common.h"
#include "test_datarate_grv.h"
#include "test_spi_latency.h"
#include "test_e2e_latency.h"
#include "test_noise.h"
#include "test_static_drift.h"

void setup() {
    Serial.begin(SERIAL_BAUD);
    while (!Serial) { delay(10); }

    // Connect to WiFi for OTA
    Serial.print("[INFO] Connecting to WiFi...");
    WiFi.begin("Minnexico", "Fenway724");
    while (WiFi.status() != WL_CONNECTED) { delay(1000); Serial.print("."); }
    Serial.print(" OK, IP: ");
    Serial.println(WiFi.localIP());
    ArduinoOTA.begin(WiFi.localIP(), "Arduino_R4", "arduino", InternalStorage);

    if (!init_bno086()) {
        while (true) { ArduinoOTA.poll(); delay(1000); }  // halt but keep OTA alive
    }

    print_menu();
}

void loop() {
    ArduinoOTA.poll();

    if (Serial.available()) {
        char c = Serial.read();
        switch (c) {
            case '1':
                run_test1_grv_datarate();
                break;
            case 'F':
            case 'f':
                run_test1_grv_datarate_full();
                break;
            case '2':
                Serial.println("[TEST2] Raw Datarate — not yet implemented");
                break;
            case '3':
                run_test3_static_drift();
                break;
            case '4':
                run_test4_spi_latency();
                break;
            case '5':
                run_test5_noise();
                break;
            case '6':
                run_test6_e2e_latency();
                break;
            default:
                break;
        }
        if (c >= '1' && c <= '6' || c == 'F' || c == 'f') {
            print_menu();
        }
    }
}
