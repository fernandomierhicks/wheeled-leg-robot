#include <Arduino.h>
#include <WiFiS3.h>
#include <ArduinoOTA.h>
#include "bno086_common.h"
#include "test_datarate_grv.h"
#include "test_datarate_raw_gyro.h"
#include "test_datarate_cal_gyro.h"

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
        while (true) { ArduinoOTA.poll(); delay(1000); }
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
                print_menu();
                break;
            case '2':
                run_test2_raw_gyro_datarate();
                print_menu();
                break;
            case '3':
                run_test3_cal_gyro_datarate();
                print_menu();
                break;
            case 'A':
            case 'a':
                Serial.println("=== Running all 3 speed tests sequentially ===");
                run_test1_grv_datarate();
                delay(500);
                run_test2_raw_gyro_datarate();
                delay(500);
                run_test3_cal_gyro_datarate();
                Serial.println("=== All tests complete ===");
                print_menu();
                break;
            default:
                break;
        }
    }
}
