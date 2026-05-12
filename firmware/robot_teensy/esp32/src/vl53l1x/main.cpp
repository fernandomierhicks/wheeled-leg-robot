// VL53L1X hello-world — single sensor (sensor 1, GPIO26 XSHUT, default addr 0x29)
// Wiring: SDA=GPIO21  SCL=GPIO22  XSHUT=GPIO26  VCC=3.3V  GND=GND

#include <Arduino.h>
#include <Wire.h>
#include <VL53L1X.h>

#define PIN_XSHUT 26

static VL53L1X sensor;

void setup() {
    Serial.begin(115200);

    pinMode(PIN_XSHUT, OUTPUT);
    digitalWrite(PIN_XSHUT, LOW);
    delay(10);
    digitalWrite(PIN_XSHUT, HIGH);
    delay(10);

    Wire.begin();
    Wire.setClock(400000);

    if (!sensor.init()) {
        Serial.println("VL53L1X not found — check wiring");
        while (true) delay(1000);
    }

    sensor.setTimeout(500);
    sensor.setDistanceMode(VL53L1X::Short);
    sensor.setMeasurementTimingBudget(50000);
    sensor.startContinuous(50);

    Serial.println("Sensor 1 (GPIO26) OK");
}

void loop() {
    uint16_t dist = sensor.read(false);
    if (sensor.timeoutOccurred()) {
        Serial.println("[1] TIMEOUT");
    } else {
        Serial.printf("[1] %5.1f cm\n", dist / 10.0f);
    }
    delay(100);
}
