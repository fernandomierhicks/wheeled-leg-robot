// VL53L1X hello-world — 1 sensor
// Wiring (TX2-side of DevKit, all accessible without straddling breadboard):
//   SDA  = GPIO21  SCL  = GPIO22
//   XSHUT= GPIO19  VCC  = 3.3V   GND = GND

#include <Arduino.h>
#include <Wire.h>
#include <VL53L1X.h>

#define PIN_XSHUT 19   // TX2-side, between GPIO18 and GPIO21

static VL53L1X sensor;

void setup() {
    Serial.begin(115200);

    pinMode(PIN_XSHUT, OUTPUT);
    digitalWrite(PIN_XSHUT, LOW);   // hold in reset
    delay(10);
    digitalWrite(PIN_XSHUT, HIGH);  // release
    delay(10);                       // boot time (~1 ms needed; 10 ms safe)

    Wire.begin();           // SDA=21, SCL=22
    Wire.setClock(400000);

    if (!sensor.init()) {
        Serial.println("VL53L1X not found — check wiring");
        while (true) delay(1000);
    }

    sensor.setTimeout(500);
    sensor.setDistanceMode(VL53L1X::Short);   // up to ~1.3 m, better <50 cm
    sensor.setMeasurementTimingBudget(50000); // 50 ms per measurement
    sensor.startContinuous(50);               // 50 ms inter-measurement

    Serial.println("VL53L1X OK — ranging...");
}

void loop() {
    uint16_t dist = sensor.read(false);  // non-blocking
    if (sensor.timeoutOccurred()) {
        Serial.println("TIMEOUT");
    } else {
        Serial.printf("%5.1f cm\n", dist / 10.0f);
    }
    delay(100);
}
