// VL53L1X demo — 4 sensors, sequential XSHUT init with address reassignment
// Wiring: SDA=GPIO21  SCL=GPIO22
//   XSHUT[0]=GPIO14 → 0x30,  XSHUT[1]=GPIO27 → 0x31
//   XSHUT[2]=GPIO26 → 0x32,  XSHUT[3]=GPIO25 → 0x33

#include <Arduino.h>
#include <Wire.h>
#include <VL53L1X.h>

static const uint8_t XSHUT_PINS[4] = {14, 27, 26, 25};
static const uint8_t ADDRESSES[4]  = {0x30, 0x31, 0x32, 0x33};
static const uint8_t NUM_SENSORS   = 4;

static VL53L1X sensors[NUM_SENSORS];
static bool    sensor_ok[NUM_SENSORS] = {false};

static void i2c_recover() {
    Wire.end();
    pinMode(21, OUTPUT); digitalWrite(21, HIGH);
    pinMode(22, OUTPUT); digitalWrite(22, HIGH);
    for (int i = 0; i < 9; i++) {
        digitalWrite(22, LOW);  delayMicroseconds(5);
        digitalWrite(22, HIGH); delayMicroseconds(5);
    }
    digitalWrite(21, LOW);  delayMicroseconds(5);
    digitalWrite(22, HIGH); delayMicroseconds(5);
    digitalWrite(21, HIGH); delayMicroseconds(5);
    delay(10);
    Wire.begin();
    Wire.setClock(400000);
}

void setup() {
    Serial.begin(115200);

    // Hold all sensors in reset
    for (uint8_t i = 0; i < NUM_SENSORS; i++) {
        pinMode(XSHUT_PINS[i], OUTPUT);
        digitalWrite(XSHUT_PINS[i], LOW);
    }
    delay(10);

    Wire.begin();
    Wire.setClock(400000);

    // Bring up one sensor at a time: init at 0x29, reassign, move on
    for (uint8_t i = 0; i < NUM_SENSORS; i++) {
        digitalWrite(XSHUT_PINS[i], HIGH);
        delay(10);

        if (!sensors[i].init()) {
            Serial.printf("Sensor %u (GPIO%u) not found — skipping\n", i, XSHUT_PINS[i]);
            i2c_recover();
            continue;
        }

        sensors[i].setAddress(ADDRESSES[i]);
        sensors[i].setTimeout(500);
        sensors[i].setDistanceMode(VL53L1X::Short);
        sensors[i].setMeasurementTimingBudget(50000);
        sensors[i].startContinuous(50);
        sensor_ok[i] = true;

        Serial.printf("Sensor %u (GPIO%u) OK → 0x%02X\n", i, XSHUT_PINS[i], ADDRESSES[i]);
    }
}

void loop() {
    for (uint8_t i = 0; i < NUM_SENSORS; i++) {
        if (!sensor_ok[i]) {
            Serial.printf("[%u]   N/A      ", i);
            continue;
        }
        uint16_t dist = sensors[i].read(true);
        if (sensors[i].timeoutOccurred()) {
            Serial.printf("[%u] TIMEOUT   ", i);
        } else if (sensors[i].ranging_data.range_status != 0) {
            Serial.printf("[%u] err%-2u     ", i, sensors[i].ranging_data.range_status);
        } else {
            Serial.printf("[%u] %6.1f cm  ", i, dist / 10.0f);
        }
    }
    Serial.println();
}
