// VL53L1X demo — 4 sensors, sequential XSHUT init with address reassignment
// Wiring: SDA=GPIO21  SCL=GPIO22
//   XSHUT[0]=GPIO14 → 0x30,  XSHUT[1]=GPIO27 → 0x31
//   XSHUT[2]=GPIO26 → 0x32,  XSHUT[3]=GPIO25 → 0x33
// NOTE: GPIO25 is also DAC1; BSP boot may reconfigure it, so we re-assert
//       XSHUT HIGH before the first read and re-init on timeout.

#include <Arduino.h>
#include <Wire.h>
#include <VL53L1X.h>
#include <driver/dac.h>   // to disable DAC on GPIO25/26 explicitly

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

static void i2c_scan() {
    Serial.print("  I2C scan:");
    for (uint8_t addr = 1; addr < 127; addr++) {
        Wire.beginTransmission(addr);
        if (Wire.endTransmission() == 0) Serial.printf(" 0x%02X", addr);
    }
    Serial.println();
}

static bool init_sensor(uint8_t i) {
    pinMode(XSHUT_PINS[i], OUTPUT);
    digitalWrite(XSHUT_PINS[i], HIGH);
    delay(10);
    // Reset the library's stored address to the VL53L1X boot default.
    // After XSHUT power-cycle the sensor reverts to 0x29; without this,
    // init() would still try to talk to the old reassigned address.
    // setAddress() writes to whatever the current address is (may fail if sensor
    // is gone), but always updates the internal field — that side-effect is what
    // we need here.
    sensors[i].setAddress(0x29);
    if (!sensors[i].init()) {
        Serial.printf("Sensor %u (GPIO%u) not found\n", i, XSHUT_PINS[i]);
        i2c_scan();
        i2c_recover();
        return false;
    }
    sensors[i].setAddress(ADDRESSES[i]);
    sensors[i].setTimeout(500);
    sensors[i].setDistanceMode(VL53L1X::Short);
    sensors[i].setMeasurementTimingBudget(50000);
    Serial.printf("Sensor %u (GPIO%u) OK → 0x%02X\n", i, XSHUT_PINS[i], ADDRESSES[i]);
    return true;
}

void setup() {
    Serial.begin(115200);

    // Disable DAC on GPIO25/GPIO26 so those pins work as digital XSHUT outputs.
    // Without this the ESP32 BSP may leave them in analog/DAC mode after boot,
    // letting the XSHUT line droop LOW and putting the sensor back into reset.
    dac_output_disable(DAC_CHANNEL_1); // GPIO25
    dac_output_disable(DAC_CHANNEL_2); // GPIO26

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
        sensor_ok[i] = init_sensor(i);
    }
}

void loop() {
    // Re-assert all XSHUT pins HIGH each loop to guard against DAC/peripheral
    // reconfiguring GPIO25/26 under us after setup().
    for (uint8_t i = 0; i < NUM_SENSORS; i++) {
        if (sensor_ok[i]) digitalWrite(XSHUT_PINS[i], HIGH);
    }

    for (uint8_t i = 0; i < NUM_SENSORS; i++) {
        if (!sensor_ok[i]) {
            Serial.printf("[%u]   N/A      ", i);
            continue;
        }
        uint16_t dist = sensors[i].readSingle(true);
        if (sensors[i].timeoutOccurred()) {
            Serial.printf("[%u] TIMEOUT   ", i);
            // Attempt recovery: power-cycle this sensor and re-init it
            Serial.printf(" (re-init sensor %u)\n", i);
            digitalWrite(XSHUT_PINS[i], LOW);
            delay(10);
            sensor_ok[i] = init_sensor(i);
        } else if (sensors[i].ranging_data.range_status != 0) {
            uint8_t st = sensors[i].ranging_data.range_status;
            const char* desc;
            switch (st) {
                case 1:  desc = "sigma_fail(noise)"; break;
                case 2:  desc = "signal_fail(weak)"; break;
                case 3:  desc = "min_range(too_close)"; break;
                case 4:  desc = "phase_fail(out_of_range)"; break;
                case 5:  desc = "hardware_fail"; break;
                case 7:  desc = "wraparound"; break;
                default: desc = "unknown"; break;
            }
            Serial.printf("[%u] err%u=%s  ", i, st, desc);
        } else {
            Serial.printf("[%u] %6.1f cm  ", i, dist / 10.0f);
        }
    }
    Serial.println();
}
