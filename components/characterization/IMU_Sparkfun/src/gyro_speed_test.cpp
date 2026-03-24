// gyro_speed_test.cpp — BNO086 gyro-Y-only speed test.
// SparkFun library takes MILLISECONDS (internally * 1000 → µs).

#include <Arduino.h>
#include <SPI.h>
#include <SparkFun_BNO08x_Arduino_Library.h>

#define PIN_IMU_CS   10
#define PIN_IMU_INT  2
#define PIN_IMU_RST  3

#define PORT1_PIDR  (*(volatile const uint16_t *)0x40040026)
static inline bool imu_int_asserted() {
    return (PORT1_PIDR & (1U << 4)) == 0;
}

static BNO08x sensor;

static void measure(const char *name, uint8_t expected_id, uint32_t seconds) {
    while (imu_int_asserted()) { sensor.getSensorEvent(); }
    delay(200);

    for (uint32_t s = 0; s < seconds; s++) {
        uint32_t count = 0, other = 0;
        uint32_t t_min = 999999, t_max = 0, t_sum = 0;
        float last_val = 0.0f;

        uint32_t window_start = micros();
        while (micros() - window_start < 1000000UL) {
            if (!imu_int_asserted()) continue;

            uint32_t t0 = micros();
            if (!sensor.getSensorEvent()) continue;
            uint32_t t1 = micros();
            uint8_t report = sensor.getSensorEventID();

            if (report == expected_id) {
                if (expected_id == SENSOR_REPORTID_GYROSCOPE_CALIBRATED)
                    last_val = sensor.getGyroY();
                else if (expected_id == SENSOR_REPORTID_RAW_GYROSCOPE)
                    last_val = (float)sensor.getRawGyroY();
                else if (expected_id == SENSOR_REPORTID_GYRO_INTEGRATED_ROTATION_VECTOR)
                    last_val = sensor.getGyroIntegratedRVangVelY();

                count++;
                uint32_t dt = t1 - t0;
                t_sum += dt;
                if (dt < t_min) t_min = dt;
                if (dt > t_max) t_max = dt;
            } else {
                other++;
            }
        }

        Serial.print(name);
        Serial.print(": ");
        Serial.print(count);
        Serial.print(" Hz  |  SPI: min=");
        Serial.print(count > 0 ? t_min : 0);
        Serial.print(" avg=");
        Serial.print(count > 0 ? t_sum / count : 0);
        Serial.print(" max=");
        Serial.print(count > 0 ? t_max : 0);
        Serial.print(" us  |  other=");
        Serial.print(other);
        Serial.print("  |  val=");
        Serial.println(last_val, 4);
    }
}

void setup() {
    Serial.begin(1000000);
    while (!Serial && millis() < 3000) {}
    Serial.println("\n=== BNO086 Gyro-Y Speed Test ===\n");

    SPI.begin();
    if (!sensor.beginSPI(PIN_IMU_CS, PIN_IMU_INT, PIN_IMU_RST, 3000000, SPI)) {
        Serial.println("FATAL: BNO086 not detected");
        while (true) { delay(1000); }
    }
    Serial.println("BNO086 init OK");
    delay(100);
    if (sensor.wasReset()) {}
    for (int i = 0; i < 10; i++) { sensor.getSensorEvent(); delay(10); }

    // ── Test 1: Calibrated Gyro @ 1 ms (1 kHz) ────────────────────────────
    Serial.println("\n--- Calibrated Gyro @ 1 ms (1 kHz req) ---");
    sensor.enableGyro(1);
    delay(500);
    measure("CalGyro@1ms", SENSOR_REPORTID_GYROSCOPE_CALIBRATED, 3);

    // Disable by setting very long interval
    sensor.enableGyro(65);  // 65 ms ≈ 15 Hz (effectively off for our purposes)
    delay(200);
    while (imu_int_asserted()) { sensor.getSensorEvent(); }

    // ── Test 2: Raw Gyro @ 1 ms (1 kHz) ───────────────────────────────────
    Serial.println("\n--- Raw Gyro @ 1 ms (1 kHz req) ---");
    sensor.enableRawGyro(1);
    delay(500);
    measure("RawGyro@1ms", SENSOR_REPORTID_RAW_GYROSCOPE, 3);

    sensor.enableRawGyro(65);
    delay(200);
    while (imu_int_asserted()) { sensor.getSensorEvent(); }

    // ── Test 3: GIRV @ 1 ms (1 kHz) ───────────────────────────────────────
    Serial.println("\n--- GIRV @ 1 ms (1 kHz req) ---");
    sensor.enableGyroIntegratedRotationVector(1);
    delay(500);
    measure("GIRV@1ms", SENSOR_REPORTID_GYRO_INTEGRATED_ROTATION_VECTOR, 3);

    sensor.enableGyroIntegratedRotationVector(65);
    delay(200);
    while (imu_int_asserted()) { sensor.getSensorEvent(); }

    // ── Test 4: Calibrated Gyro @ 2 ms (500 Hz) ───────────────────────────
    Serial.println("\n--- Calibrated Gyro @ 2 ms (500 Hz req) ---");
    sensor.enableGyro(2);
    delay(500);
    measure("CalGyro@2ms", SENSOR_REPORTID_GYROSCOPE_CALIBRATED, 3);

    Serial.println("\n=== DONE ===");
}

void loop() { delay(1000); }
