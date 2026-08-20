// BNO086 robustness test. This intentionally polls at the production 500 Hz
// cadence, pauses long enough to reproduce the SD-start scheduler stall, then
// forces a real sensor reset and verifies automatic recovery.

#include <Arduino.h>
#include <stdarg.h>
#include "IMU.h"
#include "config.h"
#include "../test_led.h"

#ifndef IMU_TEST_SOAK_MS
#define IMU_TEST_SOAK_MS 10000
#endif

struct RunningStats {
    uint32_t count = 0;
    double mean = 0.0;
    double m2 = 0.0;
    float minimum = INFINITY;
    float maximum = -INFINITY;

    void add(float value) {
        count++;
        const double delta = value - mean;
        mean += delta / count;
        m2 += delta * (value - mean);
        if (value < minimum) minimum = value;
        if (value > maximum) maximum = value;
    }

    float stddev() const {
        return count > 1 ? sqrtf((float)(m2 / (count - 1))) : 0.0f;
    }

    float peak_to_peak() const {
        return count ? maximum - minimum : 0.0f;
    }
};

struct LowPassProbe {
    float cutoff_hz = 0.0f;
    float alpha = 0.0f;
    bool initialized = false;
    float state[3] = {};
    RunningStats stats[3];

    void begin(float cutoff, float sample_hz) {
        cutoff_hz = cutoff;
        alpha = 1.0f - expf(-2.0f * PI * cutoff / sample_hz);
    }

    void add(float x, float y, float z) {
        const float input[3] = {x, y, z};
        if (!initialized) {
            for (uint8_t axis = 0; axis < 3; ++axis) state[axis] = input[axis];
            initialized = true;
        } else {
            for (uint8_t axis = 0; axis < 3; ++axis)
                state[axis] += alpha * (input[axis] - state[axis]);
        }
        for (uint8_t axis = 0; axis < 3; ++axis) stats[axis].add(state[axis]);
    }

    float low_frequency_delay_ms(float sample_hz) const {
        return (1.0f - alpha) / alpha * 1000.0f / sample_hz;
    }
};

// Production supplies this from main.cpp, which is intentionally excluded
// from test builds. The test prints its own detailed progress over Serial.
void comm_log(uint8_t level, const char *fmt, ...) {
    char message[256];
    va_list args;
    va_start(args, fmt);
    vsnprintf(message, sizeof(message), fmt, args);
    va_end(args);
    Serial.printf("[DRIVER L%u] %s\n", (unsigned)level, message);
}

static uint8_t g_failures = 0;

static const char *state_name(ImuState s) {
    switch (s) {
        case ImuState::NOT_READY: return "NOT_READY";
        case ImuState::INITIALIZING: return "INITIALIZING";
        case ImuState::NOMINAL: return "NOMINAL";
        case ImuState::DEGRADED: return "DEGRADED";
        case ImuState::ERROR: return "ERROR";
        default: return "UNKNOWN";
    }
}

static void check(bool condition, const char *label) {
    Serial.printf("[%s] %s\n", condition ? "PASS" : "FAIL", label);
    if (!condition) g_failures++;
}

static void poll_for(uint32_t duration_ms, uint32_t period_us = 2000) {
    const uint32_t started = millis();
    uint32_t next_us = micros();
    while (millis() - started < duration_ms) {
        imu_update();
        next_us += period_us;
        const int32_t remaining = (int32_t)(next_us - micros());
        if (remaining > 0) delayMicroseconds((uint32_t)remaining);
        else next_us = micros();
    }
}

static bool wait_for_nominal(uint32_t timeout_ms) {
    const uint32_t started = millis();
    while (millis() - started < timeout_ms) {
        imu_update();
        if (imu_state() == ImuState::NOMINAL) return true;
        delayMicroseconds(2000);
    }
    return false;
}

static bool wait_for_reset_recovery(uint32_t previous_reset_count,
                                    uint32_t timeout_ms) {
    const uint32_t started = millis();
    bool recovery_observed = false;
    while (millis() - started < timeout_ms) {
        imu_update();
        ImuDiagnostics d;
        imu_get_diagnostics(&d);
        if (imu_state() != ImuState::NOMINAL ||
            d.sensor_resets > previous_reset_count) {
            recovery_observed = true;
        }
        if (recovery_observed && imu_state() == ImuState::NOMINAL &&
            d.sensor_resets > previous_reset_count &&
            millis() - imu_last_update_ms() < 50) {
            return true;
        }
        delayMicroseconds(2000);
    }
    return false;
}

static void print_diagnostics(const char *label) {
    ImuDiagnostics d;
    imu_get_diagnostics(&d);
    Serial.printf(
        "[DIAG] %s state=%s grv=%lu gyro=%lu accel=%lu invalid=%lu\n",
        label, state_name(imu_state()),
        (unsigned long)d.grv_reports, (unsigned long)d.gyro_reports,
        (unsigned long)d.accel_reports, (unsigned long)d.invalid_reports);
    Serial.printf(
        "[DIAG] stalls=%lu resets=%lu attempts=%lu recovered=%lu q_overflow=%lu "
        "decode=%lu transport=%lu gyro_rv_shtp_gaps=%lu\n",
        (unsigned long)d.stream_stalls,
        (unsigned long)d.sensor_resets, (unsigned long)d.recovery_attempts,
        (unsigned long)d.successful_recoveries, (unsigned long)d.queue_overflows,
        (unsigned long)d.decode_errors, (unsigned long)d.transport_errors,
        (unsigned long)d.gyro_rv_shtp_sequence_gaps);
    Serial.printf(
        "[DIAG] max_poll=%lums max_grv=%lums max_gyro=%lums ages=%lu/%lums loss=%.2f%%\n",
        (unsigned long)d.max_poll_gap_ms, (unsigned long)d.max_grv_gap_ms,
        (unsigned long)d.max_gyro_gap_ms, (unsigned long)d.grv_age_ms,
        (unsigned long)d.gyro_age_ms, imu_packet_loss() * 100.0f);
}

static float timing_average(uint64_t sum, uint32_t samples) {
    return samples ? (float)sum / samples : 0.0f;
}

static void print_stream_timing(const char *name, const ImuStreamTiming &t) {
    Serial.printf(
        "[TIMING] %s sensor_gap avg=%.1fus min=%lu max=%lu n=%lu dup=%lu regress=%lu\n",
        name, timing_average(t.timestamp_gap_sum_us, t.timestamp_gap_samples),
        (unsigned long)t.timestamp_gap_min_us,
        (unsigned long)t.timestamp_gap_max_us,
        (unsigned long)t.timestamp_gap_samples,
        (unsigned long)t.duplicate_timestamps,
        (unsigned long)t.regressed_timestamps);
    Serial.printf(
        "[TIMING] %s arrival_gap avg=%.1fus min=%lu max=%lu; delivery_age avg=%.1fus min=%lu max=%lu\n",
        name, timing_average(t.arrival_gap_sum_us, t.arrival_gap_samples),
        (unsigned long)t.arrival_gap_min_us,
        (unsigned long)t.arrival_gap_max_us,
        timing_average(t.delivery_age_sum_us, t.delivery_age_samples),
        (unsigned long)t.delivery_age_min_us,
        (unsigned long)t.delivery_age_max_us);
}

static void force_external_sensor_reset() {
    pinMode(PIN_IMU_RST, OUTPUT);
    digitalWrite(PIN_IMU_RST, HIGH);
    delay(10);
    digitalWrite(PIN_IMU_RST, LOW);
    delay(10);
    digitalWrite(PIN_IMU_RST, HIGH);
    delay(10);
}

void setup() {
    Serial.begin(115200);
    // GUI/PlatformIO flashing returns before Windows has always reopened the
    // USB CDC endpoint. Keep the banner and one-shot assertions observable.
    while (!Serial && millis() < 10000) {}
    Serial.printf("\n=== IMU robustness test (%s @ %lu Hz request, accel %s) ===\n",
                  IMU_USE_GYRO_INTEGRATED_RV ? "Gyro Integrated RV"
                                             : "GRV + calibrated gyro",
                  (unsigned long)IMU_REQUIRED_RATE_HZ,
                  IMU_ENABLE_LINEAR_ACCEL ? "enabled" : "disabled");
    test_led_begin();

    imu_init();
    check(wait_for_nominal(5000), "initializes with both required streams live");
    print_diagnostics("after init");

    Serial.println("[TEST] 3 s report-rate measurement at production 500 Hz polling...");
    ImuDiagnostics before;
    ImuDiagnostics after;
    imu_get_diagnostics(&before);
    const uint32_t rate_started = millis();
    RunningStats pitch_noise;
    RunningStats gx_noise;
    RunningStats gy_noise;
    RunningStats gz_noise;
    LowPassProbe filtered_10;
    LowPassProbe filtered_20;
    LowPassProbe filtered_40;
    LowPassProbe filtered_60;
    LowPassProbe filtered_80;
    filtered_10.begin(10.0f, (float)IMU_REQUIRED_RATE_HZ);
    filtered_20.begin(20.0f, (float)IMU_REQUIRED_RATE_HZ);
    filtered_40.begin(40.0f, (float)IMU_REQUIRED_RATE_HZ);
    filtered_60.begin(60.0f, (float)IMU_REQUIRED_RATE_HZ);
    filtered_80.begin(80.0f, (float)IMU_REQUIRED_RATE_HZ);
    uint32_t seen_grv = before.grv_reports;
    uint32_t seen_gyro = before.gyro_reports;
    uint32_t next_us = micros();
    while (millis() - rate_started < 3000) {
        imu_update();
        ImuDiagnostics live;
        imu_get_diagnostics(&live);
        if (live.grv_reports != seen_grv) {
            pitch_noise.add(imu_pitch());
            seen_grv = live.grv_reports;
        }
        if (live.gyro_reports != seen_gyro) {
            const float gx = imu_roll_rate();
            const float gy = imu_pitch_rate();
            const float gz = imu_yaw_rate();
            gx_noise.add(gx);
            gy_noise.add(gy);
            gz_noise.add(gz);
            filtered_10.add(gx, gy, gz);
            filtered_20.add(gx, gy, gz);
            filtered_40.add(gx, gy, gz);
            filtered_60.add(gx, gy, gz);
            filtered_80.add(gx, gy, gz);
            seen_gyro = live.gyro_reports;
        }
        next_us += 2000;
        const int32_t remaining = (int32_t)(next_us - micros());
        if (remaining > 0) delayMicroseconds((uint32_t)remaining);
        else next_us = micros();
    }
    const uint32_t rate_elapsed = millis() - rate_started;
    imu_get_diagnostics(&after);
    const float grv_hz = (after.grv_reports - before.grv_reports) * 1000.0f / rate_elapsed;
    const float gyro_hz = (after.gyro_reports - before.gyro_reports) * 1000.0f / rate_elapsed;
    const float accel_hz = (after.accel_reports - before.accel_reports) * 1000.0f / rate_elapsed;
    Serial.printf("[RATE] attitude=%.1f Hz angular_velocity=%.1f Hz accel=%.1f Hz\n",
                  grv_hz, gyro_hz, accel_hz);
#if IMU_USE_GYRO_INTEGRATED_RV
    const float integrated_min_hz = IMU_REQUIRED_RATE_HZ * 0.85f;
    const float integrated_max_hz = IMU_REQUIRED_RATE_HZ * 1.15f;
    check(grv_hz >= integrated_min_hz && grv_hz <= integrated_max_hz,
          "integrated attitude rate tracks the requested rate");
    check(gyro_hz >= integrated_min_hz && gyro_hz <= integrated_max_hz,
          "integrated angular-velocity rate tracks the requested rate");
#else
    // BNO086 report intervals are requests, not exact schedules. With all three
    // production reports enabled, a 160 Hz required-stream request measures in
    // this band on the physical robot. Assert the measured operating envelope.
    check(grv_hz >= 170.0f && grv_hz <= 200.0f, "GRV rate is in the measured operating band");
    check(gyro_hz >= 170.0f && gyro_hz <= 200.0f, "gyro rate is in the measured operating band");
#endif
#if IMU_ENABLE_LINEAR_ACCEL
    check(accel_hz >= 7.0f && accel_hz <= 16.0f, "linear acceleration rate is stable near 10 Hz");
#else
    check(after.accel_reports == before.accel_reports,
          "linear acceleration is completely disabled");
#endif
    check(imu_packet_loss() < 0.02f, "required-stream packet loss below 2%");
    Serial.printf(
        "[NOISE] pitch std=%.6f rad p-p=%.6f; gyro xyz std=%.6f/%.6f/%.6f rad/s p-p=%.6f/%.6f/%.6f\n",
        pitch_noise.stddev(), pitch_noise.peak_to_peak(),
        gx_noise.stddev(), gy_noise.stddev(), gz_noise.stddev(),
        gx_noise.peak_to_peak(), gy_noise.peak_to_peak(), gz_noise.peak_to_peak());
    Serial.printf(
        "[FILTER] 10Hz xyz std=%.6f/%.6f/%.6f rad/s estimated_low_freq_delay=%.2fms\n",
        filtered_10.stats[0].stddev(), filtered_10.stats[1].stddev(),
        filtered_10.stats[2].stddev(),
        filtered_10.low_frequency_delay_ms((float)IMU_REQUIRED_RATE_HZ));
    Serial.printf(
        "[FILTER] 20Hz xyz std=%.6f/%.6f/%.6f rad/s estimated_low_freq_delay=%.2fms\n",
        filtered_20.stats[0].stddev(), filtered_20.stats[1].stddev(),
        filtered_20.stats[2].stddev(),
        filtered_20.low_frequency_delay_ms((float)IMU_REQUIRED_RATE_HZ));
    Serial.printf(
        "[FILTER] 40Hz xyz std=%.6f/%.6f/%.6f rad/s estimated_low_freq_delay=%.2fms\n",
        filtered_40.stats[0].stddev(), filtered_40.stats[1].stddev(),
        filtered_40.stats[2].stddev(),
        filtered_40.low_frequency_delay_ms((float)IMU_REQUIRED_RATE_HZ));
    Serial.printf(
        "[FILTER] 60Hz xyz std=%.6f/%.6f/%.6f rad/s estimated_low_freq_delay=%.2fms\n",
        filtered_60.stats[0].stddev(), filtered_60.stats[1].stddev(),
        filtered_60.stats[2].stddev(),
        filtered_60.low_frequency_delay_ms((float)IMU_REQUIRED_RATE_HZ));
    Serial.printf(
        "[FILTER] 80Hz xyz std=%.6f/%.6f/%.6f rad/s estimated_low_freq_delay=%.2fms\n",
        filtered_80.stats[0].stddev(), filtered_80.stats[1].stddev(),
        filtered_80.stats[2].stddev(),
        filtered_80.low_frequency_delay_ms((float)IMU_REQUIRED_RATE_HZ));
    print_stream_timing("attitude", after.grv_timing);
    print_stream_timing("angular_velocity", after.gyro_timing);
    check(after.stream_stalls == before.stream_stalls &&
              after.sensor_resets == before.sensor_resets,
          "rate measurement has no spontaneous stall or reset");

    Serial.println("[TEST] pausing all polls for 150 ms (measured SD-start stall model)...");
    delay(150);
    check(wait_for_nominal(2000), "150 ms scheduler stall recovers without reboot");
    check(millis() - imu_last_update_ms() < 50, "both required streams are fresh after stall");
    print_diagnostics("after 150 ms stall");

    Serial.println("[TEST] forcing an unexpected BNO086 hardware reset...");
    imu_get_diagnostics(&before);
    force_external_sensor_reset();
    check(wait_for_reset_recovery(before.sensor_resets, 3000),
          "unexpected sensor reset re-enables reports automatically");
    imu_get_diagnostics(&after);
    check(after.sensor_resets > before.sensor_resets, "reset notification was observed");
    check(after.successful_recoveries > before.successful_recoveries,
          "reset recovery reached fresh dual-stream data");
    print_diagnostics("after forced reset");

    Serial.printf("[TEST] %.1f s production-cadence soak...\n",
                  IMU_TEST_SOAK_MS / 1000.0f);
    ImuDiagnostics soak_before;
    imu_get_diagnostics(&soak_before);
    poll_for(IMU_TEST_SOAK_MS);
    imu_get_diagnostics(&after);
    check(imu_state() == ImuState::NOMINAL, "remains NOMINAL through soak");
    check(after.queue_overflows == 0, "sensor-event queue has no overflows");
    check(after.decode_errors == 0, "sensor reports have no decode errors");
    check(after.transport_errors == 0, "SPI transport has no errors");
    check(after.gyro_rv_shtp_sequence_gaps == 0,
          "dedicated Gyro RV SHTP channel has no sequence gaps");
    check(after.invalid_reports == 0, "all consumed sensor values are valid");
    check(millis() - imu_last_update_ms() < 50, "both required streams remain fresh");
    check(after.stream_stalls == soak_before.stream_stalls &&
              after.sensor_resets == soak_before.sensor_resets &&
              after.recovery_attempts == soak_before.recovery_attempts,
          "soak has no spontaneous stall, reset, or recovery");
    print_diagnostics("final");

    Serial.printf("\n=== IMU robustness test done: %u failure(s) ===\n\n", g_failures);
    test_led_done(g_failures);
}

void loop() {
    imu_update();
    static uint32_t last_print = 0;
    if (millis() - last_print >= 1000) {
        last_print = millis();
        print_diagnostics("live");
    }
    delayMicroseconds(2000);
}
