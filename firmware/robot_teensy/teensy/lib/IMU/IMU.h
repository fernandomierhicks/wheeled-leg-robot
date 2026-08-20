#pragma once
// IMU.h — robust BNO086 SPI driver (integrated quaternion + gyro)
//
// Call imu_init() once at startup, then imu_update() every control tick.
// Motion must not be enabled until imu_state() == ImuState::NOMINAL.

#include <stdint.h>

// Production configuration. Test environments may override these at compile
// time to retain A/B coverage of the legacy multi-report architecture.
#ifndef IMU_USE_GYRO_INTEGRATED_RV
#define IMU_USE_GYRO_INTEGRATED_RV 1
#endif
#ifndef IMU_REQUIRED_RATE_HZ
#define IMU_REQUIRED_RATE_HZ 400
#endif
#ifndef IMU_ENABLE_LINEAR_ACCEL
#define IMU_ENABLE_LINEAR_ACCEL 0
#endif

enum class ImuState : uint8_t {
    NOT_READY    = 0,  // imu_init() not yet called
    INITIALIZING = 1,  // begin_SPI() in progress / waiting for retry
    NOMINAL      = 2,  // both required streams fresh, packet loss < 10%
    DEGRADED     = 3,  // unsafe: stale/lossy data or recovery in progress
    ERROR        = 4,  // init failed, or automatic recovery is waiting to retry
};

// Timing statistics are based on the SH2 event timestamp and the Teensy time
// at which the decoded report reaches IMU.cpp. Gyro Integrated RV reports use
// their dedicated SHTP channel's transfer timestamp (not an acquisition
// timestamp), which is why duplicate timestamps are tracked explicitly.
struct ImuStreamTiming {
    uint32_t timestamp_gap_samples;
    uint64_t timestamp_gap_sum_us;
    uint32_t timestamp_gap_min_us;
    uint32_t timestamp_gap_max_us;
    uint32_t duplicate_timestamps;
    uint32_t regressed_timestamps;
    uint32_t arrival_gap_samples;
    uint64_t arrival_gap_sum_us;
    uint32_t arrival_gap_min_us;
    uint32_t arrival_gap_max_us;
    uint32_t delivery_age_samples;
    uint64_t delivery_age_sum_us;
    uint32_t delivery_age_min_us;
    uint32_t delivery_age_max_us;
};

// Cumulative counters since imu_init(). These make intermittent transport and
// scheduling failures observable without putting Serial prints in the 500 Hz
// path. Ages are UINT32_MAX until the corresponding stream has produced data.
struct ImuDiagnostics {
    uint32_t grv_reports;
    uint32_t gyro_reports;
    uint32_t accel_reports;
    uint32_t invalid_reports;
    uint32_t stream_stalls;
    uint32_t sensor_resets;
    uint32_t recovery_attempts;
    uint32_t successful_recoveries;
    uint32_t queue_overflows;
    uint32_t decode_errors;
    uint32_t transport_errors;
    uint32_t gyro_rv_shtp_sequence_gaps;
    uint32_t max_poll_gap_ms;
    uint32_t max_grv_gap_ms;
    uint32_t max_gyro_gap_ms;
    uint32_t grv_age_ms;
    uint32_t gyro_age_ms;
    ImuStreamTiming grv_timing;
    ImuStreamTiming gyro_timing;
};

// Call once; sets state to INITIALIZING. Actual init runs inside imu_update().
void imu_init();

// Non-blocking poll — call every control tick. Drives state machine + data.
void imu_update();

ImuState imu_state();
// Gyro Integrated Rotation Vector — no magnetometer
float    imu_pitch();           // rad  — positive = lean forward (+X down)
float    imu_roll();            // rad  — positive = lean right
float    imu_yaw();             // rad
// Angular velocity paired with the same integrated report
float    imu_pitch_rate();      // rad/s (gyro Y)
float    imu_roll_rate();       // rad/s (gyro X)
float    imu_yaw_rate();        // rad/s (gyro Z)
float    imu_accel_x();         // linear accel X (forward)  [m/s²] — BNO086 SH2_LINEAR_ACCELERATION
float    imu_accel_y();         // linear accel Y (left)     [m/s²]
float    imu_accel_z();         // linear accel Z (up)       [m/s²]
uint32_t imu_accel_last_update_ms(); // timestamp of most recent linear-accel report
uint32_t imu_gyro_last_update_ms();  // timestamp of most recent angular-velocity report
float    imu_packet_loss();     // 0.0–1.0, rolling 1-second window
// millis() of the older required attitude/rate sample. They are paired in the
// production integrated report; legacy A/B builds still receive them separately.
uint32_t imu_last_update_ms();

void imu_get_diagnostics(ImuDiagnostics *out);
