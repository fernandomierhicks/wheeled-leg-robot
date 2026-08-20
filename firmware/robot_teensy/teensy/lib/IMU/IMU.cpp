// IMU.cpp — robust BNO086 SPI driver.
//
// Production consumes one Gyro Integrated Rotation Vector report containing a
// quaternion and angular velocity. Linear acceleration is disabled after a
// short, required wake-helper transaction during configuration.

#include <Arduino.h>
#include <SPI.h>
#include <Adafruit_BNO08x.h>
#include <sh2.h>
#include "config.h"
#include "IMU.h"
#include "comm_protocol.h"

// The dedicated report was hardware-tested at 399.7 Hz with no gaps in a
// 60-second soak, a 150 ms host pause, and a forced reset/recovery cycle.
static constexpr uint32_t IMU_RATE_HZ             = IMU_REQUIRED_RATE_HZ;
static constexpr uint32_t IMU_AUX_RATE_HZ         = 40;
// One staged normal-channel operation is always required: integrated-only
// production disables the temporary wake-helper feature after Gyro RV starts.
static constexpr uint8_t  AUX_REPORT_COUNT        = 1;
static constexpr uint32_t STREAM_TIMEOUT_MS       = 100;
static constexpr uint32_t RECOVERY_GRACE_MS       = 20;
static constexpr uint32_t RESET_NOTICE_TIMEOUT_MS = 500;
static constexpr uint32_t FIRST_DATA_TIMEOUT_MS   = 500;
static constexpr uint32_t RECOVERY_RETRY_MS       = 1000;
static constexpr uint8_t  MAX_INIT_ATTEMPTS       = 2;
static constexpr uint32_t LOSS_WINDOW_MS          = 1000;
static constexpr float    LOSS_THRESHOLD          = 0.10f;
// A normal 500 Hz tick receives less than one report on average. Sixteen lets
// the driver catch up after a delayed tick without allowing unbounded work.
static constexpr uint8_t  MAX_DRAIN_EVENTS        = 16;

enum class RecoveryPhase : uint8_t {
    NONE,
    GRACE,
    WAIT_RESET,
    WAIT_DATA,
    RETRY_DELAY,
};

static Adafruit_BNO08x   _sensor(PIN_IMU_RST);
static sh2_SensorValue_t _sv;

static ImuState _state = ImuState::NOT_READY;
static RecoveryPhase _recovery_phase = RecoveryPhase::NONE;
static bool _session_open = false;
static bool _reports_configured = false;
static uint8_t _aux_reports_configured = 0;
static uint8_t _init_attempts = 0;
static uint32_t _phase_started_ms = 0;
static uint32_t _last_poll_ms = 0;

static float _pitch = 0.0f;
static float _roll = 0.0f;
static float _yaw = 0.0f;
static float _pitch_rate = 0.0f;
static float _roll_rate = 0.0f;
static float _yaw_rate = 0.0f;
static float _accel_x = 0.0f;
static float _accel_y = 0.0f;
static float _accel_z = 0.0f;

static uint32_t _last_grv_ms = 0;
static uint32_t _last_gyro_ms = 0;
static uint32_t _last_accel_ms = 0;
static uint64_t _last_grv_sensor_us = 0;
static uint64_t _last_gyro_sensor_us = 0;
static uint32_t _last_grv_arrival_us = 0;
static uint32_t _last_gyro_arrival_us = 0;

static float _packet_loss = 0.0f;
static uint32_t _loss_window_start = 0;
static uint32_t _win_received = 0;
static uint32_t _win_dropped = 0;
#if IMU_USE_GYRO_INTEGRATED_RV
static uint32_t _last_integrated_gap_count = 0;
#else
static uint8_t _last_grv_seq = 0;
static uint8_t _last_gyro_seq = 0;
static bool _grv_seq_valid = false;
static bool _gyro_seq_valid = false;
#endif

static ImuDiagnostics _diag = {};

static inline bool imu_int_asserted() {
    return digitalRead(PIN_IMU_INT) == LOW;
}

static void reset_loss_window(uint32_t now) {
    _loss_window_start = now;
    _win_received = 0;
    _win_dropped = 0;
}

static void reset_sequence_tracking(uint32_t now) {
#if IMU_USE_GYRO_INTEGRATED_RV
    _last_integrated_gap_count = _sensor.gyroRvSequenceGapCount();
#else
    _grv_seq_valid = false;
    _gyro_seq_valid = false;
#endif
    _packet_loss = 0.0f;
    reset_loss_window(now);
}

static void finish_loss_window(uint32_t now) {
    if (now - _loss_window_start >= LOSS_WINDOW_MS) {
        const uint32_t total = _win_received + _win_dropped;
        _packet_loss = total ? (float)_win_dropped / (float)total : 0.0f;
        reset_loss_window(now);
    }
}

#if IMU_USE_GYRO_INTEGRATED_RV
static void track_integrated_packet() {
    const uint32_t gaps = _sensor.gyroRvSequenceGapCount();
    _win_dropped += gaps - _last_integrated_gap_count;
    _last_integrated_gap_count = gaps;
    _win_received++;
    finish_loss_window(millis());
}
#else
static void track_packet(uint8_t seq, uint8_t *last_seq, bool *valid) {
    if (*valid) {
        // uint8_t subtraction intentionally handles sequence wraparound.
        const uint8_t gap = (uint8_t)(seq - *last_seq - 1);
        // A larger discontinuity is a reset, not meaningful packet loss.
        if (gap < 64) _win_dropped += gap;
    }
    _win_received++;
    *last_seq = seq;
    *valid = true;

    finish_loss_window(millis());
}
#endif

// Intrinsic Z-Y-X (yaw -> pitch -> roll), body to world.
static void quat_to_euler(float qi, float qj, float qk, float qr,
                          float *pitch, float *roll, float *yaw) {
    const float r00 = 1.0f - 2.0f * (qj * qj + qk * qk);
    const float r10 = 2.0f * (qi * qj + qk * qr);
    const float r20 = 2.0f * (qi * qk - qj * qr);
    const float r21 = 2.0f * (qj * qk + qi * qr);
    const float r22 = 1.0f - 2.0f * (qi * qi + qj * qj);

    *pitch = atan2f(-r20, sqrtf(r00 * r00 + r10 * r10));
    *roll = atan2f(r21, r22);
    *yaw = atan2f(r10, r00);
}

static bool valid_and_normalize_quat(float *qi, float *qj, float *qk, float *qr) {
    if (!isfinite(*qi) || !isfinite(*qj) || !isfinite(*qk) || !isfinite(*qr)) return false;
    const float norm_sq = *qi * *qi + *qj * *qj + *qk * *qk + *qr * *qr;
    if (!isfinite(norm_sq) || norm_sq < 0.25f || norm_sq > 2.25f) return false;
    const float inv_norm = 1.0f / sqrtf(norm_sq);
    *qi *= inv_norm;
    *qj *= inv_norm;
    *qk *= inv_norm;
    *qr *= inv_norm;
    return true;
}

static bool finite3(float x, float y, float z) {
    return isfinite(x) && isfinite(y) && isfinite(z);
}

static bool enable_required_reports() {
    constexpr uint32_t interval_us = 1000000UL / IMU_RATE_HZ;

    // Required streams are configured first. Each SPI write is bounded by the
    // vendored transport layer, including during recovery.
#if IMU_USE_GYRO_INTEGRATED_RV
    // A normal-channel feature transaction is required first on this tied-high
    // WAKE board. A direct post-reset Gyro Integrated RV request is acknowledged
    // but produces no data and the hub subsequently resets. Briefly enabling
    // acceleration establishes the proven wake/configuration sequence.
    constexpr uint32_t interval_aux_us = 1000000UL / IMU_AUX_RATE_HZ;
    if (!_sensor.enableReport(SH2_LINEAR_ACCELERATION, interval_aux_us, true)) {
        comm_log(LOG_LEVEL_ERROR, "IMU: failed to enable linear acceleration");
        return false;
    }
    _aux_reports_configured = 1;
#if !IMU_ENABLE_LINEAR_ACCEL
    // Mark the pending staged operation. It cannot be sent immediately after
    // the integrated request because PS0/WAKE is tied high on this board.
    _aux_reports_configured = 0;
#endif
    if (!_sensor.enableReport(SH2_GYRO_INTEGRATED_RV, interval_us, true)) {
        comm_log(LOG_LEVEL_ERROR, "IMU: failed to enable Gyro Integrated RV");
        return false;
    }
#else
    if (!_sensor.enableReport(SH2_GAME_ROTATION_VECTOR, interval_us, true)) {
        comm_log(LOG_LEVEL_ERROR, "IMU: failed to enable Game Rotation Vector");
        return false;
    }
    if (!_sensor.enableReport(SH2_GYROSCOPE_CALIBRATED, interval_us, true)) {
        comm_log(LOG_LEVEL_ERROR, "IMU: failed to enable calibrated gyro");
        return false;
    }
#endif
    return true;
}

static bool enable_next_aux_report() {
    if (_aux_reports_configured == 0) {
#if IMU_ENABLE_LINEAR_ACCEL
        constexpr uint32_t interval_aux_us = 1000000UL / IMU_AUX_RATE_HZ;
        if (_sensor.enableReport(SH2_LINEAR_ACCELERATION, interval_aux_us, true)) {
            _aux_reports_configured++;
            return true;
        }
        comm_log(LOG_LEVEL_WARN, "IMU: failed to enable linear acceleration");
#else
        // A zero interval disables the temporary feature. This is deliberately
        // staged on a sensor-originated interrupt so the command is never sent
        // to the unwakeable sleeping hub.
        if (_sensor.enableReport(SH2_LINEAR_ACCELERATION, 0, false)) {
            _aux_reports_configured++;
            return true;
        }
        comm_log(LOG_LEVEL_WARN, "IMU: failed to disable linear acceleration");
#endif
        return false;
    }
    return true;
}

static void close_session() {
    if (!_session_open) return;
    sh2_close();
    _session_open = false;
    _reports_configured = false;
    _aux_reports_configured = 0;
}

// Boot-only full initialization. begin_SPI includes the slow product-ID
// exchange, so runtime recovery deliberately uses hardwareReset() instead.
static bool attempt_init() {
    if (!_sensor.begin_SPI(PIN_IMU_CS, PIN_IMU_INT)) return false;
    _session_open = true;

    // PS0/WAKE is not routed on this board. Explicitly use reset as a bounded
    // wake operation, service one boot transfer, and configure the first report
    // before the hub can return to sleep. Once GRV is streaming it stays awake.
    _sensor.hardwareReset();
    const uint32_t wake_started = millis();
    while (!imu_int_asserted() && millis() - wake_started <= RESET_NOTICE_TIMEOUT_MS) {}
    if (!imu_int_asserted()) {
        close_session();
        return false;
    }
    sh2_SensorValue_t ignored;
    (void)_sensor.getSensorEvent(&ignored);
    (void)_sensor.wasReset();
    _sensor.clearSensorEvents();

    _aux_reports_configured = 0;
    if (enable_required_reports()) return true;
    close_session();
    return false;
}

static void note_report_gap(uint32_t previous_ms, uint32_t now,
                            uint32_t *maximum_ms) {
    if (!previous_ms) return;
    const uint32_t gap = now - previous_ms;
    if (gap > *maximum_ms) *maximum_ms = gap;
}

static void note_u32_stat(uint32_t value, uint32_t *samples, uint64_t *sum,
                          uint32_t *minimum, uint32_t *maximum) {
    if (*samples == 0 || value < *minimum) *minimum = value;
    if (*samples == 0 || value > *maximum) *maximum = value;
    (*samples)++;
    *sum += value;
}

static void note_stream_timing(uint64_t sensor_us, uint64_t *last_sensor_us,
                               uint32_t *last_arrival_us,
                               ImuStreamTiming *timing) {
    const uint32_t arrival_us = micros();
    if (*last_sensor_us) {
        if (sensor_us > *last_sensor_us) {
            const uint64_t gap64 = sensor_us - *last_sensor_us;
            if (gap64 <= 0xFFFFFFFFULL) {
                note_u32_stat((uint32_t)gap64,
                              &timing->timestamp_gap_samples,
                              &timing->timestamp_gap_sum_us,
                              &timing->timestamp_gap_min_us,
                              &timing->timestamp_gap_max_us);
            }
        } else if (sensor_us == *last_sensor_us) {
            timing->duplicate_timestamps++;
        } else {
            timing->regressed_timestamps++;
        }
    }
    if (*last_arrival_us) {
        note_u32_stat(arrival_us - *last_arrival_us,
                      &timing->arrival_gap_samples,
                      &timing->arrival_gap_sum_us,
                      &timing->arrival_gap_min_us,
                      &timing->arrival_gap_max_us);
    }

    // SH2 timestamps share the HAL micros() timebase. Ignore an implausible
    // subtraction rather than turning a clock-domain/configuration error into
    // a convincing latency statistic.
    const uint32_t delivery_age_us = arrival_us - (uint32_t)sensor_us;
    if (delivery_age_us < 1000000UL) {
        note_u32_stat(delivery_age_us,
                      &timing->delivery_age_samples,
                      &timing->delivery_age_sum_us,
                      &timing->delivery_age_min_us,
                      &timing->delivery_age_max_us);
    }
    *last_sensor_us = sensor_us;
    *last_arrival_us = arrival_us;
}

static void process_event(const sh2_SensorValue_t &value) {
    const uint32_t now = millis();

#if IMU_USE_GYRO_INTEGRATED_RV
    if (value.sensorId == SH2_GYRO_INTEGRATED_RV) {
        float qi = value.un.gyroIntegratedRV.i;
        float qj = value.un.gyroIntegratedRV.j;
        float qk = value.un.gyroIntegratedRV.k;
        float qr = value.un.gyroIntegratedRV.real;
        const float gx = value.un.gyroIntegratedRV.angVelX;
        const float gy = value.un.gyroIntegratedRV.angVelY;
        const float gz = value.un.gyroIntegratedRV.angVelZ;
        if (!valid_and_normalize_quat(&qi, &qj, &qk, &qr) ||
            !finite3(gx, gy, gz)) {
            _diag.invalid_reports++;
            return;
        }
        quat_to_euler(qi, qj, qk, qr, &_pitch, &_roll, &_yaw);
        _roll_rate = gx;
        _pitch_rate = gy;
        _yaw_rate = gz;
        note_report_gap(_last_grv_ms, now, &_diag.max_grv_gap_ms);
        note_report_gap(_last_gyro_ms, now, &_diag.max_gyro_gap_ms);
        note_stream_timing(value.timestamp, &_last_grv_sensor_us,
                           &_last_grv_arrival_us, &_diag.grv_timing);
        note_stream_timing(value.timestamp, &_last_gyro_sensor_us,
                           &_last_gyro_arrival_us, &_diag.gyro_timing);
        _last_grv_ms = now;
        _last_gyro_ms = now;
        _diag.grv_reports++;
        _diag.gyro_reports++;
        track_integrated_packet();
        return;
    }
#else
    if (value.sensorId == SH2_GAME_ROTATION_VECTOR) {
        float qi = value.un.gameRotationVector.i;
        float qj = value.un.gameRotationVector.j;
        float qk = value.un.gameRotationVector.k;
        float qr = value.un.gameRotationVector.real;
        if (!valid_and_normalize_quat(&qi, &qj, &qk, &qr)) {
            _diag.invalid_reports++;
            return;
        }
        quat_to_euler(qi, qj, qk, qr, &_pitch, &_roll, &_yaw);
        note_report_gap(_last_grv_ms, now, &_diag.max_grv_gap_ms);
        note_stream_timing(value.timestamp, &_last_grv_sensor_us,
                           &_last_grv_arrival_us, &_diag.grv_timing);
        _last_grv_ms = now;
        _diag.grv_reports++;
        track_packet(value.sequence, &_last_grv_seq, &_grv_seq_valid);
        return;
    }

    if (value.sensorId == SH2_GYROSCOPE_CALIBRATED) {
        const float x = value.un.gyroscope.x;
        const float y = value.un.gyroscope.y;
        const float z = value.un.gyroscope.z;
        if (!finite3(x, y, z)) {
            _diag.invalid_reports++;
            return;
        }
        _pitch_rate = y;
        _roll_rate = x;
        _yaw_rate = z;
        note_report_gap(_last_gyro_ms, now, &_diag.max_gyro_gap_ms);
        note_stream_timing(value.timestamp, &_last_gyro_sensor_us,
                           &_last_gyro_arrival_us, &_diag.gyro_timing);
        _last_gyro_ms = now;
        _diag.gyro_reports++;
        track_packet(value.sequence, &_last_gyro_seq, &_gyro_seq_valid);
        return;
    }
#endif

    if (value.sensorId == SH2_LINEAR_ACCELERATION) {
        const float x = value.un.linearAcceleration.x;
        const float y = value.un.linearAcceleration.y;
        const float z = value.un.linearAcceleration.z;
        if (!finite3(x, y, z)) {
            _diag.invalid_reports++;
            return;
        }
        _accel_x = x;
        _accel_y = y;
        _accel_z = z;
        _last_accel_ms = now;
        _diag.accel_reports++;
    }
}

static void drain_events() {
    for (uint8_t i = 0; i < MAX_DRAIN_EVENTS; ++i) {
        if (!_sensor.hasQueuedSensorEvent() && !imu_int_asserted()) break;
        if (!_sensor.getSensorEvent(&_sv)) break;
        process_event(_sv);
    }
}

static bool required_streams_fresh(uint32_t now) {
    return _last_grv_ms && _last_gyro_ms &&
           (now - _last_grv_ms <= STREAM_TIMEOUT_MS) &&
           (now - _last_gyro_ms <= STREAM_TIMEOUT_MS);
}

static void clear_required_freshness(uint32_t now) {
    _last_grv_ms = 0;
    _last_gyro_ms = 0;
    _last_grv_sensor_us = 0;
    _last_gyro_sensor_us = 0;
    _last_grv_arrival_us = 0;
    _last_gyro_arrival_us = 0;
    reset_sequence_tracking(now);
}

static void recovery_succeeded(uint32_t now) {
    _recovery_phase = RecoveryPhase::NONE;
    _reports_configured = true;
    _diag.successful_recoveries++;
    reset_sequence_tracking(now);
    _state = ImuState::NOMINAL;
    comm_log(LOG_LEVEL_INFO, "IMU: automatic recovery succeeded");
}

static void begin_runtime_reset(uint32_t now) {
    _state = ImuState::DEGRADED;
    _diag.recovery_attempts++;
    clear_required_freshness(now);
    _aux_reports_configured = 0;
    _sensor.hardwareReset();  // fixed 30 ms pulse sequence; no hidden 500 ms wait
    _phase_started_ms = millis();
    _recovery_phase = RecoveryPhase::WAIT_RESET;
    comm_log(LOG_LEVEL_WARN, "IMU: hardware reset, waiting for reset notice");
}

static void recovery_failed(uint32_t now, const char *reason) {
    _state = ImuState::ERROR;
    _recovery_phase = RecoveryPhase::RETRY_DELAY;
    _phase_started_ms = now;
    comm_log(LOG_LEVEL_ERROR, "IMU: recovery failed (%s); retrying in %lu ms",
             reason, (unsigned long)RECOVERY_RETRY_MS);
}

static void handle_initializing() {
    uint32_t now = millis();
    if (!_reports_configured) {
        if (_init_attempts >= MAX_INIT_ATTEMPTS) {
            _state = ImuState::ERROR;
            _phase_started_ms = millis();
            return;
        }

        _init_attempts++;
        const uint32_t started = millis();
        const bool ok = attempt_init();
        const uint32_t elapsed = millis() - started;
        comm_log(LOG_LEVEL_INFO, "IMU attempt %u/%u: %s (%lu ms)",
                 (unsigned)_init_attempts, (unsigned)MAX_INIT_ATTEMPTS,
                 ok ? "configured" : "failed", (unsigned long)elapsed);
        if (!ok) {
            if (_init_attempts >= MAX_INIT_ATTEMPTS) {
                _state = ImuState::ERROR;
                _phase_started_ms = millis();
            }
            return;
        }

        _reports_configured = true;
        _phase_started_ms = millis();
        clear_required_freshness(_phase_started_ms);
        return;
    }

    // PS0/WAKE is tied high. Use the next sensor-originated interrupt to send
    // the auxiliary configuration while the hub is already awake. The
    // corresponding outbound report is intentionally sacrificed at startup.
    if (_aux_reports_configured < AUX_REPORT_COUNT) {
        if (imu_int_asserted() && !enable_next_aux_report()) {
            close_session();
            if (_init_attempts >= MAX_INIT_ATTEMPTS) _state = ImuState::ERROR;
        } else if (millis() - _phase_started_ms > FIRST_DATA_TIMEOUT_MS) {
            comm_log(LOG_LEVEL_WARN, "IMU: no sensor interrupt for auxiliary setup");
            close_session();
            if (_init_attempts >= MAX_INIT_ATTEMPTS) _state = ImuState::ERROR;
        }
        return;
    }

    drain_events();
    now = millis();
    if (_sensor.wasReset()) {
        _diag.sensor_resets++;
        close_session();
        if (_init_attempts >= MAX_INIT_ATTEMPTS) _state = ImuState::ERROR;
        return;
    }
    if (required_streams_fresh(now)) {
        _state = ImuState::NOMINAL;
        _last_poll_ms = now;
        reset_sequence_tracking(now);
        comm_log(LOG_LEVEL_INFO, "IMU: both required streams are live");
        return;
    }
    if (now - _phase_started_ms > FIRST_DATA_TIMEOUT_MS) {
        comm_log(LOG_LEVEL_WARN, "IMU: configured but first data timed out");
        close_session();
        if (_init_attempts >= MAX_INIT_ATTEMPTS) _state = ImuState::ERROR;
    }
}

void imu_init() {
    close_session();
    _sensor.resetDiagnostics();
    _state = ImuState::INITIALIZING;
    _recovery_phase = RecoveryPhase::NONE;
    _reports_configured = false;
    _aux_reports_configured = 0;
    _init_attempts = 0;
    _phase_started_ms = millis();
    _last_poll_ms = 0;
    _last_grv_ms = 0;
    _last_gyro_ms = 0;
    _last_accel_ms = 0;
    _packet_loss = 0.0f;
    _diag = {};
    reset_sequence_tracking(millis());
}

void imu_update() {
    uint32_t now = millis();

    if (_state == ImuState::NOT_READY) return;
    if (_state == ImuState::INITIALIZING) {
        handle_initializing();
        return;
    }
    // A transient boot failure must not require a power cycle. ERROR already
    // keeps torque disabled; retry the full bounded handshake periodically
    // only when there is no live SH2 session to recover in place.
    if (_state == ImuState::ERROR && !_session_open) {
        if (now - _phase_started_ms >= RECOVERY_RETRY_MS) {
            _state = ImuState::INITIALIZING;
            _init_attempts = 0;
            _reports_configured = false;
            _aux_reports_configured = 0;
            comm_log(LOG_LEVEL_WARN, "IMU: retrying full initialization");
        }
        return;
    }
    if (!_session_open) return;

    if (_last_poll_ms) {
        const uint32_t gap = now - _last_poll_ms;
        if (gap > _diag.max_poll_gap_ms) _diag.max_poll_gap_ms = gap;
    }
    _last_poll_ms = now;

    // During a controlled reset, service one boot transfer and configure the
    // required reports immediately after RESET is observed. Draining the whole
    // boot burst first would let the no-WAKE hub go back to sleep.
    if (_recovery_phase == RecoveryPhase::WAIT_RESET) {
        if (imu_int_asserted()) {
            if (_sensor.getSensorEvent(&_sv)) process_event(_sv);
            // This is the first transaction after a reset we requested. Issue
            // the required configuration immediately, regardless of which
            // boot record happened to be first in that transfer. Waiting until
            // RESET is decoded can consume the entire wake window.
            (void)_sensor.wasReset();
            _diag.sensor_resets++;
            _sensor.clearSensorEvents();
            clear_required_freshness(now);
            _aux_reports_configured = 0;
            if (!enable_required_reports()) {
                recovery_failed(millis(), "required report configuration");
                return;
            }
            _reports_configured = true;
            _recovery_phase = RecoveryPhase::WAIT_DATA;
            _phase_started_ms = millis();
            return;
        }
        if (now - _phase_started_ms > RESET_NOTICE_TIMEOUT_MS)
            recovery_failed(now, "no reset notice");
        return;
    }

    // Add auxiliary reports only on sensor-originated interrupts, when the
    // tied-high WAKE wiring cannot leave a host command waiting on a sleeping
    // hub. Required streams remain the only condition for NOMINAL.
    if (_recovery_phase == RecoveryPhase::WAIT_DATA &&
        _aux_reports_configured < AUX_REPORT_COUNT) {
        if (imu_int_asserted() && !enable_next_aux_report()) {
            recovery_failed(millis(), "auxiliary report configuration");
            return;
        }
        if (now - _phase_started_ms > FIRST_DATA_TIMEOUT_MS)
            recovery_failed(now, "auxiliary report wake timeout");
        return;
    }

    // Service transport first. After a long scheduler pause this drains the
    // sensor's retained packet before deciding that a stream is stale.
    drain_events();
    now = millis();

    const bool saw_reset = _sensor.wasReset();
    if (saw_reset) {
        _diag.sensor_resets++;
        _state = ImuState::DEGRADED;
        _sensor.clearSensorEvents();
        clear_required_freshness(now);
        // Let safety logic observe DEGRADED this tick. GRACE is pre-expired so
        // the next tick starts a controlled reset-as-wake handshake.
        _recovery_phase = RecoveryPhase::GRACE;
        _phase_started_ms = now - RECOVERY_GRACE_MS;
        comm_log(LOG_LEVEL_WARN, "IMU: sensor reset detected; reconfiguring reports");
        return;
    }

    switch (_recovery_phase) {
    case RecoveryPhase::GRACE:
        if (required_streams_fresh(now)) {
            recovery_succeeded(now);
        } else if (now - _phase_started_ms >= RECOVERY_GRACE_MS) {
            begin_runtime_reset(now);
        }
        return;

    case RecoveryPhase::WAIT_RESET:
        return;  // handled before normal draining above

    case RecoveryPhase::WAIT_DATA:
        if (required_streams_fresh(now)) {
            recovery_succeeded(now);
        } else if (now - _phase_started_ms > FIRST_DATA_TIMEOUT_MS) {
            recovery_failed(now, "no data after reset");
        }
        return;

    case RecoveryPhase::RETRY_DELAY:
        if (now - _phase_started_ms >= RECOVERY_RETRY_MS)
            begin_runtime_reset(now);
        return;

    case RecoveryPhase::NONE:
        break;
    }

    const bool grv_stale = !_last_grv_ms || now - _last_grv_ms > STREAM_TIMEOUT_MS;
    const bool gyro_stale = !_last_gyro_ms || now - _last_gyro_ms > STREAM_TIMEOUT_MS;
    if (grv_stale || gyro_stale) {
        _diag.stream_stalls++;
        _state = ImuState::DEGRADED;
        _recovery_phase = RecoveryPhase::GRACE;
        _phase_started_ms = now;
        comm_log(LOG_LEVEL_WARN, "IMU stream stalled: GRV=%s gyro=%s; recovering",
                 grv_stale ? "STALE" : "ok", gyro_stale ? "STALE" : "ok");
        return;
    }

    _state = (_packet_loss >= LOSS_THRESHOLD) ? ImuState::DEGRADED
                                               : ImuState::NOMINAL;
}

ImuState imu_state() { return _state; }
float imu_pitch() { return _pitch; }
float imu_roll() { return _roll; }
float imu_yaw() { return _yaw; }
float imu_pitch_rate() { return _pitch_rate; }
float imu_roll_rate() { return _roll_rate; }
float imu_yaw_rate() { return _yaw_rate; }
float imu_accel_x() { return _accel_x; }
float imu_accel_y() { return _accel_y; }
float imu_accel_z() { return _accel_z; }
uint32_t imu_accel_last_update_ms() { return _last_accel_ms; }
uint32_t imu_gyro_last_update_ms() { return _last_gyro_ms; }
float imu_packet_loss() { return _packet_loss; }

uint32_t imu_last_update_ms() {
    if (!_last_grv_ms || !_last_gyro_ms) return 0;
    const uint32_t now = millis();
    return (now - _last_grv_ms >= now - _last_gyro_ms) ? _last_grv_ms
                                                        : _last_gyro_ms;
}

void imu_get_diagnostics(ImuDiagnostics *out) {
    if (!out) return;
    *out = _diag;
    out->queue_overflows = _sensor.sensorEventOverflowCount();
    out->decode_errors = _sensor.sensorDecodeErrorCount();
    out->transport_errors = _sensor.transportErrorCount();
    out->gyro_rv_shtp_sequence_gaps = _sensor.gyroRvSequenceGapCount();
    const uint32_t now = millis();
    out->grv_age_ms = _last_grv_ms ? now - _last_grv_ms : 0xFFFFFFFFUL;
    out->gyro_age_ms = _last_gyro_ms ? now - _last_gyro_ms : 0xFFFFFFFFUL;
}
