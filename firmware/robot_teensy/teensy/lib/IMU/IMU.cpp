// IMU.cpp — BNO086 SPI driver using Adafruit_BNO08x library.
//
// Uses Game Rotation Vector (no magnetometer — immune to motor field disturbance)
// at 400 Hz. Tracks packet loss via SH2 per-report sequence numbers.
// Pins: CS=10, INT=2, RST=3 (SPI0). PS0+PS1 bridged for SPI mode.

#include <Arduino.h>
#include <SPI.h>
#include <Adafruit_BNO08x.h>
#include "config.h"
#include "IMU.h"

// ── Config ───────────────────────────────────────────────────────────────────
static constexpr uint32_t IMU_RATE_HZ       = 400;
static constexpr uint32_t TIMEOUT_MS        = 100;   // silence before ERROR (measured from last poll, not wall clock)
static constexpr uint32_t RETRY_INTERVAL_MS = 1000;
static constexpr uint32_t LOSS_WINDOW_MS    = 1000;
static constexpr float    LOSS_THRESHOLD    = 0.10f;
static constexpr uint8_t  MAX_DRAIN_EVENTS  = 6;     // GRV + RV + Gyro per cycle

// ── Module state ─────────────────────────────────────────────────────────────
static Adafruit_BNO08x   _sensor(PIN_IMU_RST);
static sh2_SensorValue_t _sv;

static ImuState  _state          = ImuState::NOT_READY;
static float     _pitch          = 0.0f;
static float     _roll           = 0.0f;
static float     _yaw            = 0.0f;
static float     _pitch_mag      = 0.0f;
static float     _roll_mag       = 0.0f;
static float     _yaw_mag        = 0.0f;
static float     _pitch_rate     = 0.0f;
static float     _roll_rate      = 0.0f;
static float     _yaw_rate       = 0.0f;
static float     _packet_loss    = 0.0f;
static uint32_t  _last_update_ms = 0;
static uint32_t  _last_poll_ms   = 0;
static uint32_t  _retry_due_ms   = 0;

// Packet loss tracking
static uint32_t  _loss_window_start = 0;
static uint16_t  _win_received      = 0;
static uint16_t  _win_dropped       = 0;
static uint8_t   _last_rv_seq       = 0;
static bool      _rv_seq_valid      = false;

// ── Helpers ───────────────────────────────────────────────────────────────────
static inline bool imu_int_asserted() {
    return digitalRead(PIN_IMU_INT) == LOW;
}

static float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

static void quat_to_euler(float qi, float qj, float qk, float qr,
                           float *pitch, float *roll, float *yaw) {
    // Pitch about Y — positive = lean forward (+X down)
    float sinp = 2.0f * (qr * qj - qi * qk);
    float cosp = 1.0f - 2.0f * (qi * qi + qj * qj);
    *pitch = atan2f(sinp, cosp);

    // Roll about X — positive = lean right
    float sinr = 2.0f * (qr * qi + qj * qk);
    *roll = asinf(clampf(sinr, -1.0f, 1.0f));

    // Yaw about Z
    float siny = 2.0f * (qr * qk + qi * qj);
    float cosy = 1.0f - 2.0f * (qj * qj + qk * qk);
    *yaw = atan2f(siny, cosy);
}

static bool enable_reports() {
    constexpr uint32_t interval_us = 1000000UL / IMU_RATE_HZ;
    if (!_sensor.enableReport(SH2_GAME_ROTATION_VECTOR, interval_us)) {
        Serial.println("[IMU] Failed to enable Game Rotation Vector");
        return false;
    }
    if (!_sensor.enableReport(SH2_ROTATION_VECTOR, interval_us)) {
        Serial.println("[IMU] Failed to enable Rotation Vector (mag)");
        return false;
    }
    if (!_sensor.enableReport(SH2_GYROSCOPE_CALIBRATED, interval_us)) {
        Serial.println("[IMU] Failed to enable Gyro");
        return false;
    }
    return true;
}

// Blocks ~100 ms on first connection while BNO086 completes its reset handshake.
// On retries from ERROR state this is acceptable since motion is gated on NOMINAL.
static bool attempt_init() {
    if (!_sensor.begin_SPI(PIN_IMU_CS, PIN_IMU_INT)) {
        return false;
    }
    delay(100);
    if (_sensor.wasReset()) {
        // Dummy enable to acknowledge the reset before real report setup
        _sensor.enableReport(SH2_GAME_ROTATION_VECTOR, 100000);
    }
    // Drain stale post-reset events (no delay needed between reads)
    sh2_SensorValue_t tmp;
    for (int i = 0; i < 10; i++) {
        _sensor.getSensorEvent(&tmp);
    }
    return enable_reports();
}

static void reset_loss_window(uint32_t now) {
    _loss_window_start = now;
    _win_received      = 0;
    _win_dropped       = 0;
}

// Called on each received GRV packet. Uses SH2 sequence number to count drops.
static void track_packet(uint8_t seq) {
    if (_rv_seq_valid) {
        // uint8_t subtraction wraps correctly for sequence arithmetic
        uint8_t gap = (uint8_t)(seq - _last_rv_seq - 1);
        // Ignore implausible gaps (sensor reset or first-run anomaly)
        if (gap < 64) {
            _win_dropped += gap;
        }
    }
    _win_received++;
    _last_rv_seq   = seq;
    _rv_seq_valid  = true;

    uint32_t now = millis();
    if (now - _loss_window_start >= LOSS_WINDOW_MS) {
        uint16_t total = _win_received + _win_dropped;
        _packet_loss   = (total > 0) ? (float)_win_dropped / total : 0.0f;
        reset_loss_window(now);
    }
}

// ── Public API ────────────────────────────────────────────────────────────────
void imu_init() {
    _state             = ImuState::INITIALIZING;
    _retry_due_ms      = 0;
    _rv_seq_valid      = false;
    _packet_loss       = 0.0f;
    reset_loss_window(millis());
}

void imu_update() {
    uint32_t now = millis();

    switch (_state) {

    case ImuState::NOT_READY:
        return;

    case ImuState::INITIALIZING:
    case ImuState::ERROR:
        if (now < _retry_due_ms) return;
        if (attempt_init()) {
            _state          = ImuState::NOMINAL;
            _last_update_ms = millis();
            _last_poll_ms   = _last_update_ms;
            _rv_seq_valid   = false;
            _packet_loss    = 0.0f;
            reset_loss_window(_last_update_ms);
        } else {
            _state        = ImuState::ERROR;
            _retry_due_ms = now + RETRY_INTERVAL_MS;
        }
        return;

    case ImuState::NOMINAL:
    case ImuState::DEGRADED:
        _last_poll_ms = now;
        break;
    }

    // Recover from unexpected sensor reset (re-enable reports, reset seq tracking)
    if (_sensor.wasReset()) {
        enable_reports();
        _rv_seq_valid = false;
    }

    // Drain queued events — GRV + Gyro arrive together, cap to avoid blocking loop
    for (uint8_t i = 0; i < MAX_DRAIN_EVENTS; i++) {
        if (!imu_int_asserted()) break;
        if (!_sensor.getSensorEvent(&_sv)) break;

        if (_sv.sensorId == SH2_GAME_ROTATION_VECTOR) {
            float qi = _sv.un.gameRotationVector.i;
            float qj = _sv.un.gameRotationVector.j;
            float qk = _sv.un.gameRotationVector.k;
            float qr = _sv.un.gameRotationVector.real;
            quat_to_euler(qi, qj, qk, qr, &_pitch, &_roll, &_yaw);
            _last_update_ms = now;
            track_packet(_sv.sequence);

        } else if (_sv.sensorId == SH2_ROTATION_VECTOR) {
            float qi = _sv.un.rotationVector.i;
            float qj = _sv.un.rotationVector.j;
            float qk = _sv.un.rotationVector.k;
            float qr = _sv.un.rotationVector.real;
            quat_to_euler(qi, qj, qk, qr, &_pitch_mag, &_roll_mag, &_yaw_mag);

        } else if (_sv.sensorId == SH2_GYROSCOPE_CALIBRATED) {
            _pitch_rate = _sv.un.gyroscope.y;
            _roll_rate  = _sv.un.gyroscope.x;
            _yaw_rate   = _sv.un.gyroscope.z;
        }
    }

    // Sensor silence timeout — only fires if we've been actively polling but data stopped
    if (now - _last_update_ms > TIMEOUT_MS && now - _last_poll_ms >= TIMEOUT_MS) {
        _state        = ImuState::ERROR;
        _retry_due_ms = now + RETRY_INTERVAL_MS;
        return;
    }

    // NOMINAL vs DEGRADED based on rolling loss window
    _state = (_packet_loss >= LOSS_THRESHOLD) ? ImuState::DEGRADED : ImuState::NOMINAL;
}

// ── Accessors ─────────────────────────────────────────────────────────────────
ImuState imu_state()          { return _state; }
float    imu_pitch()          { return _pitch; }
float    imu_roll()           { return _roll; }
float    imu_yaw()            { return _yaw; }
float    imu_pitch_mag()      { return _pitch_mag; }
float    imu_roll_mag()       { return _roll_mag; }
float    imu_yaw_mag()        { return _yaw_mag; }
float    imu_pitch_rate()     { return _pitch_rate; }
float    imu_roll_rate()      { return _roll_rate; }
float    imu_yaw_rate()       { return _yaw_rate; }
float    imu_packet_loss()    { return _packet_loss; }
uint32_t imu_last_update_ms() { return _last_update_ms; }
