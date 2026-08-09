// IMU.cpp — BNO086 SPI driver using Adafruit_BNO08x library.
//
// Uses Game Rotation Vector (no magnetometer — immune to motor field disturbance)
// and calibrated gyro at 400 Hz; the magnetometer-fused Rotation Vector (10 Hz)
// and linear acceleration (50 Hz) are auxiliary — see the rate budget in Config.
// Tracks packet loss via SH2 per-report sequence numbers.
// Pins: CS=10, INT=2, RST=3 (SPI0). PS0+PS1 bridged for SPI mode.

#include <Arduino.h>
#include <SPI.h>
#include <Adafruit_BNO08x.h>
#include <sh2.h>
#include "config.h"
#include "IMU.h"
#include "comm_protocol.h"

// ── Config ───────────────────────────────────────────────────────────────────
// Per-report rates. Only GRV and gyro feed the control loop; the other two are
// requested far slower because every enabled report costs SHTP events that all
// share one FIFO and one per-tick drain budget (MAX_DRAIN_EVENTS). Running all
// four at 400 Hz asked for 1600 events/s against a 4000/s ceiling — under 3x
// headroom, and a backlog there starves whichever report sits behind the others
// in the queue. That is the leading suspect for the 2026-08-09 gyro stall, so
// the aux streams are cut to what their consumers actually need:
//   GRV  400 Hz (attitude -> LQR)      gyro 400 Hz (body rates -> LQR)
//   RV    10 Hz (absolute heading, no consumer in src/ today)
//   accel 50 Hz (telemetry display only)
// Total ~860 events/s — a ~46 % cut with no loss of control-relevant bandwidth.
static constexpr uint32_t IMU_RATE_HZ       = 400;  // required streams: GRV + gyro
static constexpr uint32_t IMU_MAG_RATE_HZ   = 10;   // RV/mag — heading reference
static constexpr uint32_t IMU_AUX_RATE_HZ   = 50;   // linear accel — telemetry only
static constexpr uint32_t TIMEOUT_MS        = 100;   // data silence while actively polling before ERROR
static constexpr uint8_t  MAX_INIT_ATTEMPTS = 2;     // it's a fast digital device — a couple of quick tries, then give up
static constexpr uint32_t LOSS_WINDOW_MS    = 1000;
static constexpr float    LOSS_THRESHOLD    = 0.10f;
static constexpr uint8_t  MAX_DRAIN_EVENTS  = 8;     // GRV + RV + Gyro + LinAccel per cycle

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
static float     _accel_x        = 0.0f;
static float     _accel_y        = 0.0f;
static float     _accel_z        = 0.0f;
static float     _packet_loss    = 0.0f;
// Per-report arrival stamps. The control loop consumes attitude (GRV) and body
// rates (gyro), and these two streams stall independently: on 2026-08-09 the
// gyro stopped arriving while GRV kept flowing, which left imu_state() NOMINAL,
// froze pitch_rate at 0.000 for 3.66 s and silently zeroed the LQR's entire
// rate-damping term. A single stamp fed only by GRV cannot see that, so the
// silence watchdog in imu_update() checks both. The two unrequired reports
// (RV/mag, linear accel) are deliberately not watched — nothing in the control
// path reads them, so faulting on them would add fragility for no safety gain.
static uint32_t  _last_grv_ms    = 0;
static uint32_t  _last_gyro_ms   = 0;
static uint8_t   _init_attempts  = 0;   // resets on imu_init() and on any fresh reconnect attempt

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

// (clampf removed with the asin-based roll extraction — atan2 needs no clamp.)

// Intrinsic Z-Y-X (yaw → pitch → roll), the standard aerospace convention.
// Pitch is the nose-down lean of body +X out of the horizontal plane and is
// independent of heading — which is what the balance controller needs, and why
// ZYX is the right family here. (Y-X-Z, whose singularity sits at |roll| = 90°
// instead, looks tempting for a machine that falls fore/aft, but its "pitch" is
// heading-dependent: it reads a 10° lean as anywhere from -9° to +54° depending
// on which way the robot happens to be facing. Do not switch to it.)
//
// Previously the numerators here were right but two adjacent sides were not:
// pitch divided by R22 instead of hypot(R00, R10), and roll used asin(R21)
// instead of atan2(R21, R22). Both are exact only when the *other* angle is
// zero, so error grew with roll — 0.8° at 15° roll, 3.3° at 30°, 18° at 60°.
// Harmless upright, meaningless once tipped over, which is exactly when a
// fallen-robot log most needs to be trustworthy.
//
// Singularity: pitch = ±90° (nose vertical) leaves roll and yaw degenerate.
// Unreachable in any state the robot is allowed to operate in — the pitch
// watchdog ESTOPs far below it — and pitch itself stays exact regardless.
static void quat_to_euler(float qi, float qj, float qk, float qr,
                           float *pitch, float *roll, float *yaw) {
    // Rotation-matrix elements this needs (body → world).
    const float r00 = 1.0f - 2.0f * (qj * qj + qk * qk);
    const float r10 = 2.0f * (qi * qj + qk * qr);
    const float r20 = 2.0f * (qi * qk - qj * qr);
    const float r21 = 2.0f * (qj * qk + qi * qr);
    const float r22 = 1.0f - 2.0f * (qi * qi + qj * qj);

    // Pitch about Y — positive = lean forward (+X down). atan2 against the
    // hypotenuse of the first column rather than asin(-r20): same value, but
    // it stays well-conditioned as the nose approaches vertical.
    *pitch = atan2f(-r20, sqrtf(r00 * r00 + r10 * r10));

    // Roll about X — positive = lean right
    *roll = atan2f(r21, r22);

    // Yaw about Z
    *yaw = atan2f(r10, r00);
}

static bool enable_reports() {
    constexpr uint32_t interval_us     = 1000000UL / IMU_RATE_HZ;
    constexpr uint32_t interval_mag_us = 1000000UL / IMU_MAG_RATE_HZ;
    constexpr uint32_t interval_aux_us = 1000000UL / IMU_AUX_RATE_HZ;
    // Required streams first: if the budget is tight, the two the control loop
    // depends on are the ones already registered.
    if (!_sensor.enableReport(SH2_GAME_ROTATION_VECTOR, interval_us)) {
        Serial.println("[IMU] Failed to enable Game Rotation Vector");
        return false;
    }
    if (!_sensor.enableReport(SH2_GYROSCOPE_CALIBRATED, interval_us)) {
        Serial.println("[IMU] Failed to enable Gyro");
        return false;
    }
    if (!_sensor.enableReport(SH2_ROTATION_VECTOR, interval_mag_us)) {
        Serial.println("[IMU] Failed to enable Rotation Vector (mag)");
        return false;
    }
    if (!_sensor.enableReport(SH2_LINEAR_ACCELERATION, interval_aux_us)) {
        Serial.println("[IMU] Failed to enable Linear Acceleration");
        return false;
    }
    return true;
}

// Blocks ~0.7-1.0 s: begin_SPI()/sh2_open()/sh2_getProdIds() inside the
// Adafruit_BNO08x/SH2 stack are all synchronous with no timeout of their own
// (measured via test_imu SH2 DEBUG output: hardwareReset ~30 ms + sh2_open
// ~150 ms + getProdIds ~690 ms). Only ever called from the INITIALIZING/ERROR
// case in imu_update() below — reached at boot (run to completion by the
// blocking wait in setup(), since STARTUP has no 500 Hz budget to protect —
// see main.cpp) or on an init failure retry within that same boot attempt.
// A sensor that goes silent mid-operation (NOMINAL/DEGRADED -> ERROR) does
// NOT come back through here — that path is terminal by design so a live
// loop() tick is never stalled by a reconnect attempt; see the silence-
// timeout comment in imu_update().
static bool attempt_init() {
    if (!_sensor.begin_SPI(PIN_IMU_CS, PIN_IMU_INT)) {
        // Adafruit_BNO08x::_init() leaks its SHTP pool slot on failure (never
        // calls sh2_close()) — without this, every retry after the first
        // failure fails begin_SPI() instantly (pool exhausted) and NOMINAL
        // becomes permanently unreachable, even if the sensor would have
        // responded on a later attempt.
        sh2_close();
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
    _init_attempts     = 0;
    _rv_seq_valid      = false;
    _packet_loss       = 0.0f;
    reset_loss_window(millis());
}

void imu_update() {
    uint32_t now = millis();

    switch (_state) {

    case ImuState::NOT_READY:
        return;

    // INITIALIZING means "still trying" (startup_fail() in state_machine.cpp
    // only faults on ERROR, so staying INITIALIZING here keeps the robot in
    // STARTUP without tripping ESTOP mid-attempt). ERROR is reached only once
    // MAX_INIT_ATTEMPTS is exhausted, and is then final — no more attempts are
    // made until imu_init() runs again (i.e. a reboot). It's a fast digital
    // device: a real connection succeeds on the first try, so there's no
    // value in pacing retries — back-to-back is fine.
    case ImuState::INITIALIZING:
    case ImuState::ERROR:
        if (_init_attempts >= MAX_INIT_ATTEMPTS) return;
        _init_attempts++;
        {
            uint32_t t0 = millis();
            bool ok = attempt_init();
            uint32_t dt = millis() - t0;
            comm_log(LOG_LEVEL_INFO, "IMU attempt %u/%u: %s (%lu ms)",
                     (unsigned)_init_attempts, (unsigned)MAX_INIT_ATTEMPTS,
                     ok ? "OK" : "FAIL", (unsigned long)dt);
            if (ok) {
                uint32_t t_ok = millis();
                _state         = ImuState::NOMINAL;
                _last_grv_ms   = t_ok;   // seed both, else the silence watchdog
                _last_gyro_ms  = t_ok;   // trips before the first packet lands
                _rv_seq_valid  = false;
                _packet_loss   = 0.0f;
                reset_loss_window(t_ok);
            } else {
                _state = (_init_attempts >= MAX_INIT_ATTEMPTS) ? ImuState::ERROR : ImuState::INITIALIZING;
            }
        }
        return;

    case ImuState::NOMINAL:
    case ImuState::DEGRADED:
        break;
    }

    // Recover from unexpected sensor reset (re-enable reports, reset seq tracking).
    // enable_reports() bails on its first failure, so a partial re-enable is a
    // real possibility: GRV comes back, the gyro doesn't, and the robot then
    // balances on a frozen rate with no indication anything is wrong. Dropping
    // this return value is how that state became reachable — treat it as fatal.
    if (_sensor.wasReset()) {
        if (!enable_reports()) {
            comm_log(LOG_LEVEL_ERROR, "IMU report re-enable FAILED after sensor reset");
            _state         = ImuState::ERROR;
            _init_attempts = MAX_INIT_ATTEMPTS;
            return;
        }
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
            _last_grv_ms = now;
            track_packet(_sv.sequence);

        } else if (_sv.sensorId == SH2_ROTATION_VECTOR) {
            float qi = _sv.un.rotationVector.i;
            float qj = _sv.un.rotationVector.j;
            float qk = _sv.un.rotationVector.k;
            float qr = _sv.un.rotationVector.real;
            quat_to_euler(qi, qj, qk, qr, &_pitch_mag, &_roll_mag, &_yaw_mag);

        } else if (_sv.sensorId == SH2_GYROSCOPE_CALIBRATED) {
            _pitch_rate   = _sv.un.gyroscope.y;
            _roll_rate    = _sv.un.gyroscope.x;
            _yaw_rate     = _sv.un.gyroscope.z;
            _last_gyro_ms = now;
        } else if (_sv.sensorId == SH2_LINEAR_ACCELERATION) {
            _accel_x = _sv.un.linearAcceleration.x;
            _accel_y = _sv.un.linearAcceleration.y;
            _accel_z = _sv.un.linearAcceleration.z;
        }
    }

    // Sensor silence timeout — we only reach here while NOMINAL/DEGRADED (i.e.
    // actively polling), so data silence alone is enough to declare ERROR.
    // Terminal, no auto-reconnect: a mid-operation dropout means the robot is
    // already past STARTUP (running_imu_fault()/other state logic reacts to
    // leaving NOMINAL immediately), and attempt_init() blocks ~1 s — fine for
    // STARTUP's one-time blocking wait in setup(), not for a live loop() tick.
    // Locking out further attempt_init() calls here (rather than resetting
    // _init_attempts for a fresh retry budget, as before) means recovery
    // requires an explicit reboot (imu_init() resets the budget).
    //
    // Checked per stream: either required report going silent is unsafe on its
    // own, and a gyro-only stall used to be completely invisible here. Naming
    // the stalled stream in the log is most of the diagnostic value — that
    // distinction is what makes this failure identifiable in a run log.
    bool grv_stale  = (now - _last_grv_ms)  > TIMEOUT_MS;
    bool gyro_stale = (now - _last_gyro_ms) > TIMEOUT_MS;
    if (grv_stale || gyro_stale) {
        comm_log(LOG_LEVEL_ERROR, "IMU stream stalled: GRV=%s gyro=%s",
                 grv_stale  ? "STALE" : "ok",
                 gyro_stale ? "STALE" : "ok");
        _state         = ImuState::ERROR;
        _init_attempts = MAX_INIT_ATTEMPTS;
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
float    imu_accel_x()        { return _accel_x; }
float    imu_accel_y()        { return _accel_y; }
float    imu_accel_z()        { return _accel_z; }
float    imu_packet_loss()    { return _packet_loss; }
// Weakest link of the two required streams, so callers doing their own
// freshness check (e.g. the arm gate in state_machine.cpp) cover the gyro too
// without needing to know there is more than one stream.
uint32_t imu_last_update_ms() {
    return (_last_grv_ms < _last_gyro_ms) ? _last_grv_ms : _last_gyro_ms;
}
