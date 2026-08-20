#pragma once

#include <math.h>
#include <stdint.h>

// The BNO086 integrated gyro report is nominally 400 Hz, so one value is held
// across some 500 Hz control ticks. Only fresh sensor samples contribute to the
// score. Accumulating vector changes makes the impact score independent of the
// control-loop cadence. Twelve milliseconds covers several fresh reports in
// both reference landings while staying much shorter than body recovery motion.
static constexpr uint32_t JUMP_LANDING_GYRO_WINDOW_MS = 12;
static constexpr uint8_t  JUMP_LANDING_GYRO_MIN_EVENTS = 2;
static constexpr float    JUMP_LANDING_GYRO_EVENT_EPS = 0.01f;
static constexpr uint8_t  JUMP_LANDING_GYRO_HISTORY = 8;

struct JumpLandingDetector {
    bool     landed;
    uint32_t retract_start_ms;
    uint32_t last_gyro_sample_ms;
    uint32_t gyro_event_ms[JUMP_LANDING_GYRO_HISTORY];
    float    gyro_event_delta[JUMP_LANDING_GYRO_HISTORY];
    uint8_t  gyro_event_head;
    uint8_t  gyro_event_count;
    float    prev_roll_rate;
    float    prev_pitch_rate;
    float    prev_yaw_rate;
};

inline void jump_landing_reset(JumpLandingDetector* d,
                               uint32_t now_ms,
                               uint32_t gyro_sample_ms,
                               float roll_rate,
                               float pitch_rate,
                               float yaw_rate) {
    d->landed              = false;
    d->retract_start_ms    = now_ms;
    d->last_gyro_sample_ms = gyro_sample_ms;
    d->gyro_event_head     = 0;
    d->gyro_event_count    = 0;
    d->prev_roll_rate      = roll_rate;
    d->prev_pitch_rate     = pitch_rate;
    d->prev_yaw_rate       = yaw_rate;
}

// Touchdown is a short, multi-sample 3-axis gyro impulse. It is blanked for
// min_air_s after RETRACT starts so the launch impulse cannot be mistaken for
// touchdown. The result latches until the next reset. Linear acceleration is
// deliberately absent from this path.
inline bool jump_landing_update(JumpLandingDetector* d,
                                uint32_t now_ms,
                                uint32_t gyro_sample_ms,
                                float roll_rate,
                                float pitch_rate,
                                float yaw_rate,
                                float min_air_s,
                                float gyro_impulse_threshold) {
    if (d->landed) return true;

    if (gyro_sample_ms != 0 && gyro_sample_ms != d->last_gyro_sample_ms) {
        d->last_gyro_sample_ms = gyro_sample_ms;
        const float dr = roll_rate  - d->prev_roll_rate;
        const float dp = pitch_rate - d->prev_pitch_rate;
        const float dy = yaw_rate   - d->prev_yaw_rate;
        const float gyro_delta = sqrtf(dr * dr + dp * dp + dy * dy);
        if (gyro_delta > JUMP_LANDING_GYRO_EVENT_EPS) {
            d->gyro_event_ms[d->gyro_event_head]    = gyro_sample_ms;
            d->gyro_event_delta[d->gyro_event_head] = gyro_delta;
            d->gyro_event_head = (uint8_t)((d->gyro_event_head + 1) % JUMP_LANDING_GYRO_HISTORY);
            if (d->gyro_event_count < JUMP_LANDING_GYRO_HISTORY) d->gyro_event_count++;
        }
        d->prev_roll_rate  = roll_rate;
        d->prev_pitch_rate = pitch_rate;
        d->prev_yaw_rate   = yaw_rate;
    }

    const uint32_t min_air_ms = (uint32_t)(min_air_s * 1000.0f);
    const bool eligible = (uint32_t)(now_ms - d->retract_start_ms) >= min_air_ms;

    float gyro_delta_sum = 0.0f;
    uint8_t gyro_events_in_window = 0;
    for (uint8_t i = 0; i < d->gyro_event_count; ++i) {
        const uint8_t index = (uint8_t)((d->gyro_event_head + JUMP_LANDING_GYRO_HISTORY - 1 - i)
                                        % JUMP_LANDING_GYRO_HISTORY);
        if ((uint32_t)(now_ms - d->gyro_event_ms[index]) > JUMP_LANDING_GYRO_WINDOW_MS) break;
        gyro_delta_sum += d->gyro_event_delta[index];
        gyro_events_in_window++;
    }
    const bool gyro_landing =
        gyro_events_in_window >= JUMP_LANDING_GYRO_MIN_EVENTS &&
        gyro_delta_sum >= gyro_impulse_threshold;

    d->landed = eligible && gyro_landing;
    return d->landed;
}
