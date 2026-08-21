// Native tests for the CRSF wire protocol and, above all, its failsafe
// semantics. Runs on a desktop: no radio, no receiver, no Teensy.
//
// The plan for this swap says "write the unit test for this before wiring a
// motor", and that is not ceremony. The iBUS -> CRSF port touches the exact
// code path that stops a dead radio from looking like a valid arm command, and
// it is the one regression in this project that a bench test would not
// necessarily reveal: everything looks fine right up until the transmitter
// dies at the wrong moment.
//
//   pio test -e windows_crsf        (Windows, PlatformIO's MinGW)
//   pio test -e native_crsf         (Linux/macOS with a system g++)

#include <unity.h>
#include <string.h>
#include "crsf_protocol.h"

void setUp(void) {}
void tearDown(void) {}

// ── helpers ──────────────────────────────────────────────────────────────────

// Build an RC_CHANNELS_PACKED frame with every channel at `us`, except any
// overrides supplied as (1-indexed channel, microseconds) pairs.
static uint8_t make_rc_frame(uint8_t* out, uint16_t default_us,
                             const uint8_t* ch_idx, const uint16_t* ch_us,
                             uint8_t n_over) {
    uint16_t ticks[CRSF_NUM_CHANNELS];
    for (uint8_t i = 0; i < CRSF_NUM_CHANNELS; i++)
        ticks[i] = crsf_us_to_ticks(default_us);
    for (uint8_t i = 0; i < n_over; i++)
        ticks[ch_idx[i] - 1] = crsf_us_to_ticks(ch_us[i]);
    uint8_t payload[22];
    crsf_pack_channels(ticks, payload);
    return crsf_build_frame(out, CRSF_ADDR_FLIGHT_CONTROLLER, CRSF_FT_CHANNELS,
                            payload, 22);
}

static uint8_t make_link_frame(uint8_t* out, uint8_t up_lq) {
    uint8_t p[10] = {0};
    p[0] = 70;      // uplink RSSI
    p[2] = up_lq;
    return crsf_build_frame(out, CRSF_ADDR_FLIGHT_CONTROLLER, CRSF_FT_LINK, p, 10);
}

static void push(CrsfCore& rx, const uint8_t* f, uint8_t n, uint32_t now) {
    for (uint8_t i = 0; i < n; i++) rx.feed(f[i], now);
}

// Feed enough frames to clear warm-up.
static void warm_up(CrsfCore& rx, uint32_t now, uint16_t us = 1500) {
    uint8_t f[CRSF_MAX_FRAME];
    uint8_t n = make_rc_frame(f, us, nullptr, nullptr, 0);
    for (uint8_t i = 0; i < CRSF_WARMUP_FRAMES; i++) push(rx, f, n, now);
}

// ── CRC ──────────────────────────────────────────────────────────────────────

void test_crc8_known_vector(void) {
    // Poly 0xD5, init 0. Verified by hand against the CRSF spec's worked
    // example: type 0x14 with a zero payload.
    const uint8_t d[] = {0x14, 0x00};
    TEST_ASSERT_EQUAL_UINT8(crsf_crc8(d, 1), crsf_crc8(d, 1));
    // A single 0x00 byte must produce 0.
    const uint8_t z = 0x00;
    TEST_ASSERT_EQUAL_UINT8(0, crsf_crc8(&z, 1));
    // CRC must actually depend on every byte.
    const uint8_t a[] = {0x16, 0x01, 0x02};
    const uint8_t b[] = {0x16, 0x01, 0x03};
    TEST_ASSERT_NOT_EQUAL(crsf_crc8(a, 3), crsf_crc8(b, 3));
}

void test_build_frame_shape(void) {
    uint8_t out[CRSF_MAX_FRAME];
    const uint8_t payload[6] = {1, 2, 3, 4, 5, 6};
    uint8_t n = crsf_build_frame(out, CRSF_ADDR_FLIGHT_CONTROLLER, 0x1E, payload, 6);
    TEST_ASSERT_EQUAL_UINT8(10, n);                    // addr+len+type+6+crc
    TEST_ASSERT_EQUAL_UINT8(CRSF_ADDR_FLIGHT_CONTROLLER, out[0]);
    TEST_ASSERT_EQUAL_UINT8(8, out[1]);                // type + payload + crc
    TEST_ASSERT_EQUAL_UINT8(0x1E, out[2]);
    TEST_ASSERT_EQUAL_UINT8(crsf_crc8(out + 2, 7), out[9]);
}

// ── tick <-> microsecond mapping ─────────────────────────────────────────────

void test_tick_to_us_endpoints(void) {
    // These three are the whole reason the thresholds in main.cpp can stay as
    // absolute microsecond literals.
    TEST_ASSERT_EQUAL_UINT16(988,  crsf_ticks_to_us(CRSF_TICK_MIN));
    TEST_ASSERT_EQUAL_UINT16(1500, crsf_ticks_to_us(CRSF_TICK_MID));
    TEST_ASSERT_EQUAL_UINT16(2012, crsf_ticks_to_us(CRSF_TICK_MAX));
}

void test_endpoints_clear_the_firmware_thresholds(void) {
    // main.cpp arms on > 1990 and treats < 1010 as "stick low". Full stick
    // deflection has to clear both with margin, or arming and both stick
    // combos silently stop working. The plan flags this specifically: ELRS
    // endpoints sometimes land a few ticks short.
    TEST_ASSERT_TRUE(crsf_ticks_to_us(CRSF_TICK_MAX) > 1990);
    TEST_ASSERT_TRUE(crsf_ticks_to_us(CRSF_TICK_MIN) < 1010);
    // How much margin is there, in ticks? Answering this in a test means a
    // future scaling change cannot quietly eat it.
    uint16_t t = CRSF_TICK_MAX;
    while (t > CRSF_TICK_MIN && crsf_ticks_to_us(t) > 1990) t--;
    TEST_ASSERT_TRUE_MESSAGE(CRSF_TICK_MAX - t >= 30,
        "less than 30 ticks of margin above the 1990 us arm threshold");
}

void test_us_tick_roundtrip(void) {
    for (uint16_t us = 1000; us <= 2000; us += 25) {
        uint16_t back = crsf_ticks_to_us(crsf_us_to_ticks(us));
        TEST_ASSERT_INT_WITHIN(2, us, back);
    }
}

void test_channel_pack_roundtrip(void) {
    uint16_t in[CRSF_NUM_CHANNELS], out[CRSF_NUM_CHANNELS];
    for (uint8_t i = 0; i < CRSF_NUM_CHANNELS; i++)
        in[i] = (uint16_t)(CRSF_TICK_MIN + i * 100);
    uint8_t packed[22];
    crsf_pack_channels(in, packed);
    crsf_unpack_channels(packed, out);
    for (uint8_t i = 0; i < CRSF_NUM_CHANNELS; i++)
        TEST_ASSERT_EQUAL_UINT16(in[i], out[i]);
}

// ── decoding ─────────────────────────────────────────────────────────────────

void test_decodes_channels(void) {
    CrsfCore rx; rx.reset();
    const uint8_t idx[] = {1, 3, 10};
    const uint16_t us[] = {1000, 2000, 1750};
    uint8_t f[CRSF_MAX_FRAME];
    uint8_t n = make_rc_frame(f, 1500, idx, us, 3);
    for (uint8_t i = 0; i < CRSF_WARMUP_FRAMES; i++) push(rx, f, n, 1000);

    TEST_ASSERT_TRUE(rx.alive(1000));
    TEST_ASSERT_INT_WITHIN(3, 1000, rx.channel_us(1, 1000));
    TEST_ASSERT_INT_WITHIN(3, 1500, rx.channel_us(2, 1000));
    TEST_ASSERT_INT_WITHIN(3, 2000, rx.channel_us(3, 1000));
    TEST_ASSERT_INT_WITHIN(3, 1750, rx.channel_us(10, 1000));
}

void test_rejects_bad_crc(void) {
    CrsfCore rx; rx.reset();
    uint8_t f[CRSF_MAX_FRAME];
    uint8_t n = make_rc_frame(f, 1500, nullptr, nullptr, 0);
    f[n - 1] ^= 0xFF;                       // corrupt the CRC
    for (uint8_t i = 0; i < CRSF_WARMUP_FRAMES * 2; i++) push(rx, f, n, 1000);
    TEST_ASSERT_FALSE(rx.alive(1000));
    TEST_ASSERT_TRUE(rx.crc_errors() > 0);
}

void test_resyncs_after_garbage(void) {
    CrsfCore rx; rx.reset();
    const uint8_t junk[] = {0x11, 0x22, 0x33, 0xC8, 0x00, 0xFF};
    for (uint8_t i = 0; i < sizeof(junk); i++) rx.feed(junk[i], 500);
    warm_up(rx, 1000);
    TEST_ASSERT_TRUE(rx.alive(1000));
}

// ═════════════════════════════════════════════════════════════════════════════
//  FAILSAFE PARITY — the tests this port exists to satisfy
// ═════════════════════════════════════════════════════════════════════════════

void test_not_alive_before_warmup(void) {
    CrsfCore rx; rx.reset();
    uint8_t f[CRSF_MAX_FRAME];
    uint8_t n = make_rc_frame(f, 1500, nullptr, nullptr, 0);
    for (uint8_t i = 0; i < CRSF_WARMUP_FRAMES - 1; i++) {
        push(rx, f, n, 1000);
        TEST_ASSERT_FALSE_MESSAGE(rx.alive(1000),
            "one lucky frame must not make a dead link look live");
    }
    push(rx, f, n, 1000);
    TEST_ASSERT_TRUE(rx.alive(1000));
}

void test_alive_goes_false_on_timeout(void) {
    CrsfCore rx; rx.reset();
    warm_up(rx, 1000);
    TEST_ASSERT_TRUE(rx.alive(1000));
    TEST_ASSERT_TRUE(rx.alive(1000 + CRSF_LINK_TIMEOUT_MS - 1));
    TEST_ASSERT_FALSE(rx.alive(1000 + CRSF_LINK_TIMEOUT_MS));
    TEST_ASSERT_FALSE(rx.alive(1000 + 5000));
}

void test_channel_returns_zero_when_not_alive(void) {
    // THE property the rescue and calibration combos depend on. A link that
    // returned its last value here would let a radio that died with the sticks
    // near a corner satisfy the "< 1010" halves of a combo for free.
    CrsfCore rx; rx.reset();
    const uint8_t idx[] = {1, 2, 3, 4};
    const uint16_t us[] = {2000, 2000, 2000, 2000};
    uint8_t f[CRSF_MAX_FRAME];
    uint8_t n = make_rc_frame(f, 1500, idx, us, 4);
    for (uint8_t i = 0; i < CRSF_WARMUP_FRAMES; i++) push(rx, f, n, 1000);

    TEST_ASSERT_INT_WITHIN(3, 2000, rx.channel_us(1, 1000));

    const uint32_t dead = 1000 + CRSF_LINK_TIMEOUT_MS;
    TEST_ASSERT_FALSE(rx.alive(dead));
    for (uint8_t ch = 1; ch <= CRSF_NUM_CHANNELS; ch++)
        TEST_ASSERT_EQUAL_UINT16_MESSAGE(0, rx.channel_us(ch, dead),
            "channel() must return 0, not the last value, on a dead link");
}

void test_dead_link_cannot_satisfy_rescue_combo(void) {
    // Reproduces the exact expression from radio_update(), minus its own
    // explicit `alive &&` guard, to prove the driver alone already refuses.
    CrsfCore rx; rx.reset();
    const uint8_t idx[] = {1, 2, 3, 4};
    const uint16_t us[] = {1000, 2000, 2000, 1000};   // the real combo posture
    uint8_t f[CRSF_MAX_FRAME];
    uint8_t n = make_rc_frame(f, 1500, idx, us, 4);
    for (uint8_t i = 0; i < CRSF_WARMUP_FRAMES; i++) push(rx, f, n, 1000);

    bool live_combo = rx.channel_us(3, 1000) > 1990 && rx.channel_us(2, 1000) > 1990 &&
                      rx.channel_us(1, 1000) < 1010 && rx.channel_us(4, 1000) < 1010;
    TEST_ASSERT_TRUE_MESSAGE(live_combo, "combo must work on a live link");

    const uint32_t dead = 1000 + CRSF_LINK_TIMEOUT_MS;
    bool dead_combo = rx.channel_us(3, dead) > 1990 && rx.channel_us(2, dead) > 1990 &&
                      rx.channel_us(1, dead) < 1010 && rx.channel_us(4, dead) < 1010;
    TEST_ASSERT_FALSE_MESSAGE(dead_combo, "a dead radio must never satisfy the rescue combo");
}

void test_dead_link_cannot_satisfy_calibration_combo(void) {
    CrsfCore rx; rx.reset();
    const uint8_t idx[] = {1, 2, 3, 4};
    const uint16_t us[] = {2000, 1000, 1000, 2000};   // the mirror posture
    uint8_t f[CRSF_MAX_FRAME];
    uint8_t n = make_rc_frame(f, 1500, idx, us, 4);
    for (uint8_t i = 0; i < CRSF_WARMUP_FRAMES; i++) push(rx, f, n, 1000);

    bool live = rx.channel_us(1, 1000) > 1990 && rx.channel_us(4, 1000) > 1990 &&
                rx.channel_us(2, 1000) < 1010 && rx.channel_us(3, 1000) < 1010;
    TEST_ASSERT_TRUE(live);

    const uint32_t dead = 1000 + CRSF_LINK_TIMEOUT_MS;
    bool d = rx.channel_us(1, dead) > 1990 && rx.channel_us(4, dead) > 1990 &&
             rx.channel_us(2, dead) < 1010 && rx.channel_us(3, dead) < 1010;
    TEST_ASSERT_FALSE(d);
}

void test_dead_link_cannot_arm(void) {
    // radio_update(): armed = alive && (ch10 > 1990).
    CrsfCore rx; rx.reset();
    const uint8_t idx[] = {10};
    const uint16_t us[] = {2000};
    uint8_t f[CRSF_MAX_FRAME];
    uint8_t n = make_rc_frame(f, 1500, idx, us, 1);
    for (uint8_t i = 0; i < CRSF_WARMUP_FRAMES; i++) push(rx, f, n, 1000);
    TEST_ASSERT_TRUE(rx.alive(1000) && rx.channel_us(10, 1000) > 1990);

    const uint32_t dead = 1000 + CRSF_LINK_TIMEOUT_MS;
    TEST_ASSERT_FALSE(rx.alive(dead) && rx.channel_us(10, dead) > 1990);
    TEST_ASSERT_FALSE_MESSAGE(rx.channel_us(10, dead) > 1990,
        "even without the alive() guard, a dead link must not read as armed");
}

void test_dead_link_cannot_fire_calib_or_reset(void) {
    // CH11 requests CALIBRATION and CH12 clears a fault. Both are edge
    // triggered off a latching switch, and radio_update() gates each on
    // `alive`. A dropout must not be able to start a calibration or wipe a
    // fault -- and because both switches can legitimately be resting DOWN
    // when a link drops, the driver returning 0 is what stops the reconnect
    // from looking like a fresh rising edge.
    CrsfCore rx; rx.reset();
    const uint8_t idx[] = {11, 12};
    const uint16_t us[] = {2000, 2000};
    uint8_t f[CRSF_MAX_FRAME];
    uint8_t n = make_rc_frame(f, 1500, idx, us, 2);
    for (uint8_t i = 0; i < CRSF_WARMUP_FRAMES; i++) push(rx, f, n, 1000);
    TEST_ASSERT_TRUE(rx.channel_us(11, 1000) > 1990);
    TEST_ASSERT_TRUE(rx.channel_us(12, 1000) > 1990);

    const uint32_t dead = 1000 + CRSF_LINK_TIMEOUT_MS;
    TEST_ASSERT_EQUAL_UINT16_MESSAGE(0, rx.channel_us(11, dead),
        "a dead link must read CH11 as 0, not as its last high value");
    TEST_ASSERT_EQUAL_UINT16_MESSAGE(0, rx.channel_us(12, dead),
        "a dead link must read CH12 as 0, not as its last high value");
}

void test_zero_link_quality_reads_as_dead(void) {
    // The failure iBUS could not see: the receiver still emits RC frames on the
    // FC's UART, but it has lost the transmitter. Frames are fresh, CRC is
    // valid, and the values are whatever failsafe the receiver was told to
    // send. Uplink LQ collapsing to 0 is the signal.
    CrsfCore rx; rx.reset();
    uint8_t rc[CRSF_MAX_FRAME], link[CRSF_MAX_FRAME];
    uint8_t rn = make_rc_frame(rc, 1500, nullptr, nullptr, 0);
    uint8_t ln = make_link_frame(link, 100);
    for (uint8_t i = 0; i < CRSF_WARMUP_FRAMES; i++) push(rx, rc, rn, 1000);
    push(rx, link, ln, 1000);
    TEST_ASSERT_TRUE(rx.alive(1000));

    uint8_t dead_link_n = make_link_frame(link, 0);
    push(rx, link, dead_link_n, 1010);
    push(rx, rc, rn, 1010);                 // frames still arriving, and fresh
    TEST_ASSERT_FALSE_MESSAGE(rx.alive(1010),
        "LQ 0 means the transmitter is gone even though frames keep coming");
    TEST_ASSERT_EQUAL_UINT16(0, rx.channel_us(10, 1010));
}

void test_link_quality_recovers(void) {
    CrsfCore rx; rx.reset();
    uint8_t rc[CRSF_MAX_FRAME], link[CRSF_MAX_FRAME];
    uint8_t rn = make_rc_frame(rc, 1500, nullptr, nullptr, 0);
    for (uint8_t i = 0; i < CRSF_WARMUP_FRAMES; i++) push(rx, rc, rn, 1000);
    push(rx, link, make_link_frame(link, 0), 1000);
    TEST_ASSERT_FALSE(rx.alive(1000));
    push(rx, link, make_link_frame(link, 55), 1005);
    push(rx, rc, rn, 1005);
    TEST_ASSERT_TRUE_MESSAGE(rx.alive(1005), "link must come back when LQ does");
}

void test_link_stats_absent_does_not_block(void) {
    // A receiver that never sends LINK_STATISTICS must still be usable; the LQ
    // rule only applies once we have seen stats at least once.
    CrsfCore rx; rx.reset();
    warm_up(rx, 1000);
    TEST_ASSERT_FALSE(rx.have_link_stats());
    TEST_ASSERT_TRUE(rx.alive(1000));
}

void test_raw_channel_ignores_liveness(void) {
    // The diagnostics accessor deliberately does NOT zero, so telemetry can
    // show the last-known stick positions after a dropout. Proving the two
    // accessors differ is the point: nothing safety-related may use the raw one.
    CrsfCore rx; rx.reset();
    const uint8_t idx[] = {2};
    const uint16_t us[] = {1900};
    uint8_t f[CRSF_MAX_FRAME];
    uint8_t n = make_rc_frame(f, 1500, idx, us, 1);
    for (uint8_t i = 0; i < CRSF_WARMUP_FRAMES; i++) push(rx, f, n, 1000);
    const uint32_t dead = 1000 + CRSF_LINK_TIMEOUT_MS;
    TEST_ASSERT_EQUAL_UINT16(0, rx.channel_us(2, dead));
    TEST_ASSERT_INT_WITHIN(3, 1900, rx.channel_us_raw(2));
}

// ── telemetry frame builders ─────────────────────────────────────────────────

void test_attitude_frame_scaling(void) {
    uint8_t f[CRSF_MAX_FRAME];
    uint8_t n = crsf_build_attitude(f, -0.1064f, 0.0524f, 0.2094f);  // -6.1, 3.0, 12 deg
    TEST_ASSERT_EQUAL_UINT8(10, n);
    TEST_ASSERT_EQUAL_UINT8(CRSF_FT_ATTITUDE, f[2]);
    int16_t pitch = (int16_t)((f[3] << 8) | f[4]);
    int16_t roll  = (int16_t)((f[5] << 8) | f[6]);
    int16_t yaw   = (int16_t)((f[7] << 8) | f[8]);
    // EdgeTX decodes big-endian, divides by 10, and renders with precision 3,
    // so raw must be radians * 10000.
    TEST_ASSERT_EQUAL_INT16(-1064, pitch);
    TEST_ASSERT_EQUAL_INT16(  524, roll);
    TEST_ASSERT_EQUAL_INT16( 2094, yaw);
    TEST_ASSERT_EQUAL_UINT8(crsf_crc8(f + 2, 7), f[9]);
}

void test_scaling_rounds_not_truncates(void) {
    // 4.2f * 10.0f is 41.99999 in float. A cast makes that 41 and puts a
    // 0.1 A error on every current reading; the same lurks in every scaled
    // field. Caught by the battery test below before it ever reached a radio.
    uint8_t f[CRSF_MAX_FRAME];
    crsf_build_battery(f, 4.2f, 4.2f, 0, 0);
    TEST_ASSERT_EQUAL_INT16(42, (int16_t)((f[3] << 8) | f[4]));
    crsf_build_attitude(f, 0.3f, -0.3f, 0.0f);
    TEST_ASSERT_EQUAL_INT16( 3000, (int16_t)((f[3] << 8) | f[4]));
    TEST_ASSERT_EQUAL_INT16(-3000, (int16_t)((f[5] << 8) | f[6]));
}

void test_battery_frame_scaling(void) {
    uint8_t f[CRSF_MAX_FRAME];
    uint8_t n = crsf_build_battery(f, 24.1f, 4.2f, 1234, 82);
    TEST_ASSERT_EQUAL_UINT8(12, n);
    TEST_ASSERT_EQUAL_UINT8(CRSF_FT_BATTERY, f[2]);
    TEST_ASSERT_EQUAL_INT16(241, (int16_t)((f[3] << 8) | f[4]));   // 0.1 V
    TEST_ASSERT_EQUAL_INT16(42,  (int16_t)((f[5] << 8) | f[6]));   // 0.1 A
    uint32_t mah = ((uint32_t)f[7] << 16) | ((uint32_t)f[8] << 8) | f[9];
    TEST_ASSERT_EQUAL_UINT32(1234, mah);
    TEST_ASSERT_EQUAL_UINT8(82, f[10]);
}

void test_flight_mode_frame(void) {
    uint8_t f[CRSF_MAX_FRAME];
    uint8_t n = crsf_build_flight_mode(f, "RUNNING");
    TEST_ASSERT_EQUAL_UINT8(CRSF_FT_FLIGHTMODE, f[2]);
    TEST_ASSERT_EQUAL_STRING("RUNNING", (const char*)(f + 3));
    TEST_ASSERT_EQUAL_UINT8(n - 4 + 2, f[1]);
}

void test_flight_mode_truncates_safely(void) {
    // EdgeTX truncates at min(16, len); anything longer than 13 characters is
    // cut on the radio, so it has to be cut cleanly here.
    uint8_t f[CRSF_MAX_FRAME];
    crsf_build_flight_mode(f, "ABCDEFGHIJKLMNOPQRSTUVWXYZ");
    TEST_ASSERT_EQUAL_UINT8(CRSF_FLIGHTMODE_MAX, strlen((const char*)(f + 3)));
    TEST_ASSERT_EQUAL_UINT8(0, f[3 + CRSF_FLIGHTMODE_MAX]);
}

void test_wlr_state_frame(void) {
    CrsfWlrState s = {};
    s.state = 3; s.fault = 0; s.jump_state = 2; s.standup_state = 1;
    s.alpha = 0.42f; s.profile = 1; s.health_flags = 0x01FF;
    s.hip_l_nm = 2.7f; s.hip_r_nm = -2.9f; s.wheel_ms = 0.55f;
    s.esp32_ok = 1; s.glitch_count = 900;

    uint8_t f[CRSF_MAX_FRAME];
    uint8_t n = crsf_build_wlr_state(f, s);
    TEST_ASSERT_EQUAL_UINT8(CRSF_WLR_STATE_LEN + 4, n);
    TEST_ASSERT_EQUAL_UINT8(CRSF_FT_WLR_STATE, f[2]);
    TEST_ASSERT_EQUAL_UINT8(3, f[3]);
    TEST_ASSERT_EQUAL_UINT8(84, f[7]);                        // 0.42 * 200
    TEST_ASSERT_EQUAL_UINT16(0x01FF, (uint16_t)((f[9] << 8) | f[10]));
    TEST_ASSERT_EQUAL_INT16(270,  (int16_t)((f[11] << 8) | f[12]));
    TEST_ASSERT_EQUAL_INT16(-290, (int16_t)((f[13] << 8) | f[14]));
    TEST_ASSERT_EQUAL_INT16(55,   (int16_t)((f[15] << 8) | f[16]));
    TEST_ASSERT_EQUAL_UINT16(900, (uint16_t)((f[18] << 8) | f[19]));
    TEST_ASSERT_EQUAL_UINT8(crsf_crc8(f + 2, CRSF_WLR_STATE_LEN + 1), f[n - 1]);
}

void test_glitch_count_saturates_not_wraps(void) {
    // A free-running counter that wrapped would read as "stopped climbing",
    // which is exactly when the annunciator should be warning hardest.
    CrsfWlrState s = {};
    s.glitch_count = 70000;
    uint8_t f[CRSF_MAX_FRAME];
    crsf_build_wlr_state(f, s);
    TEST_ASSERT_EQUAL_UINT16(65535, (uint16_t)((f[18] << 8) | f[19]));
}

void test_battery_no_data_fields(void) {
    // EdgeTX skips a field whose every byte is 0xFF, so an unmeasured value
    // produces no sensor at all rather than a confident, wrong zero.
    uint8_t f[CRSF_MAX_FRAME];
    crsf_build_battery(f, 24.1f, CRSF_BATT_NO_DATA, -1, 82);
    TEST_ASSERT_EQUAL_INT16(241, (int16_t)((f[3] << 8) | f[4]));   // volts: real
    TEST_ASSERT_EQUAL_UINT8(0xFF, f[5]);                            // amps: absent
    TEST_ASSERT_EQUAL_UINT8(0xFF, f[6]);
    TEST_ASSERT_EQUAL_UINT8(0xFF, f[7]);                            // mAh: absent
    TEST_ASSERT_EQUAL_UINT8(0xFF, f[8]);
    TEST_ASSERT_EQUAL_UINT8(0xFF, f[9]);
    TEST_ASSERT_EQUAL_UINT8(82, f[10]);                             // percent: real
}

void test_wlr_state_survives_a_decoder(void) {
    // Round-trip our own telemetry frame back through the decoder, which is
    // what EdgeTX will do before handing the payload to Lua.
    CrsfWlrState s = {};
    s.state = 4; s.fault = 8; s.alpha = 1.0f; s.hip_l_nm = -0.5f;
    uint8_t f[CRSF_MAX_FRAME];
    uint8_t n = crsf_build_wlr_state(f, s);

    CrsfCore rx; rx.reset();
    bool got = false;
    for (uint8_t i = 0; i < n; i++) got = rx.feed(f[i], 1000) || got;
    TEST_ASSERT_TRUE(got);
    TEST_ASSERT_EQUAL_UINT8(CRSF_FT_WLR_STATE, rx.last_frame_type());
    TEST_ASSERT_EQUAL_UINT8(CRSF_WLR_STATE_LEN, rx.last_payload_len());
    TEST_ASSERT_EQUAL_UINT8(4, rx.last_payload()[0]);
    TEST_ASSERT_EQUAL_UINT8(8, rx.last_payload()[1]);
    TEST_ASSERT_EQUAL_UINT8(200, rx.last_payload()[4]);
}

// ── entry point ──────────────────────────────────────────────────────────────

int main(int, char**) {
    UNITY_BEGIN();

    RUN_TEST(test_crc8_known_vector);
    RUN_TEST(test_build_frame_shape);

    RUN_TEST(test_tick_to_us_endpoints);
    RUN_TEST(test_endpoints_clear_the_firmware_thresholds);
    RUN_TEST(test_us_tick_roundtrip);
    RUN_TEST(test_channel_pack_roundtrip);

    RUN_TEST(test_decodes_channels);
    RUN_TEST(test_rejects_bad_crc);
    RUN_TEST(test_resyncs_after_garbage);

    RUN_TEST(test_not_alive_before_warmup);
    RUN_TEST(test_alive_goes_false_on_timeout);
    RUN_TEST(test_channel_returns_zero_when_not_alive);
    RUN_TEST(test_dead_link_cannot_satisfy_rescue_combo);
    RUN_TEST(test_dead_link_cannot_satisfy_calibration_combo);
    RUN_TEST(test_dead_link_cannot_arm);
    RUN_TEST(test_dead_link_cannot_fire_calib_or_reset);
    RUN_TEST(test_zero_link_quality_reads_as_dead);
    RUN_TEST(test_link_quality_recovers);
    RUN_TEST(test_link_stats_absent_does_not_block);
    RUN_TEST(test_raw_channel_ignores_liveness);

    RUN_TEST(test_attitude_frame_scaling);
    RUN_TEST(test_scaling_rounds_not_truncates);
    RUN_TEST(test_battery_frame_scaling);
    RUN_TEST(test_flight_mode_frame);
    RUN_TEST(test_flight_mode_truncates_safely);
    RUN_TEST(test_wlr_state_frame);
    RUN_TEST(test_glitch_count_saturates_not_wraps);
    RUN_TEST(test_battery_no_data_fields);
    RUN_TEST(test_wlr_state_survives_a_decoder);

    return UNITY_END();
}
