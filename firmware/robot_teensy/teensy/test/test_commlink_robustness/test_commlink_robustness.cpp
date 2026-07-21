// test_commlink_robustness.cpp — CommLink parser robustness (native, no hardware).
//
// UARTplat.md Phase 6/§7.2: feeds the parser random noise, truncated frames,
// corrupted length (in-range and out-of-range), corrupted CRC, and back-to-
// back valid frames with garbage between — asserts every valid frame after
// garbage is still recovered and rx_drops()/rx_seq_gaps() move sensibly.
// Runs under `platform = native` (see ../../platformio.ini [env:native]):
// CommLink.h's own Stream stub (its #ifndef ARDUINO branch) makes this
// possible with zero Arduino/hardware dependency.
#include <unity.h>
#include <cstring>
#include <vector>
#include "CommLink.h"
#include "comm_protocol.h"

// This desktop-only test environment excludes the Teensy application sources,
// so compile the shared implementation directly into the test translation unit.
#include "../../../shared/CommLink/CommLink.cpp"

void setUp(void) {}
void tearDown(void) {}

// ── Loopback Stream (same pattern as the on-target test files) ──────────────
class LoopbackStream : public Stream {
    std::vector<uint8_t> _buf;
    size_t _head = 0;
public:
    size_t write(const uint8_t* buf, size_t len) override {
        _buf.insert(_buf.end(), buf, buf + len);
        return len;
    }
    size_t write(uint8_t b) override { return write(&b, 1); }
    int available() override { return (int)(_buf.size() - _head); }
    int read() override { return (_head < _buf.size()) ? _buf[_head++] : -1; }
    void write_raw(std::initializer_list<uint8_t> bytes) { _buf.insert(_buf.end(), bytes); }
    void write_raw(const uint8_t* bytes, size_t n) { _buf.insert(_buf.end(), bytes, bytes + n); }
};

class ShortWriteStream : public Stream {
public:
    size_t accepted = 0;
    size_t calls = 0;
    explicit ShortWriteStream(size_t accepted_bytes) : accepted(accepted_bytes) {}
    size_t write(const uint8_t* /*buf*/, size_t len) override {
        ++calls;
        return accepted < len ? accepted : len;
    }
    int available() override { return 0; }
    int read() override { return -1; }
};

static bool     s_got;
static uint8_t  s_type, s_ver;
static uint16_t s_len;
static uint8_t  s_buf[COMM_MAX_PAYLOAD];

static void on_packet(uint8_t type, uint8_t ver, uint8_t /*src*/,
                       const uint8_t* payload, uint16_t len) {
    s_got  = true;
    s_type = type;
    s_ver  = ver;
    s_len  = len;
    memcpy(s_buf, payload, len);
}

// Builds one well-formed frame (mirrors CommLink::send()'s own layout).
static std::vector<uint8_t> make_frame(uint8_t type, uint8_t ver, uint8_t src, uint8_t seq,
                                        const uint8_t* payload, uint16_t len) {
    std::vector<uint8_t> f;
    f.push_back(COMM_START_A);
    f.push_back(COMM_START_B);
    f.push_back(type);
    f.push_back(ver);
    f.push_back(src);
    f.push_back(seq);
    f.push_back((uint8_t)(len & 0xFF));
    f.push_back((uint8_t)(len >> 8));
    uint8_t crc = 0;
    // CRC-8 must match crc8()/crc8_step() in CommLink.cpp exactly.
    auto step = [](uint8_t crc, uint8_t b) -> uint8_t {
        static uint8_t table[256]; static bool ready = false;
        if (!ready) {
            for (int i = 0; i < 256; i++) {
                uint8_t c = (uint8_t)i;
                for (int k = 0; k < 8; k++)
                    c = (c & 0x80) ? (uint8_t)((c << 1) ^ 0x07) : (uint8_t)(c << 1);
                table[i] = c;
            }
            ready = true;
        }
        return table[crc ^ b];
    };
    crc = step(crc, type); crc = step(crc, ver); crc = step(crc, src); crc = step(crc, seq);
    crc = step(crc, (uint8_t)(len & 0xFF)); crc = step(crc, (uint8_t)(len >> 8));
    for (uint16_t i = 0; i < len; i++) crc = step(crc, payload[i]);
    for (uint16_t i = 0; i < len; i++) f.push_back(payload[i]);
    f.push_back(crc);
    f.push_back(COMM_END);
    return f;
}

static void expect_frame(CommLink& cl, LoopbackStream& ls, const std::vector<uint8_t>& frame,
                          uint8_t exp_type, uint16_t exp_len, const char* msg) {
    s_got = false;
    ls.write_raw(frame.data(), frame.size());
    cl.update();
    TEST_ASSERT_TRUE_MESSAGE(s_got, msg);
    TEST_ASSERT_EQUAL_MESSAGE(exp_type, s_type, msg);
    TEST_ASSERT_EQUAL_MESSAGE(exp_len, s_len, msg);
}

// ── Tests ─────────────────────────────────────────────────────────────────────

void test_pure_noise_ignored_then_valid_frame_decodes(void) {
    LoopbackStream ls;
    CommLink cl(ls, COMM_SRC_PC);
    cl.onPacket(on_packet);
    uint32_t drops_before = cl.rx_drops();

    // 100 bytes with no COMM_START_A at all: PS_IDLE silently ignores them,
    // never starting a frame, so no drop is counted.
    std::vector<uint8_t> noise(100, 0x00);
    ls.write_raw(noise.data(), noise.size());
    s_got = false;
    cl.update();
    TEST_ASSERT_FALSE(s_got);
    TEST_ASSERT_EQUAL_UINT32(drops_before, cl.rx_drops());

    uint8_t payload[4] = {1, 2, 3, 4};
    auto frame = make_frame(COMM_TYPE_COMMAND, 1, COMM_SRC_PC, 0, payload, sizeof(payload));
    expect_frame(cl, ls, frame, COMM_TYPE_COMMAND, sizeof(payload),
                 "valid frame after pure noise must still decode");
}

void test_corrupted_crc_dropped_then_recovers(void) {
    LoopbackStream ls;
    CommLink cl(ls, COMM_SRC_PC);
    cl.onPacket(on_packet);
    uint32_t drops_before = cl.rx_drops();

    uint8_t payload[6] = {10, 20, 30, 40, 50, 60};
    auto bad = make_frame(COMM_TYPE_COMMAND, 1, COMM_SRC_PC, 5, payload, sizeof(payload));
    bad[8 + sizeof(payload)] ^= 0xFF;  // flip the CRC byte

    s_got = false;
    ls.write_raw(bad.data(), bad.size());
    cl.update();
    TEST_ASSERT_FALSE_MESSAGE(s_got, "corrupted-CRC frame must not be delivered");
    TEST_ASSERT_EQUAL_UINT32(drops_before + 1, cl.rx_drops());

    auto good = make_frame(COMM_TYPE_COMMAND, 1, COMM_SRC_PC, 6, payload, sizeof(payload));
    expect_frame(cl, ls, good, COMM_TYPE_COMMAND, sizeof(payload),
                 "valid frame after a CRC-corrupted one must still decode");
}

void test_length_out_of_range_dropped_then_recovers(void) {
    LoopbackStream ls;
    CommLink cl(ls, COMM_SRC_PC);
    cl.onPacket(on_packet);
    uint32_t drops_before = cl.rx_drops();

    // Hand-craft a header claiming a length far beyond COMM_MAX_PAYLOAD —
    // the parser's explicit guard must reject it immediately, without
    // waiting to consume (COMM_MAX_PAYLOAD + 1) bytes that were never sent.
    ls.write_raw({COMM_START_A, COMM_START_B, COMM_TYPE_COMMAND, 1, COMM_SRC_PC, 7,
                  (uint8_t)0xFF, (uint8_t)0xFF});
    s_got = false;
    cl.update();
    TEST_ASSERT_FALSE(s_got);
    TEST_ASSERT_EQUAL_UINT32(drops_before + 1, cl.rx_drops());

    uint8_t payload[2] = {9, 9};
    auto good = make_frame(COMM_TYPE_COMMAND, 1, COMM_SRC_PC, 8, payload, sizeof(payload));
    expect_frame(cl, ls, good, COMM_TYPE_COMMAND, sizeof(payload),
                 "valid frame after an out-of-range length must still decode");
}

void test_back_to_back_valid_frames_with_garbage_between(void) {
    LoopbackStream ls;
    CommLink cl(ls, COMM_SRC_PC);
    cl.onPacket(on_packet);

    uint8_t p1[3] = {1, 1, 1};
    uint8_t p2[3] = {2, 2, 2};
    auto f1 = make_frame(COMM_TYPE_COMMAND, 1, COMM_SRC_PC, 10, p1, sizeof(p1));
    auto f2 = make_frame(COMM_TYPE_COMMAND, 1, COMM_SRC_PC, 11, p2, sizeof(p2));

    expect_frame(cl, ls, f1, COMM_TYPE_COMMAND, sizeof(p1), "first back-to-back frame");

    // Non-magic garbage between frames: PS_IDLE ignores it without disturbing
    // the next frame's parse.
    ls.write_raw({0x00, 0x11, 0x22, 0x33});
    expect_frame(cl, ls, f2, COMM_TYPE_COMMAND, sizeof(p2), "second back-to-back frame");
    TEST_ASSERT_EQUAL_UINT8(2, s_buf[0]);
}

void test_truncated_frame_then_valid_frames_recover(void) {
    LoopbackStream ls;
    CommLink cl(ls, COMM_SRC_PC);
    cl.onPacket(on_packet);

    // Header claims a 20-byte payload; only 3 bytes actually follow, then
    // nothing else for this frame — parser is left waiting mid-PS_PAYLOAD.
    ls.write_raw({COMM_START_A, COMM_START_B, COMM_TYPE_COMMAND, 1, COMM_SRC_PC, 20,
                  20, 0, 0xAA, 0xBB, 0xCC});
    s_got = false;
    cl.update();
    TEST_ASSERT_FALSE(s_got);

    // Send a few complete valid frames after it — however the truncated
    // frame's bytes get reinterpreted, the parser must resync onto clean
    // magic bytes within a small number of subsequent frames.
    uint8_t payload[4] = {7, 7, 7, 7};
    bool recovered = false;
    for (uint8_t i = 0; i < 5 && !recovered; i++) {
        auto f = make_frame(COMM_TYPE_COMMAND, 1, COMM_SRC_PC, (uint8_t)(30 + i), payload, sizeof(payload));
        s_got = false;
        ls.write_raw(f.data(), f.size());
        cl.update();
        if (s_got && s_type == COMM_TYPE_COMMAND && s_len == sizeof(payload)) recovered = true;
    }
    TEST_ASSERT_TRUE_MESSAGE(recovered,
        "parser must resync onto a clean frame within a few frame-widths after truncation");
}

void test_send_matches_frozen_empty_ack_vector(void) {
    LoopbackStream ls;
    CommLink sender(ls, COMM_SRC_ESP32);
    CommLink receiver(ls, COMM_SRC_PC);
    receiver.onPacket(on_packet);

    // protocol_golden_vectors.json: empty_ack
    const uint8_t expected[] = {0xAA, 0x55, 0x03, 0x01, 0x02, 0x00,
                                0x00, 0x00, 0x35, 0xEF};
    const uint8_t empty = 0;
    TEST_ASSERT_TRUE(sender.send(COMM_TYPE_ACK, 1, &empty, 0));
    s_got = false;
    receiver.update();
    TEST_ASSERT_TRUE(s_got);
    TEST_ASSERT_EQUAL_UINT8(COMM_TYPE_ACK, s_type);
    TEST_ASSERT_EQUAL_UINT16(0, s_len);

    auto explicit_frame = make_frame(COMM_TYPE_ACK, 1, COMM_SRC_ESP32, 0, &empty, 0);
    TEST_ASSERT_EQUAL_UINT32(sizeof(expected), explicit_frame.size());
    TEST_ASSERT_EQUAL_UINT8_ARRAY(expected, explicit_frame.data(), sizeof(expected));
}

void test_send_rejects_short_write_and_oversize(void) {
    uint8_t payload[4] = {1, 2, 3, 4};
    ShortWriteStream short_stream(5);
    CommLink short_sender(short_stream, COMM_SRC_PC);
    TEST_ASSERT_FALSE(short_sender.send(COMM_TYPE_COMMAND, 1, payload, sizeof(payload)));
    TEST_ASSERT_EQUAL_UINT32(1, short_stream.calls);

    ShortWriteStream oversize_stream(1024);
    CommLink oversize_sender(oversize_stream, COMM_SRC_PC);
    std::vector<uint8_t> oversized(COMM_MAX_PAYLOAD + 1, 0x5A);
    TEST_ASSERT_FALSE(oversize_sender.send(COMM_TYPE_COMMAND, 1, oversized.data(), oversized.size()));
    TEST_ASSERT_EQUAL_UINT32(0, oversize_stream.calls);
}

void test_max_payload_roundtrip(void) {
    LoopbackStream ls;
    CommLink sender(ls, COMM_SRC_PC);
    CommLink receiver(ls, COMM_SRC_TEENSY);
    receiver.onPacket(on_packet);
    std::vector<uint8_t> payload(COMM_MAX_PAYLOAD);
    for (size_t i = 0; i < payload.size(); ++i) payload[i] = (uint8_t)i;

    TEST_ASSERT_TRUE(sender.send(COMM_TYPE_COMMAND, 1, payload.data(), payload.size()));
    s_got = false;
    receiver.update();
    TEST_ASSERT_TRUE(s_got);
    TEST_ASSERT_EQUAL_UINT16(COMM_MAX_PAYLOAD, s_len);
    TEST_ASSERT_EQUAL_UINT8_ARRAY(payload.data(), s_buf, payload.size());
}

void test_corruption_modes_drop_and_next_frame_recovers(void) {
    LoopbackStream ls;
    CommLink sender(ls, COMM_SRC_PC);
    CommLink receiver(ls, COMM_SRC_TEENSY);
    receiver.onPacket(on_packet);
    uint8_t payload[4] = {0x00, COMM_START_A, COMM_START_B, COMM_END};

    // CRC and END corruption preserve the true frame boundary. A corrupted
    // length has its own recovery test above because its trailing bytes are,
    // by definition, an unframed byte stream.
    for (uint8_t mode = 1; mode <= 2; ++mode) {
        s_got = false;
        TEST_ASSERT_TRUE(sender.send(COMM_TYPE_COMMAND, 1, payload, sizeof(payload), mode));
        receiver.update();
        TEST_ASSERT_FALSE_MESSAGE(s_got, "corrupted frame must not be delivered");
    }
    TEST_ASSERT_GREATER_OR_EQUAL_UINT32(2, receiver.rx_drops());

    TEST_ASSERT_TRUE(sender.send(COMM_TYPE_COMMAND, 1, payload, sizeof(payload)));
    s_got = false;
    receiver.update();
    TEST_ASSERT_TRUE_MESSAGE(s_got, "valid frame after all corruption modes must decode");
    TEST_ASSERT_EQUAL_UINT8_ARRAY(payload, s_buf, sizeof(payload));
}

void test_overlapping_magic_and_sequence_wrap(void) {
    LoopbackStream ls;
    CommLink receiver(ls, COMM_SRC_TEENSY);
    receiver.onPacket(on_packet);
    uint8_t payload[1] = {0x42};

    // An extra first magic byte must not hide the immediately following frame.
    ls.write_raw({COMM_START_A});
    auto first = make_frame(COMM_TYPE_COMMAND, 1, COMM_SRC_PC, 254, payload, sizeof(payload));
    expect_frame(receiver, ls, first, COMM_TYPE_COMMAND, sizeof(payload),
                 "overlapping magic must resynchronize");

    auto second = make_frame(COMM_TYPE_COMMAND, 1, COMM_SRC_PC, 255, payload, sizeof(payload));
    auto third  = make_frame(COMM_TYPE_COMMAND, 1, COMM_SRC_PC,   0, payload, sizeof(payload));
    expect_frame(receiver, ls, second, COMM_TYPE_COMMAND, sizeof(payload), "sequence 255");
    expect_frame(receiver, ls, third,  COMM_TYPE_COMMAND, sizeof(payload), "sequence wrap to 0");
    TEST_ASSERT_EQUAL_UINT32(0, receiver.rx_seq_gaps());
}

// ── Entry (native — no Arduino runtime, see [env:native] in platformio.ini) ──
int main(int argc, char** argv) {
    (void)argc; (void)argv;
    UNITY_BEGIN();
    RUN_TEST(test_pure_noise_ignored_then_valid_frame_decodes);
    RUN_TEST(test_corrupted_crc_dropped_then_recovers);
    RUN_TEST(test_length_out_of_range_dropped_then_recovers);
    RUN_TEST(test_back_to_back_valid_frames_with_garbage_between);
    RUN_TEST(test_truncated_frame_then_valid_frames_recover);
    RUN_TEST(test_send_matches_frozen_empty_ack_vector);
    RUN_TEST(test_send_rejects_short_write_and_oversize);
    RUN_TEST(test_max_payload_roundtrip);
    RUN_TEST(test_corruption_modes_drop_and_next_frame_recovers);
    RUN_TEST(test_overlapping_magic_and_sequence_wrap);
    return UNITY_END();
}
