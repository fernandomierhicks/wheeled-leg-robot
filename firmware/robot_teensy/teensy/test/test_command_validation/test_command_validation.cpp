#include <unity.h>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>
#include "command_validation.h"

void setUp(void) {}
void tearDown(void) {}

static ValidatedCommand validate(const std::vector<uint8_t>& bytes, uint8_t version,
                                 bool expected) {
    ValidatedCommand out{};
    TEST_ASSERT_EQUAL(expected, validate_command_payload(version, bytes.data(), bytes.size(), &out));
    return out;
}

void test_v2_envelope_preserves_request_and_command(void) {
    std::vector<uint8_t> bytes = {0x78, 0x56, 0x34, 0x12, CMD_ID_SET_MODE, 3};
    auto out = validate(bytes, CMD_PAYLOAD_V2, true);
    TEST_ASSERT_EQUAL_HEX32(0x12345678, out.request_id);
    TEST_ASSERT_EQUAL_UINT8(CMD_ID_SET_MODE, out.bytes[0]);
    TEST_ASSERT_EQUAL_UINT16(2, out.len);
}

void test_every_fixed_command_rejects_trailing_and_missing_bytes(void) {
    const std::vector<std::vector<uint8_t>> valid = {
        {CMD_ID_SET_MODE, 3}, {CMD_ID_PING}, {CMD_ID_REBOOT},
        {CMD_ID_PARAM_GET, 0, 0}, {CMD_ID_PARAM_RESET_DEFAULTS},
        {CMD_ID_WHEEL, WHEEL_SUB_SET_MODE, 3},
        {CMD_ID_WHEEL, WHEEL_SUB_CLEAR_ERRORS},
        {CMD_ID_LOG, LOG_SUB_STOP}, {CMD_ID_LOG, LOG_SUB_LIST},
        {CMD_ID_LOG, LOG_SUB_DELETE, 0, 0},
    };
    for (auto command : valid) {
        validate(command, CMD_PAYLOAD_V1, true);
        command.push_back(0xA5);
        auto trailing = validate(command, CMD_PAYLOAD_V1, false);
        TEST_ASSERT_EQUAL_UINT8(CMD_REASON_BAD_LENGTH, trailing.reason);
    }
}

void test_nonfinite_values_are_rejected_atomically(void) {
    float values[] = {
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
    };
    for (float value : values) {
        std::vector<uint8_t> param = {CMD_ID_PARAM_SET, 1, 0, 0, 0, 0, 0};
        memcpy(param.data() + 3, &value, 4);
        auto out = validate(param, CMD_PAYLOAD_V1, false);
        TEST_ASSERT_EQUAL_UINT8(CMD_REASON_NONFINITE, out.reason);

        std::vector<uint8_t> wheel(10, 0);
        wheel[0] = CMD_ID_WHEEL;
        wheel[1] = WHEEL_SUB_SEND;
        memcpy(wheel.data() + 2, &value, 4);
        out = validate(wheel, CMD_PAYLOAD_V1, false);
        TEST_ASSERT_EQUAL_UINT8(CMD_REASON_NONFINITE, out.reason);

        std::vector<uint8_t> hip(23, 0);
        hip[0] = CMD_ID_HIP;
        hip[1] = HIP_MOTOR_L;
        hip[2] = HIP_SUB_MIT;
        memcpy(hip.data() + 3, &value, 4);
        out = validate(hip, CMD_PAYLOAD_V1, false);
        TEST_ASSERT_EQUAL_UINT8(CMD_REASON_NONFINITE, out.reason);
    }
}

void test_log_sub_get_kind_byte_is_optional_and_bounds_checked(void) {
    // 8 bytes: kind omitted, defaults to LOG_FILE_KIND_WLOG.
    validate({CMD_ID_LOG, LOG_SUB_GET, 0, 0, 0, 0, 0, 0}, CMD_PAYLOAD_V1, true);
    // 9 bytes: explicit valid kind (LOG_FILE_KIND_WLOG or LOG_FILE_KIND_PARAMS).
    validate({CMD_ID_LOG, LOG_SUB_GET, 0, 0, 0, 0, 0, 0, LOG_FILE_KIND_WLOG},
              CMD_PAYLOAD_V1, true);
    validate({CMD_ID_LOG, LOG_SUB_GET, 0, 0, 0, 0, 0, 0, LOG_FILE_KIND_PARAMS},
              CMD_PAYLOAD_V1, true);
    // 9 bytes: out-of-range kind.
    auto out = validate({CMD_ID_LOG, LOG_SUB_GET, 0, 0, 0, 0, 0, 0, 2}, CMD_PAYLOAD_V1, false);
    TEST_ASSERT_EQUAL_UINT8(CMD_REASON_INVALID_ENUM, out.reason);
    // Neither 8 nor 9 bytes.
    out = validate({CMD_ID_LOG, LOG_SUB_GET, 0, 0, 0, 0, 0}, CMD_PAYLOAD_V1, false);
    TEST_ASSERT_EQUAL_UINT8(CMD_REASON_BAD_LENGTH, out.reason);
    out = validate({CMD_ID_LOG, LOG_SUB_GET, 0, 0, 0, 0, 0, 0, 0, 0}, CMD_PAYLOAD_V1, false);
    TEST_ASSERT_EQUAL_UINT8(CMD_REASON_BAD_LENGTH, out.reason);
}

void test_invalid_ids_enums_versions_and_targets_reject(void) {
    auto out = validate({0xFE}, CMD_PAYLOAD_V1, false);
    TEST_ASSERT_EQUAL_UINT8(CMD_REASON_UNKNOWN_COMMAND, out.reason);
    out = validate({CMD_ID_SET_MODE, 10}, CMD_PAYLOAD_V1, false);
    TEST_ASSERT_EQUAL_UINT8(CMD_REASON_INVALID_TARGET, out.reason);
    out = validate({CMD_ID_WHEEL, WHEEL_SUB_SET_MODE, 4}, CMD_PAYLOAD_V1, false);
    TEST_ASSERT_EQUAL_UINT8(CMD_REASON_INVALID_ENUM, out.reason);
    out = validate({CMD_ID_PING}, 99, false);
    TEST_ASSERT_EQUAL_UINT8(CMD_REASON_BAD_VERSION, out.reason);
}

int main(int argc, char** argv) {
    (void)argc; (void)argv;
    UNITY_BEGIN();
    RUN_TEST(test_v2_envelope_preserves_request_and_command);
    RUN_TEST(test_every_fixed_command_rejects_trailing_and_missing_bytes);
    RUN_TEST(test_nonfinite_values_are_rejected_atomically);
    RUN_TEST(test_log_sub_get_kind_byte_is_optional_and_bounds_checked);
    RUN_TEST(test_invalid_ids_enums_versions_and_targets_reject);
    return UNITY_END();
}
