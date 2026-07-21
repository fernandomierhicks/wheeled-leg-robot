#pragma once

#include <math.h>
#include <stdint.h>
#include <string.h>
#include "comm_protocol.h"

struct ValidatedCommand {
    uint32_t request_id;
    const uint8_t* bytes;  // begins with CMD_ID_*
    uint16_t len;
    uint8_t reason;
};

static inline bool command_float_is_finite(const uint8_t* p) {
    float value;
    memcpy(&value, p, sizeof(value));
    return __builtin_isfinite(value);
}

// Structural validation shared by firmware and host-native tests. State,
// parameter existence/permissions, and peripheral readiness are checked by
// the Teensy admission layer after this function succeeds.
static inline bool validate_command_payload(uint8_t version, const uint8_t* payload,
                                            uint16_t len, ValidatedCommand* out) {
    out->request_id = 0;
    out->bytes = payload;
    out->len = len;
    out->reason = CMD_REASON_NONE;

    if (version == CMD_PAYLOAD_V2) {
        if (len < 5) { out->reason = CMD_REASON_BAD_LENGTH; return false; }
        memcpy(&out->request_id, payload, sizeof(out->request_id));
        if (out->request_id == 0) { out->reason = CMD_REASON_INVALID_ENUM; return false; }
        out->bytes = payload + 4;
        out->len = len - 4;
    } else if (version != CMD_PAYLOAD_V1) {
        out->reason = CMD_REASON_BAD_VERSION;
        return false;
    }

    if (out->len < 1) { out->reason = CMD_REASON_BAD_LENGTH; return false; }
    const uint8_t* p = out->bytes;
    const uint16_t n = out->len;
    switch (p[0]) {
        case CMD_ID_SET_MODE:
            if (n != 2) { out->reason = CMD_REASON_BAD_LENGTH; return false; }
            if (p[1] > 9) { out->reason = CMD_REASON_INVALID_TARGET; return false; }
            return true;
        case CMD_ID_PING:
        case CMD_ID_REBOOT:
        case CMD_ID_PARAM_RESET_DEFAULTS:
            if (n != 1) { out->reason = CMD_REASON_BAD_LENGTH; return false; }
            return true;
        case CMD_ID_HIP: {
            if (n < 3) { out->reason = CMD_REASON_BAD_LENGTH; return false; }
            if (p[1] > HIP_MOTOR_R || p[2] > HIP_SUB_MIT) {
                out->reason = CMD_REASON_INVALID_ENUM; return false;
            }
            const uint16_t expected = p[2] == HIP_SUB_MIT ? 23 : 3;
            if (n != expected) { out->reason = CMD_REASON_BAD_LENGTH; return false; }
            if (p[2] == HIP_SUB_MIT) {
                for (uint8_t offset = 3; offset < 23; offset += 4) {
                    if (!command_float_is_finite(p + offset)) {
                        out->reason = CMD_REASON_NONFINITE; return false;
                    }
                }
            }
            return true;
        }
        case CMD_ID_WHEEL:
            if (n < 2) { out->reason = CMD_REASON_BAD_LENGTH; return false; }
            if (p[1] == WHEEL_SUB_SET_MODE) {
                if (n != 3) { out->reason = CMD_REASON_BAD_LENGTH; return false; }
                if (p[2] > 3) { out->reason = CMD_REASON_INVALID_ENUM; return false; }
                return true;
            }
            if (p[1] == WHEEL_SUB_SEND) {
                if (n != 10) { out->reason = CMD_REASON_BAD_LENGTH; return false; }
                if (!command_float_is_finite(p + 2) || !command_float_is_finite(p + 6)) {
                    out->reason = CMD_REASON_NONFINITE; return false;
                }
                return true;
            }
            if (p[1] == WHEEL_SUB_CLEAR_ERRORS) {
                if (n != 2) { out->reason = CMD_REASON_BAD_LENGTH; return false; }
                return true;
            }
            out->reason = CMD_REASON_INVALID_ENUM; return false;
        case CMD_ID_SET_TELEM_TRANSPORT:
            if (n != 2) { out->reason = CMD_REASON_BAD_LENGTH; return false; }
            if (p[1] > 1) { out->reason = CMD_REASON_INVALID_ENUM; return false; }
            return true;
        case CMD_ID_PARAM_SET:
            if (n != 7) { out->reason = CMD_REASON_BAD_LENGTH; return false; }
            if (!command_float_is_finite(p + 3)) { out->reason = CMD_REASON_NONFINITE; return false; }
            return true;
        case CMD_ID_PARAM_GET:
            if (n != 3) { out->reason = CMD_REASON_BAD_LENGTH; return false; }
            return true;
        case CMD_ID_TEST_INJECT_CORRUPT:
            if (n != 4) { out->reason = CMD_REASON_BAD_LENGTH; return false; }
            if (p[2] > 1 || p[3] < 1 || p[3] > 3) {
                out->reason = CMD_REASON_INVALID_ENUM; return false;
            }
            return true;
        case CMD_ID_LOG:
            if (n < 2) { out->reason = CMD_REASON_BAD_LENGTH; return false; }
            switch (p[1]) {
                case LOG_SUB_START:     if (n == 6) return true; break;
                case LOG_SUB_STOP:
                case LOG_SUB_LIST:      if (n == 2) return true; break;
                case LOG_SUB_GET:
                case LOG_SUB_CHUNK_ACK: if (n == 8) return true; break;
                case LOG_SUB_DELETE:    if (n == 4) return true; break;
                default: out->reason = CMD_REASON_INVALID_ENUM; return false;
            }
            out->reason = CMD_REASON_BAD_LENGTH; return false;
        default:
            out->reason = CMD_REASON_UNKNOWN_COMMAND;
            return false;
    }
}
