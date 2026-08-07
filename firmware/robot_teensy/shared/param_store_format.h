#pragma once
#include <stddef.h>
#include <stdint.h>

static constexpr uint32_t PARAM_STORE_MAGIC = 0x32505257u;  // "WRP2" little-endian
// v3: no format change — the *units* of a few hip params changed when the
// AK45-10 MIT torque/velocity scale factors were corrected (see
// hip_motors.cpp). A v2 slot is still readable, and param_registry's
// migrate_hip_scale_v2_to_v3() rescales exactly those params once on load and
// re-saves as v3, so the rest of a tuned setup survives untouched.
static constexpr uint16_t PARAM_STORE_VERSION = 3;
static constexpr uint16_t PARAM_STORE_VERSION_MIN = 2;

// A stored slot is readable if its version is one we still know how to migrate
// forward from. Anything older/newer is rejected and falls back to defaults.
inline bool param_store_version_supported(uint16_t version) {
    return version >= PARAM_STORE_VERSION_MIN && version <= PARAM_STORE_VERSION;
}

struct __attribute__((packed)) ParamStoreEntry {
    uint16_t id;
    float value;
};

struct __attribute__((packed)) ParamStoreHeader {
    uint32_t magic;
    uint16_t version;
    uint16_t header_bytes;
    uint32_t generation;
    uint16_t count;
    uint16_t payload_bytes;
    uint32_t payload_crc32;
    uint32_t header_crc32;
};

static_assert(sizeof(ParamStoreEntry) == 6, "ParamStoreEntry wire size");
static_assert(sizeof(ParamStoreHeader) == 24, "ParamStoreHeader wire size");

inline uint32_t param_store_crc32(const void* data, size_t len) {
    const uint8_t* bytes = static_cast<const uint8_t*>(data);
    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0; i < len; ++i) {
        crc ^= bytes[i];
        for (uint8_t bit = 0; bit < 8; ++bit)
            crc = (crc >> 1) ^ (0xEDB88320u & (uint32_t)-(int32_t)(crc & 1u));
    }
    return ~crc;
}

inline uint32_t param_store_header_crc(ParamStoreHeader header) {
    header.header_crc32 = 0;
    return param_store_crc32(&header, sizeof(header));
}

inline bool param_store_header_valid(const ParamStoreHeader& header) {
    if (header.magic != PARAM_STORE_MAGIC ||
        !param_store_version_supported(header.version) ||
        header.header_bytes != sizeof(ParamStoreHeader) ||
        header.payload_bytes != (uint16_t)(header.count * sizeof(ParamStoreEntry)))
        return false;
    return param_store_header_crc(header) == header.header_crc32;
}

inline bool param_store_generation_newer(uint32_t candidate, uint32_t current) {
    return (int32_t)(candidate - current) > 0;
}
