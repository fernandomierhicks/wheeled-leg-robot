#pragma once
#include <stdint.h>
#include "param_ids.h"

// ── Flags ─────────────────────────────────────────────────────────────────────
#define PARAM_FLAG_PERSISTENT      (1<<0)  // written to LittleFS flash on each set
#define PARAM_FLAG_READONLY        (1<<1)  // firmware-only writes; GUI can read
#define PARAM_FLAG_COMMAND         (1<<2)  // high-freq setpoint; never persisted
#define PARAM_FLAG_FAULT_ON_BOUNDS (1<<3)  // out-of-range triggers ESTOP instead of clamp

// ── Param descriptor ──────────────────────────────────────────────────────────
struct Param {
    uint16_t    id;
    uint8_t     group_id;    // GROUP_* constant
    char        name[20];    // human-readable label for GUI / debug
    float       value;       // current value (RAM)
    float       min_val;
    float       max_val;
    uint8_t     flags;       // PARAM_FLAG_* bitmask
    void (*on_change)(float new_val);  // called after value changes; NULL if unused
};

// ── Return codes from param_set ───────────────────────────────────────────────
enum class ParamSetResult : uint8_t {
    OK          = 0,
    CLAMPED     = 1,  // value was out of range and silently clamped
    FAULT       = 2,  // FAULT_ON_BOUNDS param was out of range; caller must ESTOP
    NOT_FOUND   = 3,
    READONLY    = 4,
};

// ── API ───────────────────────────────────────────────────────────────────────

// Call once in setup() — mounts LittleFS and patches RAM table with stored values.
void param_init();

// Write a value. Clamps to [min, max] or returns FAULT if FAULT_ON_BOUNDS set.
// Calls on_change callback and persists to flash if PERSISTENT.
ParamSetResult param_set(uint16_t id, float val);

// Read current value. Returns 0.0f if id not found.
float param_get(uint16_t id);

// Returns true if the param ID is registered.
bool param_exists(uint16_t id);

// Total number of registered params.
uint16_t param_count();

// Fill buf[] with up to max descriptors from the given group.
// Returns number of entries written.
uint16_t param_get_group(uint8_t group_id, Param* buf, uint16_t max);

// Get param descriptor by registry index (0..param_count()-1).
// Returns false if idx is out of range.
bool param_by_index(uint16_t idx, Param* out);

// Force flush all PERSISTENT params to flash.
void param_save_all();

// Erase flash and revert all params to compile-time defaults.
void param_reset_defaults();

// Firmware-internal write: updates value even for PARAM_FLAG_READONLY params.
// Do NOT call from command handlers — READONLY exists to block GUI writes.
void param_force_set(uint16_t id, float val);
