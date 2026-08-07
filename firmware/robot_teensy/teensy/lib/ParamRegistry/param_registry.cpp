#include "param_registry.h"
#include <math.h>
#include "comm_protocol.h"
#include "param_store_format.h"
#include <LittleFS.h>

// ── Flash filesystem ──────────────────────────────────────────────────────────
// This board has no onboard QSPI NOR/NAND chip (confirmed on hardware — begin()
// always failed), so params live in a carved-out slice of the Teensy's own
// program flash instead. No extra hardware needed.
// Must be >= 2 sectors — LittleFS stores its metadata as a redundant pair and
// won't format a 1-block volume. Teensy 4.1's LittleFS_Program sector size is
// 64 KiB (SECTOR_SIZE in the framework's LittleFS.cpp), so the minimum viable
// size is 128 KiB; 256 KiB (4 sectors) leaves headroom and is still negligible
// next to the 7.75 MB available.
static LittleFS_Program s_fs;
static const uint32_t FS_SIZE_BYTES = 256 * 1024;
static const char* PARAM_SLOT_A = "/params_a.bin";
static const char* PARAM_SLOT_B = "/params_b.bin";
static const char* PARAM_LEGACY = "/params.bin";
static const uint16_t LEGACY_MAGIC = 0xB0B1;
static const uint8_t LEGACY_VERSION = 1;
// params.bin predates the versioned CRC slots; treat it as older than the
// oldest slot version so every unit migration applies to it.
static const uint16_t LEGACY_STORE_VERSION = PARAM_STORE_VERSION_MIN - 1;
static uint32_t s_store_generation = 0;
static uint8_t s_store_active_slot = 0xFF;
static ParamPersistenceStatus s_store_status{};
static bool s_legacy_loaded = false;

// ── Deferred flash flush ──────────────────────────────────────────────────────
// save_to_flash() is a full LittleFS remove+rewrite — far too slow to run
// inside the 500 Hz loop (and the CH9 profile torque slew calls param_set()
// every tick while ramping). PERSISTENT writes therefore only mark a dirty
// flag; param_flush_service() performs the actual write once writes have been
// quiet for FLUSH_QUIET_MS and the caller allows it (not RUNNING/JUMPING).
static bool           s_dirty          = false;
static uint32_t       s_last_change_ms = 0;
static const uint32_t FLUSH_QUIET_MS   = 1000;

// ── Registry table ────────────────────────────────────────────────────────────
// Generated from protocol/schema.json. Edit the schema, then run
// protocol/generate_protocol.py; params_tab.py only owns optional GUI subgroup
// placement.
// clang-format off
static Param g_params[] = {
#include "generated_param_table.inc"
};
// clang-format on

static const uint16_t PARAM_COUNT = sizeof(g_params) / sizeof(g_params[0]);

// Compile-time defaults, captured once in param_init() before load_from_flash()
// overwrites g_params[].value — the only record of "default" once flash restore
// or a GUI param_set() has touched a value. Sized/populated at init, not
// statically initialized, since it must mirror g_params[] exactly.
static float s_defaults[PARAM_COUNT];

// ── Internal helpers ──────────────────────────────────────────────────────────

static float migration_scale(uint16_t id, uint16_t from_version);

static Param* find(uint16_t id) {
    for (uint16_t i = 0; i < PARAM_COUNT; i++) {
        if (g_params[i].id == id) return &g_params[i];
    }
    return nullptr;
}

static void load_legacy_from_flash() {
    File f = s_fs.open(PARAM_LEGACY, FILE_READ);
    if (!f) {
        comm_log(LOG_LEVEL_WARN, "Param flash: no params.bin (first boot, or flash unmounted)");
        return;
    }

    uint16_t magic;
    uint8_t  ver;
    uint16_t count;
    if (f.read(&magic, 2) != 2 || magic != LEGACY_MAGIC) {
        comm_log(LOG_LEVEL_WARN, "Param flash: bad magic in params.bin, ignoring");
        f.close(); return;
    }
    if (f.read(&ver,   1) != 1 || ver   != LEGACY_VERSION) {
        comm_log(LOG_LEVEL_WARN, "Param flash: version mismatch in params.bin, ignoring");
        f.close(); return;
    }
    if (f.read(&count, 2) != 2) { f.close(); return; }

    uint16_t restored = 0;
    for (uint16_t i = 0; i < count; i++) {
        uint16_t id;
        float    val;
        if (f.read(&id,  2) != 2) break;
        if (f.read(&val, 4) != 4) break;
        Param* p = find(id);
        if (p && !(p->flags & PARAM_FLAG_READONLY)) {
            // params.bin predates the CRC slots entirely, so it is always in
            // the pre-v3 hip units — convert before clamping (see
            // migration_scale()).
            val *= migration_scale(p->id, LEGACY_STORE_VERSION);
            // clamp to current bounds before restoring
            if (val < p->min_val) val = p->min_val;
            if (val > p->max_val) val = p->max_val;
            p->value = val;
            restored++;
        }
    }
    f.close();
    s_legacy_loaded = restored > 0;
    comm_log(LOG_LEVEL_INFO, "Param flash: restored %u/%u params", restored, count);
}

struct SlotInfo {
    bool valid;
    uint32_t generation;
    uint16_t count;
    uint16_t version;
};

static SlotInfo inspect_slot(const char* path, ParamStoreEntry* entries = nullptr) {
    SlotInfo info{};
    File f = s_fs.open(path, FILE_READ);
    if (!f) return info;
    ParamStoreHeader header{};
    if (f.read(&header, sizeof(header)) != sizeof(header) ||
        !param_store_header_valid(header) || header.count > PARAM_COUNT ||
        f.size() != (size_t)(sizeof(header) + header.payload_bytes)) {
        f.close();
        return info;
    }
    ParamStoreEntry scratch[PARAM_COUNT];
    ParamStoreEntry* payload = entries ? entries : scratch;
    if (f.read(payload, header.payload_bytes) != header.payload_bytes ||
        param_store_crc32(payload, header.payload_bytes) != header.payload_crc32) {
        f.close();
        return info;
    }
    f.close();
    info.valid = true;
    info.generation = header.generation;
    info.count = header.count;
    info.version = header.version;
    return info;
}

// ── v2 → v3 unit migration ───────────────────────────────────────────────────
//
// The AK45-10's MIT torque/velocity full-scale constants were wrong (see the
// T_MIN/T_MAX note in hip_motors.cpp), so a handful of persisted params were
// tuned in units that no longer mean the same thing. Rescale exactly those,
// once, preserving the PHYSICAL behaviour the operator originally tuned:
//
//   * feedforward torques were packed over ±18 and decoded by the motor over
//     ±8, so only 8/18 of the commanded value was ever delivered. Multiply by
//     8/18 so the same torque still comes out now that packing is correct.
//   * calibration stall limits were compared against a reply field decoded
//     over ±20 that is really ±8, i.e. 2.5× too large. Multiply by 8/20.
//   * jump_omega_max is compared against reported hip velocity, which was
//     3.25× too large. Multiply by 20/65.
//
// Anything not listed keeps its stored value. The scale is applied to the RAW
// stored value, before the min/max clamp — several of these params also got
// tighter bounds in the same change, so clamping first would silently truncate
// the value being converted.
struct HipScaleMigration {
    uint16_t id;
    float    scale;
};
static const HipScaleMigration HIP_SCALE_V2_TO_V3[] = {
    {PARAM_HIP_RUNNING_TFF_RET,        8.0f / 18.0f},
    {PARAM_HIP_RUNNING_TFF_EXT,        8.0f / 18.0f},
    {PARAM_JUMP_TORQUE_MAX,            8.0f / 18.0f},
    {PARAM_CALIB_SEEK_TORQUE_LIMIT_NM, 8.0f / 20.0f},
    {PARAM_CALIB_MOVE_TORQUE_LIMIT_NM, 8.0f / 20.0f},
    {PARAM_JUMP_OMEGA_MAX,            20.0f / 65.0f},
};

// Multiplier to bring one param from `from_version` up to PARAM_STORE_VERSION.
static float migration_scale(uint16_t id, uint16_t from_version) {
    if (from_version >= PARAM_STORE_VERSION) return 1.0f;
    for (const HipScaleMigration& m : HIP_SCALE_V2_TO_V3) {
        if (m.id == id) return m.scale;
    }
    return 1.0f;
}

static uint16_t apply_entries(const ParamStoreEntry* entries, uint16_t count,
                              uint16_t store_version) {
    uint16_t restored = 0;
    for (uint16_t i = 0; i < count; ++i) {
        Param* p = find(entries[i].id);
        float value = entries[i].value;
        if (!p || !(p->flags & PARAM_FLAG_PERSISTENT) || !isfinite(value)) continue;
        const float scale = migration_scale(p->id, store_version);
        if (scale != 1.0f) {
            const float before = value;
            value *= scale;
            comm_log(LOG_LEVEL_WARN, "Param migrate v%u->v%u: %s %.4f -> %.4f",
                     store_version, PARAM_STORE_VERSION, p->name, before, value);
        }
        if (value < p->min_val) value = p->min_val;
        if (value > p->max_val) value = p->max_val;
        p->value = value;
        ++restored;
    }
    return restored;
}

static bool save_to_flash() {
    ParamStoreEntry entries[PARAM_COUNT];
    uint16_t count = 0;
    for (uint16_t i = 0; i < PARAM_COUNT; ++i) {
        if (!(g_params[i].flags & PARAM_FLAG_PERSISTENT)) continue;
        entries[count++] = {g_params[i].id, g_params[i].value};
    }

    const uint8_t target_slot = s_store_active_slot == 0 ? 1 : 0;
    const char* target = target_slot == 0 ? PARAM_SLOT_A : PARAM_SLOT_B;
    const uint32_t generation = s_store_generation + 1;
    ParamStoreHeader header{
        PARAM_STORE_MAGIC, PARAM_STORE_VERSION, sizeof(ParamStoreHeader),
        generation, count, (uint16_t)(count * sizeof(ParamStoreEntry)),
        param_store_crc32(entries, count * sizeof(ParamStoreEntry)), 0
    };
    header.header_crc32 = param_store_header_crc(header);

    // Only the inactive slot is replaced. A reset at any point leaves the
    // previously active generation intact and selectable on the next boot.
    s_fs.remove(target);
    File f = s_fs.open(target, FILE_WRITE);
    bool ok = f && f.write((const uint8_t*)&header, sizeof(header)) == sizeof(header) &&
              f.write((const uint8_t*)entries, header.payload_bytes) == header.payload_bytes;
    if (f) { f.flush(); f.close(); }
    SlotInfo verified = ok ? inspect_slot(target) : SlotInfo{};
    ok = verified.valid && verified.generation == generation && verified.count == count;
    if (!ok) {
        s_store_status.state = ParamPersistenceState::SAVE_FAILED;
        comm_log(LOG_LEVEL_ERROR, "Param flash: slot %c save/verify FAILED",
                 target_slot == 0 ? 'A' : 'B');
        return false;
    }
    s_store_active_slot = target_slot;
    s_store_generation = generation;
    s_store_status.active_slot = target_slot;
    s_store_status.generation = generation;
    s_store_status.valid_slot_mask |= (uint8_t)(1u << target_slot);
    comm_log(LOG_LEVEL_INFO, "Param flash: saved slot %c gen=%lu count=%u",
             target_slot == 0 ? 'A' : 'B', (unsigned long)generation, count);
    return true;
}

static void load_from_flash() {
    SlotInfo a = inspect_slot(PARAM_SLOT_A);
    SlotInfo b = inspect_slot(PARAM_SLOT_B);
    s_store_status.valid_slot_mask = (a.valid ? 1u : 0u) | (b.valid ? 2u : 0u);
    if (!a.valid && !b.valid) {
        load_legacy_from_flash();
        if (s_legacy_loaded) {
            s_store_status.state = ParamPersistenceState::MIGRATED_LEGACY;
            save_to_flash();
            s_fs.remove(PARAM_LEGACY);
            comm_log(LOG_LEVEL_WARN, "Param flash: migrated legacy params.bin to CRC slots");
        } else {
            s_store_status.state = ParamPersistenceState::DEFAULTS;
            s_store_status.active_slot = 0xFF;
            comm_log(LOG_LEVEL_WARN, "Param flash: no valid CRC slot; using defaults");
        }
        return;
    }
    uint8_t chosen = !b.valid ? 0 : (!a.valid ? 1 :
        (param_store_generation_newer(b.generation, a.generation) ? 1 : 0));
    const char* path = chosen == 0 ? PARAM_SLOT_A : PARAM_SLOT_B;
    ParamStoreEntry entries[PARAM_COUNT];
    SlotInfo selected = inspect_slot(path, entries);
    uint16_t restored = apply_entries(entries, selected.count, selected.version);
    s_store_active_slot = chosen;
    s_store_generation = selected.generation;
    s_store_status.active_slot = chosen;
    s_store_status.generation = selected.generation;
    s_store_status.state = (a.valid && b.valid)
        ? ParamPersistenceState::LOADED_REDUNDANT
        : ParamPersistenceState::RECOVERED_SINGLE_SLOT;
    comm_log((a.valid && b.valid) ? LOG_LEVEL_INFO : LOG_LEVEL_WARN,
             "Param flash: loaded slot %c gen=%lu restored=%u/%u valid_mask=0x%02X",
             chosen == 0 ? 'A' : 'B', (unsigned long)selected.generation,
             restored, selected.count, s_store_status.valid_slot_mask);

    // apply_entries() already converted units for an older store. Persist the
    // result so the conversion runs exactly once.
    if (selected.version < PARAM_STORE_VERSION) {
        comm_log(LOG_LEVEL_WARN, "Param flash: migrated store v%u -> v%u, re-saving",
                 selected.version, PARAM_STORE_VERSION);
        save_to_flash();
    }
}

// ── Public API ────────────────────────────────────────────────────────────────

void param_init() {
    for (uint16_t i = 0; i < PARAM_COUNT; i++) s_defaults[i] = g_params[i].value;

    // Deliberately no format-on-failure retry here. LittleFS_Program::begin()
    // already does mount -> format -> mount internally (framework LittleFS.cpp),
    // so a virgin chip is formatted for us on first boot and this call would
    // only ever fire after that internal format had *also* failed — i.e. real
    // hardware trouble, where reformatting cannot help. All it would achieve is
    // erasing both CRC-protected generations, defeating the entire two-slot
    // recovery design from inside its own init path. Mount failure now means
    // "run on compiled defaults and say so"; a wipe must be deliberate.
    bool mounted = s_fs.begin(FS_SIZE_BYTES);
    if (mounted) {
        comm_log(LOG_LEVEL_INFO, "Param flash: mounted (%s)", s_fs.getMediaName());
    } else {
        comm_log(LOG_LEVEL_ERROR,
                 "Param flash: mount FAILED — params will NOT persist across reboot");
    }
    load_from_flash();
}

ParamSetResult param_set(uint16_t id, float val) {
    Param* p = find(id);
    if (!p)                          return ParamSetResult::NOT_FOUND;
    if (p->flags & PARAM_FLAG_READONLY) return ParamSetResult::READONLY;
    // A NaN bypasses ordinary min/max comparisons and would otherwise be
    // persisted and propagated into control arithmetic. Treat every
    // non-finite value as an invalid write with no side effect.
    if (!isfinite(val))              return ParamSetResult::NONFINITE;

    ParamSetResult result = ParamSetResult::OK;

    if (val < p->min_val || val > p->max_val) {
        val    = (val < p->min_val) ? p->min_val : p->max_val;
        result = ParamSetResult::CLAMPED;
    }

    p->value = val;
    if (p->on_change) p->on_change(val);
    if (p->flags & PARAM_FLAG_PERSISTENT) {
        s_dirty          = true;
        s_last_change_ms = millis();
    }

    return result;
}

void param_flush_service(bool allow_flush) {
    if (!s_dirty || !allow_flush) return;
    if (millis() - s_last_change_ms < FLUSH_QUIET_MS) return;
    if (save_to_flash()) s_dirty = false;
}

float param_get(uint16_t id) {
    const Param* p = find(id);
    return p ? p->value : 0.0f;
}

bool param_exists(uint16_t id) {
    return find(id) != nullptr;
}

uint16_t param_count() {
    return PARAM_COUNT;
}

uint16_t param_get_group(uint8_t group_id, Param* buf, uint16_t max) {
    uint16_t n = 0;
    for (uint16_t i = 0; i < PARAM_COUNT && n < max; i++) {
        if (g_params[i].group_id == group_id) buf[n++] = g_params[i];
    }
    return n;
}

bool param_by_index(uint16_t idx, Param* out) {
    if (idx >= PARAM_COUNT) return false;
    *out = g_params[idx];
    return true;
}

void param_save_all() {
    if (save_to_flash()) s_dirty = false;
}

void param_reset_defaults() {
    for (uint16_t i = 0; i < PARAM_COUNT; i++) {
        if (g_params[i].flags & PARAM_FLAG_READONLY) continue;  // matches param_set()'s own write gate
        g_params[i].value = s_defaults[i];
        if (g_params[i].on_change) g_params[i].on_change(g_params[i].value);
    }
    if (save_to_flash()) s_dirty = false;
    comm_log(LOG_LEVEL_WARN, "Param flash: all params reset to compile-time defaults");
}

void param_force_set(uint16_t id, float val) {
    Param* p = find(id);
    if (!p) return;
    p->value = val;
    if (p->flags & PARAM_FLAG_PERSISTENT) {
        s_dirty = true;
        s_last_change_ms = millis();
    }
}

ParamPersistenceStatus param_persistence_status() {
    return s_store_status;
}
