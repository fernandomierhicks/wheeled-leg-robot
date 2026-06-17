#include "param_registry.h"
#include <LittleFS.h>

// ── Flash filesystem ──────────────────────────────────────────────────────────
static LittleFS_QSPI s_fs;
static const char*   PARAMS_FILE = "/params.bin";
static const uint16_t MAGIC      = 0xB0B1;
static const uint8_t  VERSION    = 1;

// ── Registry table ────────────────────────────────────────────────────────────
// Add new params here. Defaults are the compile-time values previously in config.h.
// clang-format off
static Param g_params[] = {
    // id                        group          name                     value     min       max     flags                   on_change
    {PARAM_ESTOP_HIP_DISABLE, GROUP_HIP,   "estop_hip_disable",  1.0f,      0.0f,     1.0f,  PARAM_FLAG_PERSISTENT, nullptr},
    {PARAM_CALIB_SEEK_SPEED,  GROUP_CALIB, "calib_seek_speed",  0.17453f,  0.01f,    1.0f,  PARAM_FLAG_PERSISTENT, nullptr},
    {PARAM_CALIB_KP,          GROUP_CALIB, "calib_kp",          16.0f,     0.0f,   500.0f,  PARAM_FLAG_PERSISTENT, nullptr},
    {PARAM_CALIB_KD,          GROUP_CALIB, "calib_kd",           0.05f,    0.0f,     5.0f,  PARAM_FLAG_PERSISTENT, nullptr},
    {PARAM_CALIB_HOLD_KP,     GROUP_CALIB, "calib_hold_kp",      1.0f,     0.0f,   500.0f,  PARAM_FLAG_PERSISTENT, nullptr},
    {PARAM_CALIB_HOLD_KD,     GROUP_CALIB, "calib_hold_kd",      0.05f,    0.0f,     5.0f,  PARAM_FLAG_PERSISTENT, nullptr},
    {PARAM_CALIB_STALL_CUR,   GROUP_CALIB, "calib_stall_cur",    0.75f,    0.1f,    10.0f,  PARAM_FLAG_PERSISTENT, nullptr},
    {PARAM_CALIB_STALL_DEADBAND, GROUP_CALIB, "calib_stall_db",  0.015f,   0.001f,  0.5f,  PARAM_FLAG_PERSISTENT, nullptr},
    {PARAM_CALIB_STALL_TICKS, GROUP_CALIB, "calib_stall_ticks",  45.0f,    5.0f,  500.0f,  PARAM_FLAG_PERSISTENT, nullptr},
    {PARAM_CALIB_MARGIN,      GROUP_CALIB, "calib_margin",       0.17453f, 0.0f,    1.5708f, PARAM_FLAG_PERSISTENT, nullptr},
    {PARAM_CALIB_SAFETY_BOUND,GROUP_CALIB, "calib_safety_bound", 6.28319f, 1.0f,   25.1327f, PARAM_FLAG_PERSISTENT, nullptr},
    {PARAM_CALIB_L_SEEK_DIR,  GROUP_CALIB, "calib_l_seek_dir",   1.0f,    -1.0f,    1.0f,  PARAM_FLAG_PERSISTENT, nullptr},
    {PARAM_CALIB_R_SEEK_DIR,  GROUP_CALIB, "calib_r_seek_dir",  -1.0f,    -1.0f,    1.0f,  PARAM_FLAG_PERSISTENT, nullptr},
    {PARAM_CALIB_DONE,        GROUP_CALIB, "calib_done",         0.0f,     0.0f,    1.0f,  PARAM_FLAG_READONLY, nullptr},

    // GROUP_COMMAND — radio-derived setpoints (firmware-written, never persisted)
    {PARAM_RADIO_HIP_CMD, GROUP_COMMAND, "radio_hip_cmd",     0.0f,     0.0f,    1.0f,  PARAM_FLAG_READONLY|PARAM_FLAG_COMMAND, nullptr},

    // GROUP_IBUS — RC receiver live channel readings (firmware-written via param_force_set)
    {PARAM_IBUS_CH0,   GROUP_IBUS, "ibus_ch0",   1500.0f, 1000.0f, 2000.0f, PARAM_FLAG_READONLY|PARAM_FLAG_COMMAND, nullptr},
    {PARAM_IBUS_CH1,   GROUP_IBUS, "ibus_ch1",   1500.0f, 1000.0f, 2000.0f, PARAM_FLAG_READONLY|PARAM_FLAG_COMMAND, nullptr},
    {PARAM_IBUS_CH2,   GROUP_IBUS, "ibus_ch2",   1500.0f, 1000.0f, 2000.0f, PARAM_FLAG_READONLY|PARAM_FLAG_COMMAND, nullptr},
    {PARAM_IBUS_CH3,   GROUP_IBUS, "ibus_ch3",   1500.0f, 1000.0f, 2000.0f, PARAM_FLAG_READONLY|PARAM_FLAG_COMMAND, nullptr},
    {PARAM_IBUS_CH4,   GROUP_IBUS, "ibus_ch4",   1500.0f, 1000.0f, 2000.0f, PARAM_FLAG_READONLY|PARAM_FLAG_COMMAND, nullptr},
    {PARAM_IBUS_CH5,   GROUP_IBUS, "ibus_ch5",   1500.0f, 1000.0f, 2000.0f, PARAM_FLAG_READONLY|PARAM_FLAG_COMMAND, nullptr},
    {PARAM_IBUS_CH6,   GROUP_IBUS, "ibus_ch6",   1500.0f, 1000.0f, 2000.0f, PARAM_FLAG_READONLY|PARAM_FLAG_COMMAND, nullptr},
    {PARAM_IBUS_CH7,   GROUP_IBUS, "ibus_ch7",   1500.0f, 1000.0f, 2000.0f, PARAM_FLAG_READONLY|PARAM_FLAG_COMMAND, nullptr},
    {PARAM_IBUS_CH8,   GROUP_IBUS, "ibus_ch8",   1500.0f, 1000.0f, 2000.0f, PARAM_FLAG_READONLY|PARAM_FLAG_COMMAND, nullptr},
    {PARAM_IBUS_CH9,   GROUP_IBUS, "ibus_ch9",   1500.0f, 1000.0f, 2000.0f, PARAM_FLAG_READONLY|PARAM_FLAG_COMMAND, nullptr},
    {PARAM_IBUS_CH10,  GROUP_IBUS, "ibus_ch10",  1500.0f, 1000.0f, 2000.0f, PARAM_FLAG_READONLY|PARAM_FLAG_COMMAND, nullptr},
    {PARAM_IBUS_CH11,  GROUP_IBUS, "ibus_ch11",  1500.0f, 1000.0f, 2000.0f, PARAM_FLAG_READONLY|PARAM_FLAG_COMMAND, nullptr},
    {PARAM_IBUS_CH12,  GROUP_IBUS, "ibus_ch12",  1500.0f, 1000.0f, 2000.0f, PARAM_FLAG_READONLY|PARAM_FLAG_COMMAND, nullptr},
    {PARAM_IBUS_CH13,  GROUP_IBUS, "ibus_ch13",  1500.0f, 1000.0f, 2000.0f, PARAM_FLAG_READONLY|PARAM_FLAG_COMMAND, nullptr},
    {PARAM_IBUS_ALIVE, GROUP_IBUS, "ibus_alive",    0.0f,    0.0f,    1.0f, PARAM_FLAG_READONLY|PARAM_FLAG_COMMAND, nullptr},
};
// clang-format on

static const uint16_t PARAM_COUNT = sizeof(g_params) / sizeof(g_params[0]);

// ── Internal helpers ──────────────────────────────────────────────────────────

static Param* find(uint16_t id) {
    for (uint16_t i = 0; i < PARAM_COUNT; i++) {
        if (g_params[i].id == id) return &g_params[i];
    }
    return nullptr;
}

static void load_from_flash() {
    File f = s_fs.open(PARAMS_FILE, FILE_READ);
    if (!f) return;

    uint16_t magic;
    uint8_t  ver;
    uint16_t count;
    if (f.read(&magic, 2) != 2 || magic != MAGIC) { f.close(); return; }
    if (f.read(&ver,   1) != 1 || ver   != VERSION) { f.close(); return; }
    if (f.read(&count, 2) != 2) { f.close(); return; }

    for (uint16_t i = 0; i < count; i++) {
        uint16_t id;
        float    val;
        if (f.read(&id,  2) != 2) break;
        if (f.read(&val, 4) != 4) break;
        Param* p = find(id);
        if (p && !(p->flags & PARAM_FLAG_READONLY)) {
            // clamp to current bounds before restoring
            if (val < p->min_val) val = p->min_val;
            if (val > p->max_val) val = p->max_val;
            p->value = val;
        }
    }
    f.close();
}

static void save_to_flash() {
    s_fs.remove(PARAMS_FILE);
    File f = s_fs.open(PARAMS_FILE, FILE_WRITE);
    if (!f) return;

    // Count persistent params
    uint16_t count = 0;
    for (uint16_t i = 0; i < PARAM_COUNT; i++) {
        if (g_params[i].flags & PARAM_FLAG_PERSISTENT) count++;
    }

    f.write((uint8_t*)&MAGIC,   2);
    f.write((uint8_t*)&VERSION, 1);
    f.write((uint8_t*)&count,   2);

    for (uint16_t i = 0; i < PARAM_COUNT; i++) {
        if (!(g_params[i].flags & PARAM_FLAG_PERSISTENT)) continue;
        f.write((uint8_t*)&g_params[i].id,    2);
        f.write((uint8_t*)&g_params[i].value, 4);
    }
    f.close();
}

// ── Public API ────────────────────────────────────────────────────────────────

void param_init() {
    if (!s_fs.begin()) {
        s_fs.format();
        s_fs.begin();
    }
    load_from_flash();
}

ParamSetResult param_set(uint16_t id, float val) {
    Param* p = find(id);
    if (!p)                          return ParamSetResult::NOT_FOUND;
    if (p->flags & PARAM_FLAG_READONLY) return ParamSetResult::READONLY;

    ParamSetResult result = ParamSetResult::OK;

    if (val < p->min_val || val > p->max_val) {
        if (p->flags & PARAM_FLAG_FAULT_ON_BOUNDS) {
            return ParamSetResult::FAULT;  // caller must trigger ESTOP
        }
        val    = (val < p->min_val) ? p->min_val : p->max_val;
        result = ParamSetResult::CLAMPED;
    }

    p->value = val;
    if (p->on_change) p->on_change(val);
    if (p->flags & PARAM_FLAG_PERSISTENT) save_to_flash();

    return result;
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
    save_to_flash();
}

void param_reset_defaults() {
    s_fs.remove(PARAMS_FILE);
    // Values already at compile-time defaults in g_params[] — no RAM changes needed.
}

void param_force_set(uint16_t id, float val) {
    Param* p = find(id);
    if (!p) return;
    p->value = val;
}
