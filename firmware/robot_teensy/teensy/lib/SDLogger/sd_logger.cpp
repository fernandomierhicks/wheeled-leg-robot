#include "sd_logger.h"
#include <SdFat.h>
#include <RingBuf.h>
#include <string.h>
#include <stdio.h>

// Preallocated once per log so writes never hit the FAT allocator mid-flight.
static const uint64_t WLOG_PREALLOC_BYTES = 64ULL * 1024 * 1024;  // 64 MB

static SdFs   sd;
static bool   g_available = false;

// ── Active recording ───────────────────────────────────────────────────────
static FsFile g_wr_file;
static bool   g_active = false;
static uint16_t g_active_index = 0;
static bool   g_has_deadline = false;
static uint32_t g_stop_deadline_ms = 0;
static uint32_t g_last_flush_ms = 0;
static bool   g_overflow = false;

// Ring buffer cushion between the 500 Hz write() calls and the SD write-out;
// placed in DMAMEM to keep it out of the fast, small RAM1 region.
DMAMEM static RingBuf<FsFile, 32768> rb;

// ── Retrieval / transfer ───────────────────────────────────────────────────
static FsFile   g_rd_file;
static bool     g_xfer_active = false;
static uint16_t g_xfer_index = 0;
static uint32_t g_xfer_chunk = 0;
static uint32_t g_xfer_total_chunks = 0;
static uint64_t g_xfer_file_size = 0;
static uint32_t g_crc_state = 0;
// Pacing (audit W5): at most g_xfer_burst chunks per g_xfer_interval_ms window.
static uint32_t g_xfer_interval_ms = 0;
static uint8_t  g_xfer_burst = 2;
static uint32_t g_xfer_last_ms = 0;

static sd_logger_sender_t g_sender = nullptr;

// ── CRC32 (IEEE 802.3, reflected, poly 0xEDB88320) — matches zlib.crc32() ──
static uint32_t g_crc_table[256];
static bool     g_crc_table_ready = false;

static void crc32_init_table() {
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t c = i;
        for (int k = 0; k < 8; k++) {
            c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
        }
        g_crc_table[i] = c;
    }
    g_crc_table_ready = true;
}

static void crc32_begin() {
    if (!g_crc_table_ready) crc32_init_table();
    g_crc_state = 0xFFFFFFFFu;
}

static void crc32_feed(const uint8_t* buf, size_t len) {
    for (size_t i = 0; i < len; i++) {
        g_crc_state = g_crc_table[(g_crc_state ^ buf[i]) & 0xFF] ^ (g_crc_state >> 8);
    }
}

static uint32_t crc32_final() { return g_crc_state ^ 0xFFFFFFFFu; }

// ── Helpers ─────────────────────────────────────────────────────────────────
static void make_filename(uint16_t idx, char* buf, size_t n) {
    snprintf(buf, n, "LOG%04u.WLOG", idx);
}

// Expects the exact 8.3 short name this module writes: "LOGnnnn.WLOG".
static bool parse_log_filename(const char* name, uint16_t* idx_out) {
    if (strlen(name) != 12) return false;
    if (strncmp(name, "LOG", 3) != 0) return false;
    if (strcmp(name + 7, ".WLOG") != 0) return false;
    for (int i = 3; i < 7; i++) {
        if (name[i] < '0' || name[i] > '9') return false;
    }
    *idx_out = (uint16_t)atoi(name + 3);
    return true;
}

static void emit_status(uint16_t idx, uint8_t status) {
    if (!g_sender) return;
    LogInfoPayload p{};
    p.info_type = LOG_INFO_STATUS;
    p.file_index = idx;
    p.status = status;
    g_sender(COMM_TYPE_LOG_INFO, LOG_INFO_PAYLOAD_V1, &p, sizeof(p));
}

// ── Public API ──────────────────────────────────────────────────────────────

bool sd_logger_begin() {
    g_available = sd.begin(SdioConfig(FIFO_SDIO));
    return g_available;
}

bool sd_logger_available() { return g_available; }

bool sd_logger_start(uint32_t duration_ms) {
    if (!g_available || g_active) {
        emit_status(g_active ? g_active_index : 0xFFFF, 1);
        return false;
    }

    uint16_t idx = 1;
    char name[16];
    for (; idx <= 9999; idx++) {
        make_filename(idx, name, sizeof(name));
        if (!sd.exists(name)) break;
    }
    if (idx > 9999) {
        emit_status(0xFFFF, 1);
        return false;
    }
    make_filename(idx, name, sizeof(name));

    if (!g_wr_file.open(name, O_RDWR | O_CREAT | O_TRUNC)) {
        emit_status(idx, 1);
        return false;
    }
    if (!g_wr_file.preAllocate(WLOG_PREALLOC_BYTES)) {
        g_wr_file.close();
        emit_status(idx, 1);
        return false;
    }

    WlogHeader hdr;
    memset(&hdr, 0, sizeof(hdr));
    memcpy(hdr.magic, "WLRLOG", 7);  // 6 letters + implicit '\0' = 7 of 8 bytes
    hdr.format_version = WLOG_FORMAT_V1;
    hdr.telem_version = TELEM_VERSION;
    hdr.record_size = sizeof(LogRecord);
    hdr.sample_rate_hz = WLOG_SAMPLE_HZ;
    hdr.start_millis = millis();
    if (g_wr_file.write(&hdr, sizeof(hdr)) != sizeof(hdr)) {
        g_wr_file.close();
        emit_status(idx, 1);
        return false;
    }

    rb.begin(&g_wr_file);
    g_active_index = idx;
    g_overflow = false;
    g_has_deadline = duration_ms != 0;
    g_stop_deadline_ms = millis() + duration_ms;
    g_last_flush_ms = millis();
    g_active = true;
    emit_status(idx, 0);
    return true;
}

void sd_logger_stop() {
    if (!g_active) return;
    rb.sync();
    g_wr_file.truncate();
    g_wr_file.close();
    g_active = false;
    emit_status(g_active_index, g_overflow ? 2 : 0);
}

bool sd_logger_is_active() { return g_active; }

void sd_logger_write(const LogRecord* rec) {
    if (!g_active) return;
    rb.write(rec, sizeof(LogRecord));
    if (rb.getWriteError()) {
        if (!g_overflow) {
            g_overflow = true;
            comm_log(LOG_LEVEL_WARN, "SDLogger: ring buffer overflow, samples dropped");
        }
        rb.clearWriteError();
    }
}

void sd_logger_service() {
    if (!g_active) return;

    // Auto-stop before the preallocated file fills — mirrors SdFat's own
    // TeensySdioLogger reference example.
    if ((uint64_t)g_wr_file.curPosition() + rb.bytesUsed() >= WLOG_PREALLOC_BYTES - 512) {
        sd_logger_stop();
        return;
    }

    if (rb.bytesUsed() >= 512 && !g_wr_file.isBusy()) {
        rb.writeOut(512);
    }

    uint32_t now = millis();
    if (now - g_last_flush_ms >= 1000) {
        g_wr_file.flush();
        g_last_flush_ms = now;
    }

    if (g_has_deadline && (int32_t)(now - g_stop_deadline_ms) >= 0) {
        sd_logger_stop();
    }
}

uint16_t sd_logger_active_index() { return g_active_index; }

void sd_logger_list() {
    if (!g_sender) return;

    FsFile dir;
    if (dir.open("/")) {
        FsFile entry;
        char name[32];
        while (entry.openNext(&dir, O_RDONLY)) {
            if (!entry.isDir() && entry.getName(name, sizeof(name)) > 0) {
                uint16_t idx;
                if (parse_log_filename(name, &idx)) {
                    LogInfoPayload p{};
                    p.info_type = LOG_INFO_ENTRY;
                    p.file_index = idx;
                    p.file_size = (uint32_t)entry.fileSize();
                    g_sender(COMM_TYPE_LOG_INFO, LOG_INFO_PAYLOAD_V1, &p, sizeof(p));
                }
            }
            entry.close();
        }
        dir.close();
    }

    LogInfoPayload end{};
    end.info_type = LOG_INFO_LIST_END;
    end.file_index = 0xFFFF;
    g_sender(COMM_TYPE_LOG_INFO, LOG_INFO_PAYLOAD_V1, &end, sizeof(end));
}

void sd_logger_begin_get(uint16_t idx, uint32_t start_chunk) {
    if (g_xfer_active) {
        emit_status(idx, 1);  // only one transfer at a time
        return;
    }

    char name[16];
    make_filename(idx, name, sizeof(name));
    if (!g_rd_file.open(name, O_RDONLY)) {
        emit_status(idx, 1);
        return;
    }

    g_xfer_file_size = g_rd_file.fileSize();
    g_xfer_total_chunks = (uint32_t)((g_xfer_file_size + LOG_CHUNK_DATA - 1) / LOG_CHUNK_DATA);

    uint64_t seek_pos = (uint64_t)start_chunk * LOG_CHUNK_DATA;
    if (seek_pos > g_xfer_file_size) seek_pos = g_xfer_file_size;
    g_rd_file.seekSet(seek_pos);

    g_xfer_index = idx;
    g_xfer_chunk = start_chunk;
    crc32_begin();  // NOTE: covers only the bytes streamed from start_chunk onward
    g_xfer_active = true;

    if (g_sender) {
        LogInfoPayload p{};
        p.info_type = LOG_INFO_XFER_BEGIN;
        p.file_index = idx;
        p.file_size = (uint32_t)g_xfer_file_size;
        p.total_chunks = g_xfer_total_chunks;
        g_sender(COMM_TYPE_LOG_INFO, LOG_INFO_PAYLOAD_V1, &p, sizeof(p));
    }
}

void sd_logger_set_get_pacing(uint32_t interval_ms, uint8_t burst) {
    g_xfer_interval_ms = interval_ms;
    g_xfer_burst = (burst == 0) ? 1 : burst;
}

void sd_logger_service_transfer() {
    if (!g_xfer_active || !g_sender) return;

    // Pace to the requesting transport's wire rate — unthrottled streaming
    // (2 chunks/tick ≈ 490 kB/s) overruns the ESP32 CP2102 path (~92 kB/s).
    if (g_xfer_interval_ms) {
        uint32_t now = millis();
        if (now - g_xfer_last_ms < g_xfer_interval_ms) return;
        g_xfer_last_ms = now;
    }

    for (int i = 0; i < g_xfer_burst && g_xfer_active; i++) {
        uint8_t frame[sizeof(LogDataHeader) + LOG_CHUNK_DATA];
        int n = g_rd_file.read(frame + sizeof(LogDataHeader), LOG_CHUNK_DATA);
        if (n <= 0) {
            g_rd_file.close();
            g_xfer_active = false;

            LogInfoPayload p{};
            p.info_type = LOG_INFO_XFER_END;
            p.file_index = g_xfer_index;
            p.total_chunks = g_xfer_chunk;
            p.crc32 = crc32_final();
            g_sender(COMM_TYPE_LOG_INFO, LOG_INFO_PAYLOAD_V1, &p, sizeof(p));
            break;
        }

        crc32_feed(frame + sizeof(LogDataHeader), (size_t)n);

        LogDataHeader hdr;
        hdr.file_index = g_xfer_index;
        hdr.chunk_index = g_xfer_chunk;
        hdr.data_len = (uint16_t)n;
        memcpy(frame, &hdr, sizeof(hdr));

        g_sender(COMM_TYPE_LOG_DATA, LOG_DATA_PAYLOAD_V1, frame, sizeof(hdr) + (size_t)n);
        g_xfer_chunk++;
    }
}

bool sd_logger_transfer_active() { return g_xfer_active; }

void sd_logger_delete(uint16_t idx) {
    char name[16];
    make_filename(idx, name, sizeof(name));
    bool ok = sd.remove(name);
    emit_status(idx, ok ? 0 : 1);
}

void sd_logger_set_sender(sd_logger_sender_t fn) { g_sender = fn; }
