#pragma once
#include <stdint.h>
#include "comm_protocol.h"

// High-datarate SD logger. Owns the microSD card and the active .wlog file;
// does NOT own a CommLink instance — all Teensy→PC replies (LOG_INFO/LOG_DATA)
// go through the sender callback wired via sd_logger_set_sender(), mirroring
// how comm_log() fans out over both g_comm and g_comm_usb in main.cpp.

bool     sd_logger_begin();               // init SDIO card. false = no card (non-fatal)
bool     sd_logger_available();           // card present & initialised
bool     sd_logger_start(uint32_t duration_ms);  // 0 = until stop. false on error
void     sd_logger_stop();
bool     sd_logger_is_active();
void     sd_logger_write(const LogRecord* rec);  // called EVERY tick when active — RAM memcpy
void     sd_logger_service();             // drain 1 sector/tick + periodic flush + auto-stop
uint16_t sd_logger_active_index();        // current LOGxxxx index (for status reporting)

// Retrieval (driven by on_command in main.cpp; all Teensy→PC replies via callbacks)
void     sd_logger_list();                        // emit LOG_INFO ENTRY per file + LIST_END
// kind: LOG_FILE_KIND_WLOG (default) or LOG_FILE_KIND_PARAMS — which per-index
// file to stream (see comm_protocol.h).
void     sd_logger_begin_get(uint16_t idx, uint32_t start_chunk,
                              uint8_t kind = LOG_FILE_KIND_WLOG); // arm a streaming transfer
void     sd_logger_service_transfer();            // paced LOG_DATA streaming; emit XFER_END
bool     sd_logger_transfer_active();
void     sd_logger_ack_chunk(uint16_t idx, uint32_t chunk_index);
void     sd_logger_delete(uint16_t idx);          // erase .wlog + paired .PARAMS, LOG_INFO STATUS ack

// Transfer pacing (audit W5): every interval_ms, service_transfer sends at most
// `burst` chunks. Set BEFORE sd_logger_begin_get() based on the requesting
// transport's wire rate (CP2102 on the ESP32 path is only ~92 kB/s; direct
// Teensy USB is high-speed). interval_ms=0 → every tick.
void     sd_logger_set_get_pacing(uint32_t interval_ms, uint8_t burst);
void     sd_logger_set_get_ack_required(bool required);

// Send callback: emits one COMM_TYPE_LOG_INFO / COMM_TYPE_LOG_DATA frame.
// main.cpp wires this to a function that sends on g_comm and (if connected) g_comm_usb.
typedef void (*sd_logger_sender_t)(uint8_t type, uint8_t version, const void* payload, uint16_t len);
void     sd_logger_set_sender(sd_logger_sender_t fn);
