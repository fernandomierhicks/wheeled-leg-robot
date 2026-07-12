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
void     sd_logger_begin_get(uint16_t idx, uint32_t start_chunk); // arm a streaming transfer
void     sd_logger_service_transfer();            // pace 1-2 LOG_DATA chunks/tick; emit XFER_END
bool     sd_logger_transfer_active();
void     sd_logger_delete(uint16_t idx);          // erase + LOG_INFO STATUS ack

// Send callback: emits one COMM_TYPE_LOG_INFO / COMM_TYPE_LOG_DATA frame.
// main.cpp wires this to a function that sends on g_comm and (if connected) g_comm_usb.
typedef void (*sd_logger_sender_t)(uint8_t type, uint8_t version, const void* payload, uint16_t len);
void     sd_logger_set_sender(sd_logger_sender_t fn);
