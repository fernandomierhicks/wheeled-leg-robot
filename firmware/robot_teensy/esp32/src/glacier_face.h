#pragma once
#include <stdint.h>

class TFT_eSPI;

// GLACIER "Visor Eyes" face for the front TFT (display personality PERS_GLACIER).
// Two chamfered, slatted eye plates and a trace mouth over animated circuit traces,
// drawn into one 320x240 4-bit palettized sprite (38,400 B) and pushed whole each frame.

// What the face expresses. main.cpp maps the Teensy's robot state + link status onto these.
enum GlacierState : uint8_t {
    GF_OFFLINE,   // no Teensy link
    GF_STARTUP,
    GF_CALIB,
    GF_STANDBY,
    GF_RUNNING,
    GF_MANUAL,
    GF_JUMP,
    GF_REJECT,
    GF_ESTOP,
    GF_STANDUP,
    GF_DISARM,
    GF_COUNT
};

bool glacier_face_begin(TFT_eSPI* tft);                 // false if the sprite can't be allocated
void glacier_face_end();
void glacier_face_update(GlacierState st, uint32_t now_ms);  // draw and push one frame
