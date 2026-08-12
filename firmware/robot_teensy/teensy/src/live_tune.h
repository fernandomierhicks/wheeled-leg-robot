#pragma once
#include <stdint.h>

// Generic radio-knob live parameter tuning. Requires PARAM_LIVE_TUNE_MULTI_EN
// = 1 (LEGACY); in the default SIMPLE mode CH5/CH6 are the SD-log and jump
// switches and this is entirely inactive. In LEGACY mode it is active while
// RUNNING with the CH5/CH6 switch combination selecting one of 3 gain groups
// (see radio_update() in main.cpp: CH5 down+CH6 up -> group 0, CH5 up+CH6 down
// -> group 1, both down -> group 2, both up -> inactive). Knob->param
// mappings live in LIVE_TUNE_SLOTS (main.cpp) -- repoint a knob at a
// different param for a future tuning session by editing that table and
// reflashing; no other code changes needed as long as the target's read
// site uses live_tune_value() instead of a bare param_get().
//
// Safety: each slot requires "pickup" -- the knob must be swept through the
// target's current persisted value before it takes control, so entering
// live-tune mode never causes a step change from wherever the knob happens
// to be sitting. Nothing is written to the real persistent param until
// PARAM_LIVE_TUNE_LATCH is set to 1 (and only picked-up slots latch).

// Live value for a param currently driven by a picked-up live-tune slot, or
// the normal param_get(persist_param_id) otherwise. Drop-in replacement for
// param_get() at any control-loop read site that should be live-tunable.
float live_tune_value(uint16_t persist_param_id);
