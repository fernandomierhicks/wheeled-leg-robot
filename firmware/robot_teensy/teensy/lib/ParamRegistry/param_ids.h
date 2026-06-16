#pragma once
#include <stdint.h>

// ── Group IDs (high byte of param ID) ────────────────────────────────────────
#define GROUP_SYSTEM   0x00
#define GROUP_CALIB    0x01
#define GROUP_HIP      0x02
#define GROUP_WHEEL    0x03
#define GROUP_CONTROL  0x04
#define GROUP_COMMAND  0x05
#define GROUP_IBUS     0x06

// ── Param IDs ─────────────────────────────────────────────────────────────────
// Convention: high byte = group, low byte = index within group.

// GROUP_HIP — hip motor behaviour
#define PARAM_ESTOP_HIP_DISABLE   0x0200  // 1=exit MIT on ESTOP entry and re-enter on reset, 0=leave MIT running

// GROUP_CALIB — hardstop calibration tuning
#define PARAM_CALIB_SEEK_SPEED    0x0100  // ramp speed toward hardstop [rad/s]
#define PARAM_CALIB_KP            0x0101  // position gain while seeking
#define PARAM_CALIB_KD            0x0102  // damping while seeking
#define PARAM_CALIB_HOLD_KP       0x0103  // position gain while holding at home
#define PARAM_CALIB_HOLD_KD       0x0104  // damping while holding at home
#define PARAM_CALIB_STALL_CUR     0x0105  // current threshold to declare a hardstop [A]
#define PARAM_CALIB_STALL_DEADBAND 0x0106 // max pos movement per tick to count as stalled [rad]
#define PARAM_CALIB_STALL_TICKS   0x0107  // consecutive stalled ticks before declaring hardstop (int stored as float)
#define PARAM_CALIB_MARGIN        0x0108  // safety margin from each hardstop [rad]
#define PARAM_CALIB_SAFETY_BOUND  0x0109  // max seek distance before fault [rad]
#define PARAM_CALIB_L_SEEK_DIR    0x010A  // sign (+1/-1) toward left hip bottom hardstop
#define PARAM_CALIB_R_SEEK_DIR    0x010B  // sign (+1/-1) toward right hip bottom hardstop
#define PARAM_CALIB_DONE          0x010C  // 1.0 = calibration completed at least once (persisted across reboots)

// GROUP_COMMAND — high-freq setpoints from radio/GUI
#define PARAM_RADIO_HIP_CMD       0x0500  // hip extension command from CH3 [0=retracted, 1=extended]; stale when radio dead

// GROUP_IBUS — RC receiver channel readings (live, READONLY, firmware-written)
#define PARAM_IBUS_CH0     0x0600  // RC channel 0  [1000–2000 µs]
#define PARAM_IBUS_CH1     0x0601  // RC channel 1
#define PARAM_IBUS_CH2     0x0602  // RC channel 2
#define PARAM_IBUS_CH3     0x0603  // RC channel 3
#define PARAM_IBUS_CH4     0x0604  // RC channel 4
#define PARAM_IBUS_CH5     0x0605  // RC channel 5
#define PARAM_IBUS_CH6     0x0606  // RC channel 6
#define PARAM_IBUS_CH7     0x0607  // RC channel 7
#define PARAM_IBUS_CH8     0x0608  // RC channel 8
#define PARAM_IBUS_CH9     0x0609  // RC channel 9
#define PARAM_IBUS_CH10    0x060A  // RC channel 10
#define PARAM_IBUS_CH11    0x060B  // RC channel 11
#define PARAM_IBUS_CH12    0x060C  // RC channel 12
#define PARAM_IBUS_CH13    0x060D  // RC channel 13
#define PARAM_IBUS_ALIVE   0x060E  // link alive: 1.0 = packet within 500 ms, 0.0 = lost
