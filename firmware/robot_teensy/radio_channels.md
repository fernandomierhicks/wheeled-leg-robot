# Radio Channel Map

**TX Profile:** `ROBOT_03`  
**Reversals:** none

---

## Channel Assignments

| Ch  | TX Control         | Function             | Details                                                     |
|-----|--------------------|----------------------|-------------------------------------------------------------|
| C1  | Roll stick         | *(unused)*           | —                                                           |
| C2  | Pitch stick        | Forward velocity     | 1000–2000 → −1…+1 × `RADIO_VEL_MAX` → `PARAM_V_CMD_MS`   |
| C3  | Throttle stick     | Hip height / angle   | 1000–2000 → 0…1 → `PARAM_RADIO_HIP_CMD`                   |
| C4  | Rudder stick       | Yaw rate             | 1000–2000 → −1…+1 × `RADIO_YAW_MAX` → `PARAM_OMEGA_CMD_RDS` |
| C5  | SWA (left switch)  | Start calibration    | Rising edge > 1990, only from STANDBY                      |
| C6  | SWB / joystick btn | Launch / jump        | Rising edge > 1990, only from RUNNING                      |
| C7  | First knob         | Pitch trim           | 1000–2000 → ±5° (±0.0873 rad) → `PARAM_RADIO_PITCH_TRIM`  |
| C8  | Second knob        | *(unused)*           | —                                                           |
| C9  | 3-pos switch       | Speed profile        | < 1333 = profile 1, 1333–1667 = profile 2, > 1667 = profile 3 |
| C10 | Right switch (ARM) | Arm / disarm         | > 1990 → RUNNING (requires calibration); drop → STANDBY    |
