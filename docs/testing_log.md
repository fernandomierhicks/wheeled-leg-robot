7/13/2026

## FablePlan Phase 0 + Phase 1 bench verification

Full audit-fix pass verified on the bench (see FablePlan.md §15 for the checklist).
Fresh flash on Teensy 4.1 and ESP32, GUI restarted from the same checkout — required
because the frame checksum changed from XOR to CRC-8 (flag day: mixed old/new ends
reject all frames).

Covered: safety fixes F1-F10 (IMU silence watchdog + IMU_LOST/WHEEL_FEEDBACK_LOST
ESTOPs, command permission matrix, ESTOP request flush, level-based disarm, 500 ms
MANUAL watchdog + GUI CMD_ID_PING heartbeat, param clamp cleanup, per-motor hip send
gating), D1 (calibration via setpoint cache — single MIT frame/tick), R1 (deferred
param flash flush), R2 (loop-overrun counter), and comms robustness W1-W7 + D4
(CRC-8, GUI checksum enforcement + link-health counters, seq gap counting,
seq-adjacent A/B pairing, SD GET gate + per-transport pacing, comm_log 120 B,
WheelMotors via comm_log, 50 Hz watchdog pet).

---

7/12/2026

## Calibration

Right leg only, no wheel. Robot body clamped to desk. Opposite dynamics than real robot. Retractio is hard, extension easy (real robot would be backwards)

Retraction moves needs 
calib_stall_curr_btm to be 5A, to counteract the weight of the wheel motor. 
calib_stall_curr_top to be 4A, to allow holding the weight of the wheel motor while smoothly going "down" into the extended position. 

After calibration.Able to move hip up and down along with turning wheel at the same time. Both with sine waves. 

GOOD TO PROCEED TO WIRE OTHER LEG!!!




7/17/2026

## Calibration

Both legs with wheels. 

calib_l_seek_dir=-1
calib_r_seek_dir=+1

Retraction moves needs 
calib_stall_curr_btm to be 1A, to counteract the weight of the wheel motor + wheel. 
calib_stall_curr_top to be 5A, to allow holding the weight of the wheel motor while smoothly going "down" into the extended position. 


Hips need hip_running_kp=20 to be able to move up and down in running mode. 


