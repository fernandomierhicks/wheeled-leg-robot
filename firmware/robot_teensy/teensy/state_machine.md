# Teensy safety state machine

The numeric IDs and ordered transition list are frozen by
`test/state_machine_contract.json`; `software/gui/tests/test_state_machine_contract.py`
fails if this document's source implementation changes without an intentional
contract update.

Safety priority within every active state is: explicit ESTOP, motor feedback
fault, IMU fault, normal disarm/abort, then sequence completion or operator
requests.

| State | Normal exits | Safety exits |
| --- | --- | --- |
| STARTUP | readiness → STANDBY | ESTOP or startup failure → ESTOP |
| STANDBY | manual, calibration, arm → RUNNING/STANDING_UP | ESTOP or motor fault → ESTOP |
| MANUAL | operator exit or GUI timeout → STANDBY | ESTOP or motor fault → ESTOP |
| CALIBRATION | complete/operator exit → STANDBY | ESTOP, motor fault, calibration failure → ESTOP |
| RUNNING | disarm → DISARMING; jump → JUMPING | ESTOP, motor fault, or stale/non-nominal IMU → ESTOP |
| JUMPING | disarm → DISARMING; complete → RUNNING | ESTOP, motor fault, or IMU fault → ESTOP |
| STANDING_UP | disarm → DISARMING; capture → RUNNING | ESTOP, motor fault, or IMU fault → ESTOP |
| DISARMING | hip ramp complete → STANDBY | ESTOP, motor fault, or IMU fault → ESTOP |
| CMD_REJECT | one-second indication complete → STANDBY | ESTOP or motor fault → ESTOP |
| ESTOP | soft clear → STANDBY; reset → STARTUP | outputs remain in the ESTOP policy |

Arming is admitted only from STANDBY, only with an IMU that is NOMINAL and
no more than 50 ms old, and only after the configured calibration/motor gates
pass. `CMD_PAYLOAD_V2` callers receive a correlated rejection result when a
guard fails. Repeated ESTOP while already in ESTOP is idempotent and does not
leave a stale event that can fire after reset.
