"""constants.py — Enums, axis states, and default values for ODrive 0.5.6."""

# Axis states (from odrive.enums)
AXIS_STATES = {
    0: "Undefined",
    1: "Idle",
    2: "Startup Sequence",
    3: "Full Calibration",
    4: "Motor Calibration",
    6: "Encoder Index Search",
    7: "Encoder Offset Calibration",
    8: "Closed Loop",
    9: "Lockin Spin",
    10: "Encoder Dir Find",
    11: "Homing",
}

# Motor types
MOTOR_TYPE_NAMES = ["High Current (0)", "Gimbal (2)", "ACIM (3)"]
MOTOR_TYPE_VALUES = [0, 2, 3]

# Encoder types
ENCODER_TYPE_NAMES = ["SPI Absolute AMS (257)", "Incremental (0)"]
ENCODER_TYPE_VALUES = [257, 0]

# Anticogging map size (fixed in ODrive 0.5.6 firmware)
COGGING_MAP_SIZE = 3600

# Timing
POLL_MS = 100
