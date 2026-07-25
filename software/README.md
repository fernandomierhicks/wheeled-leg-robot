# Robot software

Desktop software and archived host-side tools for the wheeled-leg robot.

> **AI maintenance note:** If you find anything here that is stale while
> working in this tree, update this README in the same change.

## Layout

- `gui/` — active PyQt6 operator application, automated tests, analysis code,
  and standalone command-line tools.
- `archive/` — retained earlier dashboard code; it is not the active GUI.

## Run the GUI

From the repository root:

```sh
python -m pip install -r software/gui/requirements.txt
python software/gui/main.py
```

The GUI can receive telemetry directly from Teensy or ESP32 USB serial, or
from the ESP32 over WiFi. It can flash both PlatformIO firmware projects,
inspect and edit firmware parameters, visualize live or replayed telemetry,
and retrieve/analyze SD-card logs.

The GUI and firmware share protocol definitions generated from
`firmware/robot_teensy/protocol/schema.json`. After changing the schema, run:

```sh
python firmware/robot_teensy/protocol/generate_protocol.py
python firmware/robot_teensy/protocol/generate_protocol.py --check
```

## Tests

From the GUI directory:

```sh
cd software/gui
python -m unittest discover -s tests -p "test_*.py"
```

Tests cover generated protocol consistency, golden vectors, transport and
state-machine contracts, telemetry/log routing, parameter sidecars, log
analysis, and single-instance behavior. Some GUI-oriented tests require the
packages in `software/gui/requirements.txt`.

## Useful command-line tools

`software/gui/tools/` contains utilities for robot control, WiFi capture and
benchmarking, hardware-run analysis, link baselining, log conversion, and
safe test triggering. Run a tool with `--help` before using it; tools that can
change robot state include their own safety checks, but the operator remains
responsible for keeping the robot mechanically safe.

Downloaded and captured run data belongs under `software/gui/data/` and is
ignored by Git.
