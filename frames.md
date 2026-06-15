# Coordinate Frames — Issue & Proposed Solutions

## Issue

`software/gui/robot_visualizer_tab.py` draws a coordinate frame (X red / Y
green / Z blue) at the body origin, both hip motors, and both wheel motors.
Currently `_set_frame()` draws every frame aligned with the **body** frame
(X fwd, Y left, Z up) — it doesn't reflect how the hip/wheel motors are
actually mounted.

Physically, both hip motors and both wheel motors are mounted with their
output shaft (the motor's rotation axis) pointing **outward, away from the
robot's midplane**:

- Left side (hip_L, wheel_L): shaft points along body **+Y**
- Right side (hip_R, wheel_R): shaft points along body **-Y**

By convention (URDF revolute joints, most CAD rotary mates), a motor's local
**+Z axis is its shaft/rotation axis**. So in the visualizer, the blue (Z)
axis of the hip/wheel frames should point along ±Y (outward), not along body
Z (up) as it does today.

This should be a *real* transform, not a one-off visual hack — the goal is
for the visualizer to be a faithful "TF tree" representation of what the
code/firmware assumes about each frame's orientation, the same way ROS TF
represents static mounting transforms between links.

### Related existing case: IMU

`firmware/robot_teensy/teensy/lib/IMU/IMU.cpp` already does an ad-hoc axis
remap for the BNO086, because the sensor is mounted rotated relative to the
body frame:

```cpp
_pitch_rate = _sv.un.gyroscope.y;
_roll_rate  = _sv.un.gyroscope.x;
```

i.e. firmware swaps the sensor's raw X/Y gyro axes so that `imu_pitch_rate()`
/ `imu_roll_rate()` come out already expressed in **body-frame** convention.
Because of this, the IMU frame in the visualizer can be drawn as identity
(= body frame) — no extra transform needed there *for now*. But this is the
same kind of "local mounting transform" problem as the hip/wheel motors,
just already (partially) solved in firmware rather than documented
explicitly anywhere.

## Proposed Solutions

### Option A — Python "frame mount table" in the GUI (visualization-only)

Add a small dict of static rotation matrices (one per frame: `hip_L`,
`hip_R`, `wheel_L`, `wheel_R`, `body`, `imu`), each expressing "local +Z =
shaft axis -> body ±Y" for the motors and identity for body/IMU. Apply this
as `R_world = R_body @ R_mount` when drawing each frame in
`_set_frame()`.

- **Pros:** small, contained, immediately fixes the visual issue, documents
  the convention in one readable place.
- **Cons:** lives only in the GUI — firmware has no reference to it, so if
  firmware-side 3D transforms are ever needed (e.g. a full TF-style
  pipeline), this table would need to be ported/kept in sync manually.

### Option B — Shared frame-definition source, firmware as source of truth

Define the mounting transforms (rotation + translation per joint) once in a
firmware header (e.g. `shared/comm_protocol.h` or a new
`shared/robot_frames.h`), and have the GUI parse/mirror those constants for
visualization.

- **Pros:** firmware becomes the canonical source of truth, matching the
  project's existing pattern of `shared/comm_protocol.h` being the
  GUI/firmware contract.
- **Cons:** firmware currently does **no** 3D transform math at all — hip
  motors are scalar joint angles, wheel motors are scalar velocities, and
  the IMU remap is a simple axis swap, not a rotation matrix. Introducing a
  shared rotation-matrix representation would be infrastructure with no
  current firmware consumer — likely premature until a real use case (e.g.
  full-body state estimation needing multiple sensor frames) exists.

### Option C — Hybrid (recommended starting point)

Do Option A now (small Python mount table in the GUI, clearly documented,
following the same "+Z = shaft axis, points outward" convention used here),
with a comment explicitly noting the IMU.cpp gyro x/y swap as the
"firmware-side equivalent" of this same mounting-transform concept. If/when
firmware grows an actual need for 3D mounting transforms (multi-IMU, full
TF tree, etc.), promote the table to a shared header at that point — don't
build that infrastructure speculatively now.

## Recommendation

Option C: fix the visualizer now with a small, well-documented Python mount
table (Option A's implementation), and explicitly note the parallel to
`IMU.cpp`'s existing remap so the convention is discoverable from both
sides. Revisit Option B only when a concrete firmware-side need for 3D
frame transforms arises.
