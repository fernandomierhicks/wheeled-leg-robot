# Jump Controller — Implementation Plan

## Context

Phase 5 of the control architecture (see `docs/Control.MD`) specifies a jump recovery state
machine. The existing archive `simulation/mujoco/archive/4bar_jump_sim/4bar_jump_sim.py` has
a proven single-leg jump implementation (crouch + max-torque extension + torque-speed limiting,
achieving ~283 mm jump height). This plan ports that logic into the `master_sim` framework
as a new controller + scenario (S10).

**Phase 1 approach: constant gains everywhere.** The JumpController only controls the hip
trajectory and torque — it does NOT modify LQR gains, switch to impedance mode, or change
any other controller parameters. The existing balance LQR + position-servo hip control runs
unchanged throughout. The only special behaviour is during the EXTEND phase, where the hip
gets a direct torque override instead of PD position tracking.

---

## Mode State Machine

```
[BALANCE] ──trigger()──▶ [CROUCH] ──trajectory done──▶ [EXTEND]
                                                          │
                                                     liftoff
                                                          │
[BALANCE] ◀── immediate ── [SETTLED] ◀── pitch stable ── [LANDING] ◀── contact ── [FLYING]
                              500 ms                                                  ▲
                                                                                      │
                                                                              wheels off ground
                                                                                 for 50 ms
```

| Mode | Hip Control | Balance LQR | Entry Trigger |
|------|-------------|-------------|---------------|
| BALANCE | Normal position servo (KP=50, KD=3) | Normal | default |
| CROUCH | Position servo tracking trajectory to Q_RET | Normal | `trigger()` called |
| EXTEND | **Direct torque override** (torque-speed limited) | Normal | Crouch trajectory complete |
| FLYING | Position servo holding current angle | Normal | Both wheels off ground for 50 ms |
| LANDING | Position servo holding current angle | Normal | Either wheel touches down |
| SETTLED | Normal position servo | Normal | |pitch| < 2° for 500 ms |

**Key simplification vs Control.MD:** No LQR gain scaling (K_pitch × 2), no impedance
switching, no suspension scaling. All controllers run with their baseline gains.

---

## `JumpGains` Dataclass

Added to `params.py` → `GainSet.jump`:

```python
@dataclass(frozen=True)
class JumpGains:
    crouch_time: float = 2.0          # [s] PD trajectory duration to Q_RET
    max_torque: float = 7.0           # [N·m] peak hip torque during extension
    omega_max: float = 18.85          # [rad/s] no-load speed (AK45-10 KV75@24V, 10:1)
    ramp_up_s: float = 0.010          # [s] torque ramp-up to avoid shock
    ramp_down_rad: float = 0.15       # [rad] taper zone before Q_EXT
    liftoff_debounce_s: float = 0.05  # [s] wheels-off-ground window for FLYING
    settle_pitch_deg: float = 2.0     # [deg] pitch threshold for SETTLED
    settle_time_s: float = 0.5        # [s] stable window for SETTLED
```

---

## `JumpController` Class

**File:** `master_sim/controllers/jump.py`

### API

```python
class JumpController:
    def __init__(self, robot: RobotGeometry, gains: JumpGains, dt: float)
    def trigger(self)                    # Start CROUCH (idempotent after first call)
    def update(self, t, hip_q_avg, hip_dq_avg,
               wheel_z_L, wheel_z_R, wheel_r, pitch) -> ModeOutput
```

### `ModeOutput` Dataclass

```python
@dataclass
class ModeOutput:
    mode: RobotMode
    hip_torque_override: Optional[float]  # None except during EXTEND
    q_hip_target: float                   # Hip position target
    dq_hip_target: float                  # Hip velocity feed-forward
```

### CROUCH Phase

Linear interpolation from capture angle → Q_RET over `crouch_time`:

```python
alpha = min(1.0, (t - crouch_start) / crouch_time)
q_target = q_crouch_start + (Q_RET - q_crouch_start) * alpha
dq_target = (Q_RET - q_crouch_start) / crouch_time   # constant velocity
```

Existing position-servo PD (KP=50, KD=3) tracks this trajectory.

### EXTEND Phase — Torque-Speed Limited Output

Ported from archive `4bar_jump_sim.py`:

```python
ramp_in = min(1.0, (t - extend_start) / ramp_up_s)

ramp_out = 1.0
if hip_q < Q_EXT + ramp_down_rad:
    ramp_out = max(0.0, (hip_q - Q_EXT) / ramp_down_rad)

speed_limit = max(0.0, 1.0 - abs(hip_dq) / omega_max)
torque = -max_torque * ramp_in * ramp_out * speed_limit
```

Negative sign because extension direction is Q_EXT < Q_RET (more negative angle).

### Transition Conditions

| From | To | Condition |
|------|----|-----------|
| BALANCE | CROUCH | `trigger()` called |
| CROUCH | EXTEND | `t - crouch_start >= crouch_time` |
| EXTEND | FLYING | both `wheel_z > wheel_r + 0.005` for `liftoff_debounce_s` |
| FLYING | LANDING | either `wheel_z < wheel_r + 0.005` |
| LANDING | SETTLED | `|pitch| < settle_pitch_deg` for `settle_time_s` |
| SETTLED | BALANCE | immediate (next tick) |

---

## Integration into `SimController.tick()`

**File:** `master_sim/sim_loop.py`

### New `tick()` Parameter

```python
def tick(self, model, data, *, ..., jump_active: bool = False) -> dict:
```

### Logic Added to `tick()` Body

After sensor reading, before hip control:

```python
if jump_active:
    hip_dq_avg = (data.qvel[self.d_hip_L] + data.qvel[self.d_hip_R]) / 2.0
    mode_out = self.jump_ctrl.update(
        data.time, hip_q_avg, hip_dq_avg,
        data.xpos[self.wheel_bid_L][2], data.xpos[self.wheel_bid_R][2],
        robot.wheel_r, pitch_true)
    q_hip_target = mode_out.q_hip_target
    dq_hip_target = mode_out.dq_hip_target
else:
    mode_out = None
```

Hip control section — insert before existing position-servo path:

```python
if mode_out is not None and mode_out.hip_torque_override is not None:
    # EXTEND phase: direct torque bypass
    data.ctrl[self.act_hip_L] = mode_out.hip_torque_override
    data.ctrl[self.act_hip_R] = mode_out.hip_torque_override
else:
    # Existing position servo / impedance path (unchanged)
    ...
```

Add to telemetry dict:

```python
mode=mode_out.mode.name if mode_out else "BALANCE",
```

### Changes to `run()` Loop

1. **Trigger jump at scheduled time:**
   ```python
   if scenario.jump_time is not None and data.time >= scenario.jump_time:
       ctrl.jump_ctrl.trigger()
   ```

2. **Skip liftoff_kill** when `jump_active` is in flags (intentional liftoff).

---

## Scenario S10

**File:** `master_sim/scenarios/s10_jump.py`

```python
CONFIG = ScenarioConfig(
    name="s10_jump",
    display_name="S10 — Jump",
    duration=8.0,
    active_controllers=frozenset({"lqr"}),
    hip_mode="jump",
    jump_time=1.0,
    fitness_fn=fitness,
    group="jump",
    order=10.0,
)
```

### `ScenarioConfig` Changes

**File:** `master_sim/scenarios/base.py`

- Add `jump_time: Optional[float] = None` field
- Add `hip_mode="jump"` handling in `tick_flags`:
  ```python
  jump_active=True, use_impedance=False, use_roll_leveling=False, use_suspension=True
  ```

### Registration

**File:** `master_sim/scenarios/__init__.py` — import S10 CONFIG, add to SCENARIOS dict.

---

## Timeline

| Phase | Duration |
|-------|----------|
| 1s | Balance settling |
| 2s | Crouch (PD trajectory to Q_RET) |
| ~0.1s | Extend (max torque, torque-speed limited) |
| ~0.3s | Flying (ballistic arc) |
| ~0.1s | Landing (impact) |
| ~0.5s+ | Settle (pitch < 2° for 500 ms) |
| remainder | Balance |

**Total scenario duration:** 8 s

---

## Files Modified/Created

| File | Action |
|------|--------|
| `master_sim/params.py` | Add `JumpGains`, add to `GainSet` |
| `master_sim/controllers/jump.py` | **NEW** — `RobotMode`, `ModeOutput`, `JumpController` |
| `master_sim/sim_loop.py` | Wire JumpController into `tick()` + `run()` |
| `master_sim/scenarios/base.py` | Add `jump_time`, `hip_mode="jump"` in `tick_flags` |
| `master_sim/scenarios/s10_jump.py` | **NEW** — Scenario + fitness |
| `master_sim/scenarios/__init__.py` | Register S10 |

## Existing Code Reused

| Function | File | Used For |
|----------|------|----------|
| `hip_position_torque()` | `controllers/hip.py` | PD servo in CROUCH/FLYING/LANDING |
| Liftoff detection (wheel Z) | `sim_loop.py:run()` | FLYING detection |
| Torque-speed formula | archive `4bar_jump_sim.py` | EXTEND torque computation |
| `get_equilibrium_pitch()` | `physics.py` | Pitch feedforward |

---

## Verification

1. `python -m master_sim.launcher` → select S10 → Play
2. Watch: 1 s balance → 2 s crouch → extension → liftoff → flight → landing → recovery
3. Telemetry `mode` field: BALANCE → CROUCH → EXTEND → FLYING → LANDING → SETTLED
4. Jump height target: ~283 mm (archive baseline)
5. Post-landing balance recovery with constant gains
6. S1–S9 unchanged (`jump_active=False` by default)

---

## Future Work (if constant gains don't suffice)

- **FLYING/LANDING:** Scale K_pitch × 2.0 for faster pitch recovery
- **LANDING:** Switch to impedance mode with K_s × 0.3 for soft touchdown
- **CROUCH:** Separate PD gains (KP=20, KD=1) from archive instead of position-servo (KP=50, KD=3)
- **Recovery:** Add VelocityPI during post-landing phases
- **Multiple jumps:** Change `jump_time` from single float to schedule/list
