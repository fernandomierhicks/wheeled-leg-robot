# Plan: New Engineering Screen Layout for ESP32 TFT

## Context

The current engineering screen has three issues to solve simultaneously:
1. **Choppiness**: Banner, hip bars, and footer draw directly to TFT (no double-buffering) → visible tearing
2. **Layout priorities not matching**: Battery/profile/mode all crammed into one thin banner with no hierarchy
3. **Text-heavy at small scale**: Labels like "HIP L", "WHEELS", "WIFI" are illegible at 320×240 on a small mounted display

This redesign converts every region to a sprite (fixing choppiness), rethinks the visual hierarchy with icons over text, adds setpoint needles to show commanded vs actual, and introduces smooth per-mode gradient animations inspired by the `tft_benchmark` rainbow gradient technique.

---

## New Screen Layout (320×240, landscape)

```
┌──────────────────────────────────────────────────┐  y=0
│  BANNER sprite (320×44)                          │
│  [P●] [══ MODE animated bg + text ══] [🔋 23.5V] │
├────────────────┬───────────────┬─────────────────┤  y=44
│  AH sprite     │  HIP sprite   │  WHEEL sprite   │
│  (160×148)     │  (80×148)     │  (80×148)       │
│  Artificial    │  Joint icons  │  Wheel icons    │
│  horizon       │  L/R arcs     │  L/R vel bars   │
│  (slightly     │  + cmd needle │  + cmd v_ref    │
│   smaller)     │  + currents   │  + state dots   │
├────────────────┴───────────────┴─────────────────┤  y=192
│  FOOTER sprite (320×48)                          │
│  [●ms] [≋WiFi] [📡RC] [●UART]   [CRC:0 GAP:0]   │
│  [LQ●][VP●][YP●][F1●][F2●][JM●]  [←v bar 1.2m/s]│
└──────────────────────────────────────────────────┘  y=240
```

Column widths: 160 + 80 + 80 = 320 ✓  
Row heights: 44 + 148 + 48 = 240 ✓

---

## Sprite Inventory (all new or resized)

| Name | Size | Bytes | Status |
|---|---|---|---|
| `banner_sprite` | 320×44 | ~28 KB | **NEW** |
| `ah_sprite` | 160×148 | ~47 KB | resize (was 170×156) |
| `hip_sprite` | 80×148 | ~24 KB | **NEW** (was direct draw) |
| `wm_sprite` | 80×148 | ~24 KB | resize (was 70×156) |
| `footer_sprite` | 320×48 | ~31 KB | **NEW** (was direct draw) |

Total engineering sprites: ~154 KB. Face sprites (eye 20 KB + mouth 16 KB) stay unchanged. ESP32 heap + PSRAM comfortable.

---

## Mode Banner Animations

Each mode gets a distinct smooth animation — no hard-edged diagonal stripes. All use the `tft_benchmark` gradient technique: per-pixel color blending based on `(x + y + t)` phase, producing smooth diagonal shimmering. Mode name text is rendered in size-3 font centered over the animated background (text is readable at this size).

| Mode | Text | Animation | Colors | Notes |
|---|---|---|---|---|
| **Disconnected** | "NO TEENSY" | Slow breathing pulse (sin fade whole banner) | Dark slate → dim blue | No animation detail, just dim glow |
| **STARTUP** | "STARTUP" | Left→right gradient sweep | Deep blue → cyan | Boot progress feel |
| **CALIBRATION** | "CALIBRATION" | Rotating Gaussian spotlight beam | Blue + bright white beam | Smooth falloff, not hard edge |
| **STANDBY** | "STANDBY" | Very subtle shimmer (low amplitude noise) | Amber/gold | Nearly static but alive |
| **RUNNING** | "RUNNING" | Smooth flowing diagonal gradient bars | Dark green → bright green | Speed scales with profile (P1 slow, P3 fast) |
| **ESTOP** | "ESTOP" + fault | Radial ripple pulse from center | Red → dark red | Sin concentric rings expanding from center |
| **MANUAL** | "MANUAL" | Slow diagonal gradient scroll | Deep violet → magenta | ~0.05 px/ms |
| **CMD_REJECT** | "CMD REJECT" | Rapid strobe flash (3× then hold orange) | Orange → white | 8 Hz, auto-clears |
| **JUMPING** | "JUMPING" | Fast upward vertical sweep repeating | Lime → white | Vertical stripe sweeping top→bottom |

All driven by `millis()` offset, no additional state variables.

---

## Banner Widget Details (banner_sprite, 320×44)

```
x=0     x=28      x=268    x=320
│Profile│  Mode name + animated bg  │Battery│
│  ●P1  │      RUNNING              │🔋23.5V│
│ 28px  │        240px              │  52px │
```

- **Profile indicator (left 28px)**: Filled circle (blue=P1, green=P2, orange=P3) + "P1" size-1 text. Dark badge behind it.
- **Mode name (center 240px)**: Size-3 font, dark text backdrop for legibility over any animation background.
- **Battery (right 52px)**: Existing 5-bar battery icon draw code moved into `banner_sprite`. Voltage in size-1. Blinks when critical. **No bus voltage shown separately** — this is the single voltage indicator.

---

## Hip Sprite Details (hip_sprite, 80×148)

Replace vertical fill bars with **mini arc gauges** showing actual position + commanded setpoint needle:

```
    [joint icon — 16px, 2 lines at ~120°]
    
     L arc      R arc       ← 180° sweep, r=26
      ~~~ ←needle (actual, thick cyan)
      --- ←needle (commanded, thin yellow)
                            ← hip_l/r_cmd_pos_rad from telemetry

    [current bar L]  [current bar R]   ← 8px tall, 0-8A
    [● health]       [● health]        ← HEALTH_HIP_L/R_OK flags
```

- **Arc gauge**: 180° sweep, radius 26px. Thick cyan line = actual position (`hip_l_pos_rad`). Thin yellow line = commanded (`hip_l_cmd_pos_rad`). No text on arc. Range 0–2.0 rad.
- **Joint icon**: 16×10px bitmap drawn at top-center — two short lines meeting at ~120° angle.
- **Current bar**: 8px horizontal fill bar, 0-8A range, green→orange→red thresholds.
- **Health dot**: 6px circle using `HEALTH_HIP_L_OK` / `HEALTH_HIP_R_OK` from `health_flags`.

---

## Wheel Sprite Details (wm_sprite, 80×148)

```
    [wheel icon — circle + 4 spokes, 16px]

     L           R
  [vel bar]   [vel bar]     ← bidirectional, ±20 t/s
                            ← tick mark for v_ref command
   3.2t/s      3.1t/s

  [●IDLE]    [●VEL]         ← state dot + 3-char mode

  [←── v_ref bar ──→]       ← spans full width, ±2.5 m/s
   1.23 m/s (avg actual)    ← wheel_vel_avg_ms shown as label
```

- **Wheel icon**: 16px circle outline + 4 spokes drawn at top-center.
- **Per-wheel velocity bars**: 26px wide, 60px tall, zero-centered. Green=forward, orange=reverse. A horizontal tick mark shows the `v_ref` command level.
- **State dot**: 6px (green=closed-loop, yellow=idle, red=error) + 3-char mode ("IDL"/"VEL"/"POS"/"TRQ").
- **Speed bar (bottom)**: Bidirectional bar spanning full 80px, ±2.5 m/s range using `wheel_vel_avg_ms`. Command marker from `v_ref`. **This replaces the speed bar that was in the footer.**
- **No bus voltage** — already covered by banner battery icon.

---

## Footer Sprite Details (footer_sprite, 320×48)

Two rows, with speed bar moved to wheel sprite:

**Row 1 (y=0–22 within sprite): Connection status**
```
[●Xms]  [≋]    [|||]   [●]           [CRC:X GAP:Y]
 hbeat  WiFi   RC bars  UART           error counts
```
- **Heartbeat dot**: 8px, pulses with mode color.
- **Packet age**: "Xms" size-1, white if fresh, dim if stale.
- **WiFi icon**: Classic WiFi symbol — dot at bottom + 3 concentric arcs above, drawn with `drawArc()`. Cyan=active, dim=inactive.
- **RC icon**: Cellular signal bars — 4 rectangles of increasing height (widths: 4px each, heights: 4/7/10/13px), spaced 2px apart. Green=alive, red=lost. No text label.
- **UART dot**: 6px circle, green=connected, yellow=waiting.
- **CRC/GAP**: right-aligned, size-1 text.

**Row 2 (y=22–48 within sprite): Controller radio buttons**
```
[LQ●][VP●][YP●][F1●][F2●][JM●]         [status text]
  14px each, ~84px total
```
- **Controller radio buttons**: 6 buttons × 14px wide, 18px tall. Rounded rect outline. 2-char label (size-1). Filled green + bright label = active. Outline only + dim label = inactive.
- **Active states inferred from telemetry**:
  - `LQ` = `HEALTH_LQR_ACTIVE` (health_flags bit 6)
  - `VP` = `v_ref != 0.0` (vel PI is commanding a non-zero lean angle)
  - `YP` = `omega_cmd_rds != 0.0` (yaw PI is active)
  - `F1` = `ff1_out != 0.0` (hip reaction FF active)
  - `F2` = `ff2_out != 0.0` (gravity comp FF active)
  - `JM` = `jump_state != 0` (jump FSM running)

---

## Additional Smooth Animation Opportunities

Beyond the mode banner, these regions can also carry animations:

| Where | Animation | Trigger |
|---|---|---|
| **AH border glow** | Border color shifts green→yellow→red based on `abs(pitch_rad)` severity | Continuous, computed each frame |
| **Hip arc needles** | Linear interpolation toward target each frame (~0.2 lerp factor) to smooth out data jitter | Continuous |
| **Controller buttons** | Brief 150 ms highlight flash when a button transitions active↔inactive | Edge-triggered on state change |
| **ESTOP fault text** | Horizontal "shake" oscillation on the fault description text (±3px, ~5 Hz sin) | While in ESTOP |
| **Profile indicator** | 300 ms shimmer pulse when profile changes (brightness ramp on the profile circle) | Edge-triggered on profile change |
| **Heartbeat ripple** | Ripple ring that expands from heartbeat dot (2 expanding circles that fade) | Each telemetry packet received |
| **Velocity bar trail** | Footer velocity bar keeps a ghost of last peak value that decays over 500 ms | Continuous decay |

---

## Icon Drawing Functions (new helpers in main.cpp)

```cpp
void drawWifiIcon(TFT_eSprite* spr, int x, int y, bool active);          // classic arc WiFi
void drawCellBarsIcon(TFT_eSprite* spr, int x, int y, bool active);      // 4 cell bars
void drawHipJointIcon(TFT_eSprite* spr, int x, int y);                   // two-line joint
void drawWheelIcon(TFT_eSprite* spr, int x, int y);                      // circle + 4 spokes
void drawMiniArcGauge(TFT_eSprite* spr, int cx, int cy, int r,
                      float actual_pct, float cmd_pct, uint16_t color);  // dual needle
void drawControllerButton(TFT_eSprite* spr, int x, int y,
                          const char* label, bool active);               // 14×18 button
```

---

## Gradient Animation Functions (new helpers in main.cpp)

```cpp
void bannerGradientDiagonal(TFT_eSprite* spr, uint16_t col_a, uint16_t col_b, float phase, float freq);
void bannerPulseBreathing(TFT_eSprite* spr, uint16_t col_hi, uint16_t col_lo, float phase);
void bannerPulseRadial(TFT_eSprite* spr, uint16_t col_hi, uint16_t col_lo, float phase);
void bannerSpotlightSweep(TFT_eSprite* spr, uint16_t base_col, float phase);
void bannerVerticalSweep(TFT_eSprite* spr, uint16_t col_a, uint16_t col_b, float phase);
void bannerStrobe(TFT_eSprite* spr, uint16_t col, float phase);
```

---

## Files to Modify

**Primary file**: `firmware/robot_teensy/esp32/src/main.cpp`

1. **Sprite declarations (~line 551)**: Add `TFT_eSprite banner_sprite`, `hip_sprite`, `footer_sprite`; update size constants for `ah_sprite` (160×148) and `wm_sprite` (80×148).

2. **`initEngineeringDisplay()` (~line 1207)**: Create and initialize all 5 sprites with 16-bit depth. Update divider positions: vertical at x=160, x=240 (was x=170, x=250); horizontal at y=192 (was y=190).

3. **`drawModeBanner()`**: Convert to render into `banner_sprite`, add per-mode animation helper calls, push via `banner_sprite.pushSprite(0, 0)`.

4. **`drawArtificialHorizon()`**: Resize to 160×148, adjust `pushSprite(0, 44)`.

5. **`drawHipBars()`** → rename `drawHipPanel()`: Render into `hip_sprite` with dual-needle arc gauges, joint icon, current bars, health dots; push via `hip_sprite.pushSprite(160, 44)`.

6. **`drawWheelMotors()`**: Render into resized `wm_sprite`; add wheel icon, command tick marks on velocity bars, full-width speed bar (moves from footer); push via `wm_sprite.pushSprite(240, 44)`.

7. **`drawFooter()`**: Convert to render into `footer_sprite`; replace text labels with WiFi/cell icons; add controller radio buttons row; **remove velocity bar** (now in wheel sprite); push via `footer_sprite.pushSprite(0, 192)`.

8. **`update_display()`**: No changes to call sequence needed.

---

## Telemetry Fields Used (all already in `TelemetryPayload`, no protocol changes)

| Display use | Field | Offset |
|---|---|---|
| Hip actual L/R | `hip_l_pos_rad`, `hip_r_pos_rad` | [4] |
| Hip commanded L/R | `hip_l_cmd_pos_rad`, `hip_r_cmd_pos_rad` | [154] |
| Hip current L/R | `hip_l_current_a`, `hip_r_current_a` | [42] |
| Health dots | `health_flags` (HEALTH_HIP_L/R_OK, HEALTH_LQR_ACTIVE) | [222] |
| Velocity command | `v_ref` | [194] |
| Yaw command | `omega_cmd_rds` | [194+8] |
| FF1/FF2 active | `ff1_out`, `ff2_out` | [214] |
| Jump state | `jump_state` | [225] |
| Wheel avg velocity | `wheel_vel_avg_ms` | [4] |

---

## Verification

1. Flash to ESP32; confirm engineering screen appears with new 5-sprite layout.
2. Verify **no flicker** on banner, hip panel, or footer (sprites push atomically).
3. Cycle through all 8 robot modes via RC or serial — confirm each banner animation is distinct and smooth.
4. Move hips — verify both actual (cyan) and commanded (yellow) needles move independently on arc gauges.
5. Command a forward velocity — verify speed bar in wheel sprite responds to `v_ref`, and the actual `wheel_vel_avg_ms` updates.
6. Check ~30 Hz update rate maintained in serial log (no regression from additional sprites).
7. Confirm face personality switch still works (personality enum path unchanged).
