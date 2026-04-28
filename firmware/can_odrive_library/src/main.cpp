// ODriveArduino CAN proof-of-concept
// Hardware: Arduino UNO R4 WiFi + ODrive 3.6-56
//
// Wiring: UNO R4 WiFi CAN_TX/CAN_RX → CAN transceiver → ODrive CAN H/L
// ODrive node IDs: axis 0 = NODE_0, axis 1 = NODE_1
//
// Before flashing:
//   - Set odrive CAN baud to match CAN_BAUD below (default 250 kbps)
//   - Run odrv0.axis0.requested_state = AXIS_STATE_ENCODER_INDEX_SEARCH
//     (or FULL_CALIBRATION) once before this sketch, then save_configuration()
//   - Both axes must already be calibrated

#include <Arduino.h>
#include <Arduino_CAN.h>
#include <ODriveCAN.h>
#include <ODriveHardwareCAN.hpp>

// ── Config ──────────────────────────────────────────────────────────────────
static constexpr uint32_t CAN_BAUD  = 1000000; // match odrive CAN baud setting
static constexpr uint8_t  NODE_0    = 0;        // odrv0.axis0
static constexpr uint8_t  NODE_1    = 1;        // odrv0.axis1 (or second ODrive axis0)
static constexpr float    SPIN_VEL  = 2.0f;     // turns/s

// ── ODrive instances ─────────────────────────────────────────────────────────
static ODriveCAN odrv0(wrap_can_intf(CAN), NODE_0);
static ODriveCAN odrv1(wrap_can_intf(CAN), NODE_1);

// ── State ────────────────────────────────────────────────────────────────────
struct AxisInfo {
    float    pos = 0, vel = 0;
    uint32_t error = 0;
    uint8_t  state = 0;
};
static AxisInfo ax0, ax1;

// ── CAN receive callback (required by ODriveHardwareCAN) ────────────────────
void onCanMessage(const CanMsg& msg) {
    uint8_t node = (msg.id >> 5) & 0x3F;
    if      (node == NODE_0) onReceive(msg, odrv0);
    else if (node == NODE_1) onReceive(msg, odrv1);
}

// ── ODrive callbacks ─────────────────────────────────────────────────────────
static void on_fb0(Get_Encoder_Estimates_msg_t& m, void*) { ax0.pos = m.Pos_Estimate; ax0.vel = m.Vel_Estimate; }
static void on_fb1(Get_Encoder_Estimates_msg_t& m, void*) { ax1.pos = m.Pos_Estimate; ax1.vel = m.Vel_Estimate; }
static void on_hb0(Heartbeat_msg_t& m, void*) { ax0.error = m.Axis_Error; ax0.state = (uint8_t)m.Axis_State; }
static void on_hb1(Heartbeat_msg_t& m, void*) { ax1.error = m.Axis_Error; ax1.state = (uint8_t)m.Axis_State; }

// ── Helpers ──────────────────────────────────────────────────────────────────
static void arm_axis(ODriveCAN& odrv, const char* name, uint8_t node, AxisInfo& ax) {
    // Clear errors and wait for a heartbeat confirming error=0 (up to 300 ms)
    odrv.clearErrors();
    uint32_t t0 = millis();
    while (millis() - t0 < 300) { pumpEvents(CAN); delay(5); }

    Serial.print("  "); Serial.print(name);
    Serial.print(" after clearErrors: state="); Serial.print(ax.state);
    Serial.print(" err=0x"); Serial.println(ax.error, HEX);

    odrv.setControllerMode(CONTROL_MODE_VELOCITY_CONTROL, INPUT_MODE_PASSTHROUGH);
    delay(20);

    odrv.setState(AXIS_STATE_CLOSED_LOOP_CONTROL);

    // Poll for state 8 (closed loop), pumping CAN the whole time
    t0 = millis();
    while (millis() - t0 < 2000) {
        pumpEvents(CAN);
        if (ax.state == 8) break;
        delay(10);
    }

    Serial.print("  "); Serial.print(name);
    Serial.print(" after setState: state="); Serial.print(ax.state);
    Serial.print(" err=0x"); Serial.println(ax.error, HEX);
    if (ax.state != 8) {
        Serial.print("  WARNING: "); Serial.print(name);
        Serial.println(" did not reach closed loop!");
    }
}

// ── Setup ────────────────────────────────────────────────────────────────────
void setup() {
    Serial.begin(115200);
    delay(800);
    Serial.println("=== ODrive CAN PoC ===");

    odrv0.onFeedback(on_fb0, nullptr);
    odrv1.onFeedback(on_fb1, nullptr);
    odrv0.onStatus(on_hb0, nullptr);
    odrv1.onStatus(on_hb1, nullptr);

    if (!CAN.begin(CAN_BAUD)) {
        Serial.println("ERROR: CAN.begin() failed — check transceiver wiring");
        while (1) delay(1000);
    }
    // Note: R7FA4M1_CAN has no onReceive() — use pumpEvents(CAN) in the loop instead.
    Serial.print("CAN up at "); Serial.print(CAN_BAUD / 1000); Serial.println(" kbps");

    delay(200);
    pumpEvents(CAN);   // flush initial heartbeats

    // Print heartbeats received so far (tells us which node IDs are active)
    Serial.print("Heartbeats before arm: ax0_state="); Serial.print(ax0.state);
    Serial.print(" ax1_state="); Serial.println(ax1.state);

    Serial.println("Arming axes...");
    arm_axis(odrv0, "axis0", NODE_0, ax0);
    arm_axis(odrv1, "axis1", NODE_1, ax1);
    Serial.println("Ready — entering spin loop");
}

// ── Main loop ────────────────────────────────────────────────────────────────
void loop() {
    pumpEvents(CAN);

    static uint32_t phase_start = millis();
    static bool     spinning    = false;
    uint32_t        now         = millis();

    // Alternate: spin 3 s → stop 2 s → repeat
    if (!spinning && (now - phase_start >= 2000)) {
        Serial.println(">>> SPIN");
        odrv1.setVelocity( SPIN_VEL);
        delay(1);
        odrv0.setVelocity(-SPIN_VEL);  // opposite direction if motors are mirrored
        spinning    = true;
        phase_start = now;
    } else if (spinning && (now - phase_start >= 3000)) {
        Serial.println(">>> STOP");
        odrv1.setVelocity(0.0f);
        delay(1);
        odrv0.setVelocity(0.0f);
        spinning    = false;
        phase_start = now;
    }

    // 1 Hz status print
    static uint32_t last_print = 0;
    if (now - last_print >= 1000) {
        last_print = now;
        Serial.print("[ax0] pos="); Serial.print(ax0.pos, 2);
        Serial.print(" vel=");      Serial.print(ax0.vel, 2);
        Serial.print(" err=0x");    Serial.print(ax0.error, HEX);
        Serial.print(" | [ax1] pos="); Serial.print(ax1.pos, 2);
        Serial.print("[ax1] pos="); Serial.print(ax1.pos, 2);
        Serial.print(" vel=");      Serial.print(ax1.vel, 2);
        Serial.print(" err=0x");    Serial.println(ax1.error, HEX);
    }
}
