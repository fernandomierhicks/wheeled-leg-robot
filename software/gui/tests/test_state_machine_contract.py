import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
CONTRACT_PATH = ROOT / "firmware/robot_teensy/teensy/test/state_machine_contract.json"
STATE_HEADER = ROOT / "firmware/robot_teensy/teensy/src/generated_robot_state.h"
STATE_MACHINE = ROOT / "firmware/robot_teensy/teensy/src/state_machine.cpp"
TEENSY_MAIN = ROOT / "firmware/robot_teensy/teensy/src/main.cpp"
HIP_MOTORS = ROOT / "firmware/robot_teensy/teensy/lib/HipMotors/hip_motors.cpp"
WHEEL_MOTORS = ROOT / "firmware/robot_teensy/teensy/lib/WheelMotors/wheel_motors.cpp"
COMM_PROTOCOL = ROOT / "firmware/robot_teensy/shared/comm_protocol.h"
GUI_MAIN = ROOT / "software/gui/main.py"


class StateMachineContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))

    @staticmethod
    def _function_body(source: str, name: str) -> str:
        brace = None
        for match in re.finditer(rf"\b{re.escape(name)}\s*\(", source):
            paren = source.index("(", match.start())
            paren_depth = 0
            close = None
            for pos in range(paren, len(source)):
                if source[pos] == "(":
                    paren_depth += 1
                elif source[pos] == ")":
                    paren_depth -= 1
                    if paren_depth == 0:
                        close = pos
                        break
            if close is not None:
                candidate = close + 1
                while source[candidate].isspace():
                    candidate += 1
                if source[candidate] == "{":
                    brace = candidate
                    break
        if brace is None:
            raise AssertionError(f"function definition not found: {name}")
        depth = 0
        for end in range(brace, len(source)):
            if source[end] == "{":
                depth += 1
            elif source[end] == "}":
                depth -= 1
                if depth == 0:
                    return source[brace:end + 1]
        raise AssertionError(f"unterminated function {name}")

    def test_state_numeric_ids_are_frozen(self):
        source = STATE_HEADER.read_text(encoding="utf-8")
        actual = {
            name: int(value)
            for name, value in re.findall(r"STATE_([A-Z_]+)\s*=\s*(\d+)", source)
        }
        self.assertEqual(actual, self.contract["states"])

    def test_transition_order_and_priority_are_frozen(self):
        source = STATE_MACHINE.read_text(encoding="utf-8")
        actual = [
            [state, guard, target]
            for state, guard, target in re.findall(
                r"S_([A-Z_]+)\s*->addTransition\(\s*([a-zA-Z0-9_]+)\s*,\s*S_([A-Z_]+)\s*\)",
                source,
            )
        ]
        self.assertEqual(actual, self.contract["transitions"])

    def test_every_state_has_an_emergency_path_except_estop(self):
        transitions = self.contract["transitions"]
        for state in self.contract["states"]:
            if state == "ESTOP":
                continue
            with self.subTest(state=state):
                self.assertTrue(
                    any(src == state and dst == "ESTOP" for src, _, dst in transitions),
                    f"{state} has no transition to ESTOP",
                )

    def test_all_energetic_states_abort_through_disarming(self):
        transitions = self.contract["transitions"]
        for state in ("RUNNING", "JUMPING", "STANDING_UP"):
            with self.subTest(state=state):
                self.assertIn([state, "req_disarm_running", "DISARMING"], transitions)
        self.assertIn(
            ["CALIBRATION", "req_disarm_calibration", "DISARMING"],
            transitions,
        )

    def test_safety_priority_precedes_abort_and_completion(self):
        by_state = {}
        for source, guard, target in self.contract["transitions"]:
            by_state.setdefault(source, []).append((guard, target))
        for state in ("RUNNING", "JUMPING", "STANDING_UP", "DISARMING"):
            guards = [guard for guard, _ in by_state[state]]
            self.assertLess(guards.index("req_estop"), guards.index("motor_feedback_fault"))
            if "req_disarm_running" in guards:
                self.assertLess(guards.index("motor_feedback_fault"), guards.index("req_disarm_running"))
            if state in ("JUMPING", "STANDING_UP"):
                completion = "jump_done" if state == "JUMPING" else "standup_captured"
                self.assertLess(guards.index("req_disarm_running"), guards.index(completion))

    def test_roll_watchdog_is_reposition_severity_in_firmware_and_gui(self):
        comm_source = COMM_PROTOCOL.read_text(encoding="utf-8")
        gui_source = GUI_MAIN.read_text(encoding="utf-8")

        reposition_cases = re.search(
            r"((?:\s*case FAULT_[A-Z_]+:\s*)+)return FAULT_SEVERITY_REPOSITION;",
            comm_source,
        )
        self.assertIsNotNone(reposition_cases)
        self.assertIn("case FAULT_ROLL_WATCHDOG:", reposition_cases.group(1))
        self.assertRegex(gui_source, r'0x0E:\s*"REPOSITION"\s*,\s*# ROLL_WATCHDOG')

    def test_motor_flags_do_not_create_an_implicit_firmware_safety_mode(self):
        sources = (
            STATE_MACHINE,
            ROOT / "firmware/robot_teensy/teensy/src/control_loop.cpp",
            TEENSY_MAIN,
        )
        for path in sources:
            with self.subTest(path=path.name):
                self.assertNotIn(
                    "stateMachine_no_motors_active",
                    path.read_text(encoding="utf-8"),
                )

    def test_live_motor_disable_writes_cut_each_physical_axis(self):
        main_source = TEENSY_MAIN.read_text(encoding="utf-8")
        hip_source = HIP_MOTORS.read_text(encoding="utf-8")
        wheel_source = WHEEL_MOTORS.read_text(encoding="utf-8")

        for call in (
            "hip_motor_disable_L();", "hip_motor_disable_R();",
            "wheel_motor_disable_L();", "wheel_motor_disable_R();",
        ):
            self.assertIn(call, main_source)
        self.assertIn("send_raw(AK45_ID_L, cmd)", self._function_body(hip_source, "hip_motor_disable_L"))
        self.assertIn("send_raw(AK45_ID_R, cmd)", self._function_body(hip_source, "hip_motor_disable_R"))
        self.assertIn("AXIS_IDLE", self._function_body(wheel_source, "wheel_motor_disable_L"))
        self.assertIn("AXIS_IDLE", self._function_body(wheel_source, "wheel_motor_disable_R"))

if __name__ == "__main__":
    unittest.main()
