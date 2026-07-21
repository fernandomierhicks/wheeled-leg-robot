import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
CONTRACT_PATH = ROOT / "firmware/robot_teensy/teensy/test/state_machine_contract.json"
STATE_HEADER = ROOT / "firmware/robot_teensy/teensy/src/generated_robot_state.h"
STATE_MACHINE = ROOT / "firmware/robot_teensy/teensy/src/state_machine.cpp"


class StateMachineContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))

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


if __name__ == "__main__":
    unittest.main()
