import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
CONTRACT_PATH = ROOT / "firmware/robot_teensy/teensy/test/state_machine_contract.json"
STATE_HEADER = ROOT / "firmware/robot_teensy/teensy/src/robot_state.h"
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


if __name__ == "__main__":
    unittest.main()
