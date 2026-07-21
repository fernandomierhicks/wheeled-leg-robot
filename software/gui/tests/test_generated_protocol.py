import json
import subprocess
import sys
import unittest
from pathlib import Path

from tabs import generated_protocol


ROOT = Path(__file__).resolve().parents[3]
SCHEMA = ROOT / "firmware/robot_teensy/protocol/schema.json"
GENERATOR = SCHEMA.with_name("generate_protocol.py")


class GeneratedProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema = json.loads(SCHEMA.read_text(encoding="utf-8"))

    def test_generated_files_are_current(self):
        result = subprocess.run([sys.executable, str(GENERATOR), "--check"], cwd=ROOT)
        self.assertEqual(result.returncode, 0)

    def test_python_ids_match_schema(self):
        self.assertEqual(generated_protocol.STATE_NAMES,
                         {item["id"]: item["name"] for item in self.schema["states"]})
        self.assertEqual(generated_protocol.PARAM_IDS,
                         {item["symbol"]: item["id"] for item in self.schema["parameters"]})
        self.assertEqual(len(generated_protocol.PARAM_BY_NAME), len(self.schema["parameters"]))

    def test_ids_and_parameter_names_are_unique(self):
        for section in ("states", "faults", "commands", "parameters"):
            items = self.schema[section]
            self.assertEqual(len(items), len({item["id"] for item in items}), section)
            self.assertEqual(len(items), len({item["symbol"] for item in items}), section)
        params = self.schema["parameters"]
        self.assertEqual(len(params), len({item["name"] for item in params}))
        self.assertTrue(all(len(item["name"].encode("ascii")) <= 19 for item in params))

    def test_calibration_done_is_really_persistent(self):
        calib_done = next(item for item in self.schema["parameters"]
                          if item["symbol"] == "PARAM_CALIB_DONE")
        self.assertEqual(set(calib_done["flags"]), {"persistent", "readonly"})


if __name__ == "__main__":
    unittest.main()
