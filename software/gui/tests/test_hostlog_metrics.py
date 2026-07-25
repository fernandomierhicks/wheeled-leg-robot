import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from analysis.param_sidecar import load_host_param_sidecar
from analysis.wlog_metrics import (
    DecodedRun, TELEM_VERSION, _SCALAR_FIELDS, compute_metrics, decode_hostlog,
    decode_run,
)


def telemetry(timestamp_ms: int, loop_count: int, host_ns: int, **updates) -> dict:
    record = {name: 0 for name in _SCALAR_FIELDS}
    record.update({
        "ptype": 0x01,
        "version": TELEM_VERSION,
        "timestamp_ms": timestamp_ms,
        "loop_count": loop_count,
        "host_monotonic_ns": host_ns,
        "gain_sched_alpha": 0.25,
    })
    record.update(updates)
    return record


class HostLogDecodeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "host.jsonl"

    def tearDown(self):
        self.temp.cleanup()

    def write_records(self, records, partial_tail=""):
        body = "".join(json.dumps(record) + "\n" for record in records)
        self.path.write_text(body + partial_tail, encoding="utf-8")

    def test_decodes_telem_and_ignores_non_telem_and_partial_tail(self):
        self.write_records([
            {"record_type": "capture_header", "telem_version": TELEM_VERSION},
            telemetry(100, 1, 1_000),
            {"ptype": 0x04, "log_msg": "hello", "host_monotonic_ns": 2_000},
            telemetry(120, 2, 3_000),
            telemetry(160, 3, 4_000),
        ], partial_tail='{"ptype": 1')

        run = decode_hostlog(self.path)
        self.assertEqual(run.source_kind, "host")
        self.assertEqual(run.count, 3)
        self.assertEqual(run.sample_rate_hz, 33)
        np.testing.assert_allclose(run.t_s, [0.0, 0.02, 0.06])
        np.testing.assert_array_equal(run.t_micros, [100000, 120000, 160000])
        self.assertEqual(decode_run(self.path).source_kind, run.source_kind)

    def test_rejects_interior_malformed_json(self):
        self.path.write_text(
            json.dumps(telemetry(100, 1, 1_000)) + "\n{bad}\n" +
            json.dumps(telemetry(120, 2, 2_000)) + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "line 2: malformed JSON"):
            decode_hostlog(self.path)

    def test_rejects_backward_firmware_time(self):
        self.write_records([
            telemetry(1000, 1, 1_000),
            telemetry(900, 2, 2_000),
        ])
        with self.assertRaisesRegex(ValueError, "time moved backward"):
            decode_hostlog(self.path)

    def test_uint32_timestamp_rollover_stays_monotonic(self):
        self.write_records([
            telemetry(0xFFFFFFF0, 1, 1_000),
            telemetry(4, 2, 2_000),
            telemetry(24, 3, 3_000),
        ])
        run = decode_hostlog(self.path)
        np.testing.assert_allclose(run.t_s, [0.0, 0.02, 0.04])

    def test_reconstructs_parameter_snapshot_and_changes(self):
        self.write_records([
            telemetry(100, 1, 1_000),
            {"ptype": 0x06, "host_monotonic_ns": 1_500, "param_id": 7,
             "param_name": "gain", "param_value": 1.0},
            telemetry(120, 2, 2_000),
            {"ptype": 0x06, "host_monotonic_ns": 2_500, "param_id": 7,
             "param_name": "gain", "param_value": 2.0},
            telemetry(140, 3, 3_000),
        ])
        sidecar = load_host_param_sidecar(self.path)
        self.assertIsNotNone(sidecar)
        np.testing.assert_allclose(
            sidecar.series("gain", np.asarray([100000, 120000, 140000], dtype=np.uint32)),
            [1.0, 2.0, 2.0],
        )

    def test_repeated_host_parameter_report_is_not_a_change(self):
        self.write_records([
            telemetry(100, 1, 1_000),
            {"ptype": 0x06, "host_monotonic_ns": 1_500, "param_id": 7,
             "param_name": "gain", "param_value": 1.0},
            telemetry(120, 2, 2_000),
            {"ptype": 0x06, "host_monotonic_ns": 2_500, "param_id": 7,
             "param_name": "gain", "param_value": 1.0},
            telemetry(140, 3, 3_000),
        ])
        sidecar = load_host_param_sidecar(self.path)
        self.assertEqual(len(sidecar.events), 1)
        self.assertEqual(sidecar.events[0].event, "DUMP")


class IrregularMetricTests(unittest.TestCase):
    def test_integrals_rms_and_health_fraction_are_time_weighted(self):
        t = np.asarray([0.0, 1.0, 3.0])
        fields = {name: np.zeros(3) for name in _SCALAR_FIELDS}
        fields["pitch_rad"] = np.ones(3)
        fields["theta_ref"] = np.zeros(3)
        fields["health_flags"] = np.asarray([0, 1 << 7, 1 << 7])
        run = DecodedRun(
            path=Path("synthetic.jsonl"), telem_version=TELEM_VERSION,
            sample_rate_hz=1, count=3,
            t_micros=np.asarray([0, 1_000_000, 3_000_000], dtype=np.uint32),
            t_s=t, fields=fields, has_gain_sched_alpha=False, source_kind="host",
        )
        metrics = compute_metrics(run)
        self.assertEqual(metrics["ise_pitch"], 3.0)
        self.assertAlmostEqual(metrics["rms_pitch_deg"], 57.2958, places=4)
        self.assertAlmostEqual(metrics["health_fractions"]["vel_pi_sat_frac"], 0.8333, places=4)


if __name__ == "__main__":
    unittest.main()
