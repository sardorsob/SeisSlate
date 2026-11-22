import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from obspy import UTCDateTime, read


FIXTURES = Path(__file__).resolve().parent
TOLERANCE = 1e-9


class SyntheticFixtureTest(unittest.TestCase):
    def test_generator_reproduces_frozen_fixture_with_readable_mseed(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "synthetic"
            run = subprocess.run(
                [sys.executable, "-B", str(FIXTURES / "generate_synthetic.py"), str(output)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(
                sorted(p.name for p in output.iterdir()),
                ["baseline.csv", "calibration.mseed", "expected.json", "original.mseed"],
            )
            for path in output.iterdir():
                self.assertEqual(path.read_bytes(), (FIXTURES / "synthetic" / path.name).read_bytes())

            expected = json.loads((output / "expected.json").read_text())
            for role, filename in (("original", "original.mseed"), ("calibration", "calibration.mseed")):
                path = output / filename
                source = expected["sources"][role]
                self.assertEqual(source["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
                trace = read(str(path))
                self.assertEqual([tr.id for tr in trace], [source["trace_id"]] * len(trace))
                self.assertEqual([tr.stats.npts for tr in trace], [s["npts"] for s in source["segments"]])
                self.assertEqual(
                    [tr.stats.starttime for tr in trace],
                    [UTCDateTime(s["start_utc"]) for s in source["segments"]],
                )

            original = read(str(output / "original.mseed"))
            calibration = read(str(output / "calibration.mseed"))
            self.assertEqual(len(original), 2)
            self.assertEqual(len(calibration), 1)
            self.assertEqual([tr.stats.sampling_rate for tr in (*original, *calibration)], [1000] * 3)
            self.assertEqual(original[0].stats.endtime, UTCDateTime("2000-01-01T00:00:02.149Z"))
            self.assertEqual(original[1].stats.starttime, UTCDateTime("2000-01-01T00:00:02.200Z"))

            probe = expected["snuffler_probe"]
            self.assertEqual(probe["blow_id"], "B01")
            self.assertEqual(probe["original_sample_index"], 360)
            self.assertEqual(UTCDateTime(probe["original_pick_utc"]), original[0].stats.starttime + 0.360)
            self.assertEqual(UTCDateTime(probe["nudged_pick_utc"]), original[0].stats.starttime + 0.361)
            self.assertLess(abs(original[0].data[359]), 20)
            self.assertGreater(abs(original[0].data[360]), 500)
            self.assertGreater(abs(original[0].data[1225]), 500)
            self.assertLess(abs(original[0].data[1345]), 20)
            self.assertGreater(abs(calibration[0].data[800]), 500)
            self.assertGreater(abs(calibration[0].data[880]), 500)
            self.assertGreater(abs(calibration[0].data[2145]), 500)

    def test_reproduces_notebook_fits_and_strict_range_boundaries(self):
        base = FIXTURES / "synthetic"
        self.assertTrue((base / "baseline.csv").exists(), "Generate the paired fixture and baseline")
        with (base / "baseline.csv").open(newline="") as file:
            rows = {row["blow_id"]: row for row in csv.DictReader(file)}
        expected = json.loads((base / "expected.json").read_text())
        self.assertEqual(list(rows), [f"B{i:02d}" for i in range(1, 9)])
        self.assertEqual(
            [row["blow_id"] for row in rows.values() if row["strict_included"] == "true"],
            ["B01", "B02", "B03", "B08"],
        )

        all_fit, strict = (expected["fits"][key] for key in ("all", "strict"))
        self.assertEqual(all_fit["blow_ids"], ["B01", "B02", "B03", "B04", "B05", "B06", "B08"])
        self.assertEqual(strict["blow_ids"], ["B01", "B02", "B03", "B08"])
        self.assertAlmostEqual(all_fit["slope_m_per_s"], 241500 / 919, delta=TOLERANCE)
        self.assertAlmostEqual(all_fit["intercept_m"], -2302 / 919, delta=TOLERANCE)
        self.assertAlmostEqual(strict["slope_m_per_s"], 500, delta=TOLERANCE)
        self.assertAlmostEqual(strict["intercept_m"], 2, delta=TOLERANCE)

        for row, lag, offset, residual, strict_residual in zip(
            [rows[blow_id] for blow_id in all_fit["blow_ids"]],
            [-0.04, 0.015, 0.03, 0.095, -0.06, 0.08, 0],
            [-18, 9.5, 17, -10.5, -28, 42, 2],
            [-4.983677910772579, 8.063112078346029, 11.62132752992383,
             -32.959738846572364, -9.72796517954298, 23.482045701849838, 4.504896626768226],
            [0, 0, 0, -60, 0, 0, 0],
            strict=True,
        ):
            self.assertEqual(float(row["dt_s"]), lag)
            self.assertAlmostEqual(float(row["dx_m"]), offset, delta=TOLERANCE)
            self.assertAlmostEqual(float(row["all_residual_m"]), residual, delta=TOLERANCE)
            self.assertAlmostEqual(float(row["strict_residual_m"]), strict_residual, delta=TOLERANCE)

        self.assertEqual(rows["B04"]["original_truth_utc"], "2000-01-01T00:00:01.225000Z")
        self.assertEqual(rows["B04"]["original_pick_utc"], "2000-01-01T00:00:01.345000Z")
        self.assertEqual(rows["B07"]["original_truth_utc"], "2000-01-01T00:00:02.165000Z")
        self.assertEqual(rows["B07"]["original_pick_utc"], "")
        self.assertEqual(rows["B07"]["calibration_pick_utc"], "2000-01-01T00:00:02.145000Z")
        self.assertEqual(rows["B07"]["dt_s"], "")
        self.assertEqual(rows["B07"]["all_residual_m"], "")
        self.assertEqual(rows["B07"]["strict_residual_m"], "")
