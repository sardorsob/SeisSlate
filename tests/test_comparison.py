"""Check the independent reference against hand-computable and raw inputs."""

import csv
import importlib.util
from fractions import Fraction
import json
import math
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

if importlib.util.find_spec("pyrocko") is not None:
    from tools import compare_measurements as reference


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "fixtures/rms-comparison"
MANIFEST = FIXTURE / "manifest.json"


@unittest.skipUnless(importlib.util.find_spec("pyrocko"), "requires the pinned Pyrocko desktop extra")
class ComparisonTests(unittest.TestCase):
    def test_independent_raw_comparison_and_exact_sample_support(self):
        bundle = reference.evaluate_manifest(MANIFEST)
        self.assertEqual(bundle["dependencies"]["pyrocko"], "2026.11.5")
        self.assertTrue(bundle["passed"])
        cases = {case["id"]: case for case in bundle["cases"]}
        rational = cases["rational-three-hz"]["reference"]
        self.assertEqual(rational["event_indices"], [0, 2])
        self.assertEqual(rational["background_indices"], [2, 4])
        self.assertEqual(rational["duration_ns"], "2000000000/3")
        self.assertEqual(rational["event_mean"], 4)
        self.assertEqual(rational["background_mean"], 8)
        self.assertEqual(rational["contrast"], 2)
        self.assertEqual(rational["units"], "raw units unknown")
        self.assertEqual(rational["epoch_status"], "unknown")
        self.assertEqual(cases["zero-event"]["reference"]["contrast"], 0)
        self.assertEqual(cases["zero-background"]["reference"]["reasons"], ["zero_background"])
        masked = cases["masked"]["reference"]
        self.assertEqual(masked["event_indices"], [0, 2])
        self.assertEqual(masked["event_reasons"], ["masked_samples"])
        self.assertEqual(masked["event_rms"], None)
        tiny = cases["tiny-contrast"]["comparison"]["contrast"]
        self.assertEqual(tiny["expected"], 1e-300)
        self.assertGreater(tiny["observed"], 0)
        self.assertLess(tiny["bound"], 1e-310)
        extreme = cases["extreme-values"]["reference"]
        self.assertEqual(extreme["event_rms"], 1e308)
        self.assertEqual(extreme["background_rms"], 5e307)
        self.assertEqual(extreme["contrast"], 2)

    def test_production_agrees_on_verified_raw_support_and_numerical_bounds(self):
        from seisslate.review import ReviewState
        from seisslate.rms import RmsWindow, RmsWindows, compute_rms_contrast
        from seisslate.sources import SampleSpan, SourceCatalog, load_source

        bundle = reference.evaluate_manifest(MANIFEST)
        for case in bundle["cases"]:
            with self.subTest(case=case["id"]):
                observed_source = case["source"]
                source = load_source(
                    FIXTURE / observed_source["path"], expected_sha256=observed_source["sha256"],
                    expected_size_bytes=observed_source["size_bytes"],
                )
                self.assertEqual(len(source.segments), len(observed_source["segments"]))
                for segment, independent in zip(source.segments, observed_source["segments"], strict=True):
                    self.assertEqual(segment.channel, independent["channel"])
                    self.assertEqual(segment.start_ns, independent["start_ns"])
                    self.assertEqual(segment.period_ns, Fraction(independent["period_ns"]))
                    self.assertEqual(segment.npts, independent["npts"])
                    self.assertEqual(reference.sample_sha256(segment.data), independent["sample_sha256"])

                identity = source.sha256, case["channel"]

                def window(name):
                    bounds = case["requests_ns"][name]
                    return RmsWindow(*identity, *map(Fraction, bounds)) if bounds else None

                masks = tuple(SampleSpan(source.segments[index].segment_id, start, stop)
                              for index, start, stop in case["masks"])
                actual = compute_rms_contrast(
                    SourceCatalog((source,)), ReviewState((), masks=masks),
                    RmsWindows(*identity, window("event"), window("background")),
                )
                expected = case["reference"]
                self.assertEqual(actual.available, expected["available"])
                self.assertEqual(actual.units, expected["units"])
                self.assertEqual(actual.epoch_status, expected["epoch_status"])
                self.assertEqual(list(actual.reasons), expected["reasons"])
                self.assertEqual(actual.samples_per_window, expected["samples_per_window"])
                self.assertEqual(actual.duration_ns, Fraction(expected["duration_ns"]) if expected["duration_ns"] else None)
                for name in ("event", "background"):
                    result = getattr(actual, name)
                    self.assertEqual(list(result.reasons), expected[f"{name}_reasons"])
                    self.assertEqual([result.span.start, result.span.stop] if result.span else None,
                                     expected[f"{name}_indices"])
                    self.assertEqual(
                        int(result.span.segment_id.rsplit(":", 1)[1]) if result.span else None,
                        expected[f"{name}_segment"],
                    )
                    self.assertTrue(reference.compare_value(result.mean, expected[f"{name}_mean"], case["relative_only"])["passed"])
                    self.assertTrue(reference.compare_value(result.rms, expected[f"{name}_rms"], case["relative_only"])["passed"])
                self.assertTrue(reference.compare_value(actual.contrast, expected["contrast"], case["relative_only"])["passed"])

    def test_tiny_ratio_cannot_pass_as_zero_and_lossy_integer_is_unavailable(self):
        self.assertFalse(reference.compare_value(0, 1e-300, True)["passed"])
        self.assertEqual(reference.exact_statistics(np.array([2**53, 2**53 + 2], dtype=np.int64))[:2],
                         (float(2**53 + 1), 1.0))
        with self.assertRaises(ArithmeticError):
            reference.exact_statistics(np.array([2**53, 2**53 + 1], dtype=np.int64))
        tiny = math.ulp(0.0)
        self.assertEqual(reference.exact_statistics(np.array([tiny, 3 * tiny]))[1], tiny)
        with self.assertRaises(ArithmeticError):
            reference.exact_statistics(np.array([tiny, 2 * tiny]))


@unittest.skipUnless(importlib.util.find_spec("pyrocko"), "requires the pinned Pyrocko desktop extra")
class MarkerAndCliTests(unittest.TestCase):
    def test_marker_roundtrip_has_same_sample_manifest_and_csv(self):
        import tempfile

        with tempfile.TemporaryDirectory() as workspace:
            output = Path(workspace) / "output"
            subprocess.run(
                [sys.executable, str(ROOT / "tools/compare_measurements.py"), str(MANIFEST),
                 "--case", "rational-three-hz", "--markers", str(FIXTURE / "snuffler.markers"),
                 "--confirm-grid-intent", "--output-dir", str(output)],
                cwd=workspace, check=True, capture_output=True, text=True,
            )
            marker_bundle = json.loads((output / "sample-manifest.json").read_text())
            numeric_case = reference.evaluate_manifest(MANIFEST, case_id="rational-three-hz")["cases"][0]
            marker_case = marker_bundle["cases"][0]
            self.assertEqual(marker_case["requests_ns"], numeric_case["requests_ns"])
            self.assertEqual(marker_case["reference"], numeric_case["reference"])
            self.assertEqual(marker_case["source"], numeric_case["source"])
            self.assertEqual(marker_case["marker_file"]["format"], "Snuffler 0.3 named phase intervals")
            with (output / "reference.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["event_indices"], "[0,2]")
            self.assertEqual(rows[0]["background_indices"], "[2,4]")
            self.assertEqual(rows[0]["contrast"], "2.0")

    def test_marker_requires_declared_grid_intent_and_rejects_lost_or_unknown_fields(self):
        import tempfile

        markers = FIXTURE / "snuffler.markers"
        with self.assertRaisesRegex(ValueError, "confirm-grid-intent"):
            reference.evaluate_manifest(MANIFEST, case_id="rational-three-hz", marker_path=markers)
        sample = markers.read_text()
        with tempfile.TemporaryDirectory() as workspace:
            candidate = Path(workspace) / "unsafe.markers"
            for changed, problem in (
                (sample.replace("00:00:00.667", "00:00:00.500"), "off-grid"),
                (sample.replace("0.6666666269302368", "0.8"), "duration"),
                (sample.replace("None\n", "None mystery\n"), "fields"),
                (sample.replace(" E ", " Q "), "E or B"),
                (sample + sample.splitlines()[1] + "\n", "Duplicate"),
                (sample.replace(".667", ""), "precision"),
                (sample.replace("0.3\n", "0.4\n"), "0.3"),
            ):
                with self.subTest(problem=problem):
                    candidate.write_text(changed)
                    with self.assertRaisesRegex(ValueError, problem):
                        reference.evaluate_manifest(
                            MANIFEST, case_id="rational-three-hz", marker_path=candidate,
                            confirm_grid_intent=True,
                        )

    def test_reference_only_run_reports_unassessed_comparison(self):
        import tempfile

        manifest = json.loads(MANIFEST.read_text())
        manifest["sources"]["integer"]["path"] = str(FIXTURE / "integer.mseed")
        del manifest["cases"][0]["expected"]
        with tempfile.TemporaryDirectory() as workspace:
            candidate = Path(workspace) / "manifest.json"
            candidate.write_text(json.dumps(manifest))
            bundle = reference.evaluate_manifest(candidate, case_id="rational-three-hz")
            self.assertEqual(bundle["comparison_status"], "unassessed")
            self.assertIsNone(bundle["passed"])
            case = bundle["cases"][0]
            self.assertEqual(case["comparison_status"], "unassessed")
            self.assertEqual(case["reference"]["contrast"], 2)

    def test_source_manifest_detects_independent_identity_and_support_disagreement(self):
        import tempfile

        manifest = json.loads(MANIFEST.read_text())
        for field, value, pattern in (
            ("sha256", "0" * 64, "SHA-256"),
            ("size_bytes", 512.0, "size"),
            ("sample_sha256", "0" * 64, "sample_sha256"),
            ("sampling_rate_hz", "4", "sampling_rate_hz"),
            ("sampling_rate_hz", 3.0, "sampling_rate_hz"),
            ("sampling_rate_hz", "300000000000000000001/100000000000000000000", "sampling_rate_hz"),
            ("channel", "XX.WRONG..EHZ", "channel"),
            ("start_ns", 946684800000000001, "start_ns"),
            ("start_ns", 946684800000000000.0, "start_ns"),
            ("npts", 1, "npts"),
            ("npts", 15.0, "npts"),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as workspace:
                candidate = Path(workspace) / "manifest.json"
                data = json.loads(json.dumps(manifest))
                data["sources"]["integer"]["path"] = str(FIXTURE / "integer.mseed")
                target = data["sources"]["integer"]
                if field not in target:
                    target = target["segments"][0]
                target[field] = value
                candidate.write_text(json.dumps(data))
                with self.assertRaisesRegex(ValueError, pattern):
                    reference.evaluate_manifest(candidate, case_id="rational-three-hz")


if __name__ == "__main__":
    unittest.main()
