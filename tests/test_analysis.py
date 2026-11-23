import csv
import json
import unittest
from dataclasses import replace
from pathlib import Path


FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "synthetic"
TOLERANCE = 1e-9


def by_id(fit, blow_id):
    return next(row for row in fit.observations if row.blow_id == blow_id)


class AnalysisTest(unittest.TestCase):
    def setUp(self):
        from seisslate.analysis import analyze
        from seisslate.example import load_example

        self.analyze = analyze
        self.catalog, history = load_example()
        self.initial = history.current.state

    def check_close(self, value, expected):
        self.assertAlmostEqual(value, expected, delta=TOLERANCE)

    def analyze_applied(self, applied, *, reverse=False):
        return self.analyze(self.catalog, self.initial, applied, reverse=reverse)

    def test_frozen_fits_reproduce_reference_membership_coefficients_and_residuals(self):
        result = self.analyze_applied(self.initial)
        manifest = json.loads((FIXTURE / "expected.json").read_text(encoding="utf-8"))
        with (FIXTURE / "baseline.csv").open(newline="", encoding="utf-8") as file:
            expected_rows = list(csv.DictReader(file))

        for kind, fit in (("all", result.baseline_all), ("strict", result.baseline_strict)):
            expected_fit = manifest["fits"][kind]
            self.assertEqual(fit.blow_ids, tuple(expected_fit["blow_ids"]))
            self.check_close(fit.slope_m_per_s, expected_fit["slope_m_per_s"])
            self.check_close(fit.intercept_m, expected_fit["intercept_m"])
            for expected in expected_rows:
                row = by_id(fit, expected["blow_id"])
                residual = expected[f"{kind}_residual_m"]
                if residual:
                    self.check_close(row.residual_m, float(residual))
                else:
                    self.assertIsNone(row.residual_m)

        self.check_close(result.baseline_all.slope_m_per_s, 241500 / 919)
        self.check_close(result.baseline_all.intercept_m, -2302 / 919)
        self.check_close(result.reviewed.slope_m_per_s, 500)
        self.check_close(result.reviewed.intercept_m, 2)
        self.assertIsNone(result.field_velocity_m_per_s)

    def test_exact_strict_bounds_and_zero_lag(self):
        from seisslate.analysis import in_strict_lag_window

        for lag_ns, eligible in (
            (-60_000_001, False), (-60_000_000, False), (-59_999_999, True),
            (0, True), (79_999_999, True), (80_000_000, False), (80_000_001, False),
        ):
            with self.subTest(lag_ns=lag_ns):
                self.assertIs(in_strict_lag_window(lag_ns), eligible)

        reviewed = self.analyze_applied(self.initial).reviewed
        self.assertEqual(by_id(reviewed, "B08").lag_ns, 0)
        self.assertTrue(by_id(reviewed, "B08").effective_included)
        for blow_id, lag_ns in (("B05", -60_000_000), ("B06", 80_000_000)):
            row = by_id(reviewed, blow_id)
            self.assertEqual(row.lag_ns, lag_ns)
            self.assertFalse(row.requested_included)
            self.assertFalse(row.effective_included)

    def test_nudge_preserves_partner_and_recalculates_other_residuals(self):
        from seisslate.review import nudge_pick

        baseline = self.analyze_applied(self.initial)
        applied = nudge_pick(self.initial, self.catalog, "B01", "original", 1)
        result = self.analyze_applied(applied)
        self.assertEqual(applied.blows[0].calibration, self.initial.blows[0].calibration)
        self.assertEqual(applied.blows[0].original.original_ref, self.initial.blows[0].original.original_ref)
        self.assertEqual(result.baseline_all, baseline.baseline_all)
        self.assertEqual(result.baseline_strict, baseline.baseline_strict)
        self.assertEqual(by_id(result.reviewed, "B01").lag_ns, -39_000_000)
        self.check_close(result.reviewed.slope_m_per_s, 148750 / 293)
        self.check_close(result.reviewed.intercept_m, 546 / 293)
        self.assertEqual(by_id(result.reviewed, "B02").lag_ns, 15_000_000)
        self.assertNotAlmostEqual(by_id(result.reviewed, "B02").residual_m, 0)

    def test_crossing_a_bound_preserves_membership_until_an_explicit_edit(self):
        from seisslate.review import move_pick, set_inclusion

        first_arrival = self.initial.blows[3].original.original_ref
        corrected = replace(first_arrival, sample_index=1225)
        moved = move_pick(self.initial, self.catalog, "B04", "original", corrected)
        result = self.analyze_applied(moved)
        self.assertEqual(result.reviewed.blow_ids, ("B01", "B02", "B03", "B08"))
        row = by_id(result.reviewed, "B04")
        self.assertEqual(row.lag_ns, -25_000_000)
        self.assertFalse(row.requested_included)
        self.check_close(row.residual_m, 0)
        included = set_inclusion(moved, "B04", True, "Corrected arrival")
        result = self.analyze_applied(included)
        self.assertEqual(result.reviewed.blow_ids, ("B01", "B02", "B03", "B04", "B08"))
        self.assertEqual(result.baseline_strict.blow_ids, ("B01", "B02", "B03", "B08"))

    def test_receiver_reversal_flips_lag_distance_intercept_and_residual_at_fixed_membership(self):
        forward = self.analyze_applied(self.initial)
        reversed_result = self.analyze_applied(self.initial, reverse=True)
        for original, reversed_fit in (
            (forward.baseline_all, reversed_result.baseline_all),
            (forward.baseline_strict, reversed_result.baseline_strict),
            (forward.reviewed, reversed_result.reviewed),
        ):
            self.assertEqual(reversed_fit.blow_ids, original.blow_ids)
            self.check_close(reversed_fit.slope_m_per_s, original.slope_m_per_s)
            self.check_close(reversed_fit.intercept_m, -original.intercept_m)
            for row in original.observations:
                other = by_id(reversed_fit, row.blow_id)
                if row.lag_ns is not None:
                    self.assertEqual(other.lag_ns, -row.lag_ns)
                    self.check_close(other.residual_m, -row.residual_m)
                self.check_close(other.dx_m, -row.dx_m)

    def test_import_mapping_uses_original_times_to_freeze_fits_and_reports_displacement(self):
        from seisslate.review import import_pick

        first = self.initial.blows[0]
        ref = first.original.original_ref
        reported_ns = first.original.imported_ns + 200_000
        unresolved = import_pick(self.catalog, ref.channel, reported_ns, source_hash=ref.source_hash)
        self.assertIsNone(unresolved.reviewed_ref)
        state = replace(self.initial, blows=(replace(first, original=unresolved), *self.initial.blows[1:]))
        result = self.analyze(self.catalog, state, state)
        self.assertEqual(by_id(result.baseline_strict, "B01").lag_ns, -39_800_000)
        self.assertEqual(by_id(result.reviewed, "B01").status, "unresolved")
        self.assertFalse(by_id(result.reviewed, "B01").effective_included)

        mapped = import_pick(self.catalog, ref.channel, reported_ns, ref, source_hash=ref.source_hash)
        state = replace(state, blows=(replace(first, original=mapped), *state.blows[1:]))
        result = self.analyze(self.catalog, state, state)
        self.assertEqual(by_id(result.baseline_strict, "B01").lag_ns, -39_800_000)
        reviewed = by_id(result.reviewed, "B01")
        self.assertEqual(reviewed.original_mapping_offset_ns, -200_000)
        self.assertEqual(reviewed.calibration_mapping_offset_ns, 0)
        self.assertEqual(reviewed.lag_ns, -40_000_000)

    def test_frozen_strict_membership_is_independent_of_initial_review_flags(self):
        from seisslate.review import set_inclusion

        state = set_inclusion(self.initial, "B06", True, "Include the upper-bound observation")
        state = set_inclusion(state, "B03", False, "Exclude this observation from review")
        result = self.analyze(self.catalog, state, state)
        upper_bound = by_id(result.baseline_strict, "B06")
        self.assertEqual(upper_bound.lag_ns, 80_000_000)
        self.assertFalse(upper_bound.requested_included)
        self.assertEqual(result.baseline_strict.blow_ids, ("B01", "B02", "B03", "B08"))
        self.assertTrue(by_id(result.reviewed, "B06").requested_included)
        self.assertEqual(result.reviewed.blow_ids, ("B01", "B02", "B06", "B08"))

    def test_invalid_observations_and_fit_only_exclusions_are_distinct(self):
        from seisslate.review import set_inclusion, set_pair_status

        edited = set_pair_status(self.initial, "B02", "ambiguous", "Both close blows are plausible")
        edited = set_inclusion(edited, "B03", False, "Sensitivity comparison")
        result = self.analyze_applied(edited)
        ambiguous = by_id(result.reviewed, "B02")
        excluded = by_id(result.reviewed, "B03")
        missing = by_id(result.reviewed, "B07")
        self.assertEqual(ambiguous.status, "ambiguous")
        self.assertIsNone(ambiguous.lag_ns)
        self.assertTrue(ambiguous.requested_included)
        self.assertFalse(ambiguous.effective_included)
        self.assertEqual(excluded.status, "excluded")
        self.assertEqual(excluded.lag_ns, 30_000_000)
        self.check_close(excluded.residual_m, 0)
        self.assertEqual(missing.status, "missing")
        self.assertIsNone(missing.lag_ns)
        self.assertIsNone(missing.residual_m)

    def test_masked_onset_preserves_partner_and_requested_membership(self):
        from seisslate.review import mask_spans, restore_spans
        from seisslate.sources import SampleSpan

        first = self.initial.blows[0]
        ref = first.original.reviewed_ref
        span = SampleSpan(ref.segment_id, ref.sample_index, ref.sample_index + 1)
        masked = mask_spans(self.initial, (span,))
        result = self.analyze_applied(masked)
        row = by_id(result.reviewed, "B01")
        self.assertTrue(row.requested_included)
        self.assertFalse(row.effective_included)
        self.assertEqual(row.status, "masked")
        self.assertIsNone(row.lag_ns)
        self.assertIsNone(row.residual_m)
        self.assertEqual(masked.blows[0].calibration, first.calibration)
        self.assertEqual(by_id(result.baseline_strict, "B01").lag_ns, -40_000_000)
        restored = self.analyze_applied(restore_spans(masked, (span,))).reviewed
        self.assertEqual(by_id(restored, "B01").lag_ns, -40_000_000)

    def test_nonfinite_geometry_is_rejected_without_losing_observed_lag(self):
        first = replace(self.initial.blows[0], distance_calibration_m=float("inf"))
        edited = replace(self.initial, blows=(first, *self.initial.blows[1:]))
        row = by_id(self.analyze_applied(edited).reviewed, "B01")
        self.assertEqual(row.status, "invalid_geometry")
        self.assertEqual(row.lag_ns, -40_000_000)
        self.assertIsNone(row.dx_m)
        self.assertIsNone(row.residual_m)
        self.assertTrue(row.requested_included)
        self.assertFalse(row.effective_included)

    def test_support_distinguishes_two_point_line_from_rank_deficiency(self):
        first, second = self.initial.blows[:2]
        two = replace(self.initial, blows=(first, second))
        fit = self.analyze_applied(two).reviewed
        self.assertEqual((fit.status, fit.rank, fit.residual_df), ("two_point", 2, 0))
        self.check_close(fit.slope_m_per_s, 500)
        self.check_close(fit.intercept_m, 2)
        one = replace(two, blows=(first,))
        fit = self.analyze_applied(one).reviewed
        self.assertEqual((fit.status, fit.rank), ("insufficient_support", 1))
        self.assertIsNone(fit.slope_m_per_s)
        same_lag = replace(second, original=first.original, calibration=first.calibration)
        fit = self.analyze_applied(replace(two, blows=(first, same_lag))).reviewed
        self.assertEqual((fit.status, fit.rank), ("rank_deficient", 1))
        self.assertIsNone(fit.intercept_m)


if __name__ == "__main__":
    unittest.main()
