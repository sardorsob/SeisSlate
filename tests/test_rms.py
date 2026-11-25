import json
import math
import unittest
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import numpy as np

from seisslate.review import ReviewState
from seisslate.rms import EpochEvidence, RmsEvidence, RmsWindow, RmsWindows, compute_rms_contrast
from seisslate.sources import NS_PER_SECOND, SampleSpan, Segment, Source, SourceCatalog


FIXTURES = json.loads(
    (Path(__file__).parents[1] / "fixtures/review-evidence/manifest.json").read_text(encoding="utf-8")
)["rms"]


class RmsTests(unittest.TestCase):
    def build(self, **overrides):
        inputs = {**FIXTURES["defaults"], **overrides}
        source, channel = inputs["source"], inputs["channel"]
        by_source = {}
        for spec in inputs["segments"]:
            source_hash = spec.get("source", source)
            data = np.tile(
                np.asarray(spec["samples"], dtype=spec.get("dtype", inputs["dtype"])), spec.get("repeat", 1),
            )
            segment = Segment(
                source_hash, spec.get("channel", channel), spec["id"], spec.get("start_ns", 0),
                NS_PER_SECOND / Fraction(spec.get("period_ns", inputs["period_ns"])), data,
            )
            by_source.setdefault(source_hash, []).append(segment)
        catalog = SourceCatalog(tuple(
            Source(Path(source_hash), source_hash, 0, tuple(segments))
            for source_hash, segments in by_source.items()
        ))

        def window(interval, source_hash, window_channel):
            if interval is None:
                return None
            return RmsWindow(source_hash, window_channel, *(Fraction(value) for value in interval))

        event = window(inputs["event_ns"], source, channel)
        background = window(
            inputs["background_ns"], inputs.get("background_source", source), inputs.get("background_channel", channel),
        )
        epoch = None
        if inputs["epoch_status"] == "conflicting":
            epoch = EpochEvidence(self.evidence(source, channel, 0, 8_000_000), conflicting=True)
        masks = tuple(SampleSpan(*mask) for mask in inputs["masks"])
        return catalog, ReviewState((), masks=masks), RmsWindows(source, channel, event, background, epoch)

    @staticmethod
    def evidence(source, channel, start, stop):
        return RmsEvidence(source, channel, start, stop, "synthetic-metadata", "ab" * 32, "Synthetic metadata record")

    def assert_fixture(self, result, expected):
        actual = {
            "event_mean": result.event.mean,
            "event_rms": result.event.rms,
            "background_mean": result.background.mean,
            "background_rms": result.background.rms,
            "background_stop_index": result.background.span.stop if result.background.span else None,
            "contrast": result.contrast,
            "duration_ns": result.duration_ns,
            "epoch_status": result.epoch_status,
            "qualified": result.qualified,
            "samples_per_window": result.samples_per_window,
            "units": result.units,
        }
        comparison = FIXTURES["comparison"]
        for field, value in expected.items():
            if field == "reasons":
                self.assertTrue(set(value).issubset(result.reasons), result.reasons)
            elif isinstance(value, str) and field == "duration_ns":
                self.assertEqual(actual[field], Fraction(value))
            elif isinstance(value, (float, int)) and not isinstance(value, bool) and value != 0:
                self.assertIsNotNone(actual[field], field)
                self.assertTrue(
                    math.isclose(actual[field], value, rel_tol=comparison["relative_tolerance"],
                                 abs_tol=comparison["absolute_tolerance"]),
                    (field, actual[field], value),
                )
            else:
                self.assertEqual(actual[field], value, field)
        self.assertEqual(result.available, expected["contrast"] is not None)

    def test_predeclared_independent_measurements_and_rejections(self):
        for case in FIXTURES["cases"]:
            with self.subTest(case=case["id"]):
                catalog, state, windows = self.build(**case.get("input", {}))
                original_bytes = [segment.data.tobytes() for segment in catalog.segments]
                result = compute_rms_contrast(catalog, state, windows)
                self.assert_fixture(result, case["expected"])
                self.assertEqual([segment.data.tobytes() for segment in catalog.segments], original_bytes)

    def test_exact_grid_not_rounded_utc_controls_membership(self):
        catalog, state, windows = self.build(
            period_ns="5/2", segments=[{"id": "main", "samples": [2, 6, 40, 7, 9]}],
            event_ns=[0, 3], background_ns=[6, 11],
        )
        # Exact times are 0, 2.5, 5, 7.5 and 10 ns. Rounded ties at 2 and 8 ns
        # do not affect either request's membership on the exact grid.
        event = replace(windows.event, stop_ns=Fraction(5, 2))
        result = compute_rms_contrast(catalog, state, replace(windows, event=event))
        self.assertEqual(result.event.span, SampleSpan("main", 0, 1))
        self.assertEqual(result.event.reasons, ("too_short",))
        event = replace(windows.event, stop_ns=Fraction(11, 4))
        result = compute_rms_contrast(catalog, state, replace(windows, event=event))
        self.assertEqual(result.event.span, SampleSpan("main", 0, 2))
        self.assertEqual(result.background.span, SampleSpan("main", 3, 5))
        self.assertEqual(result.duration_ns, 5)
        self.assertEqual(result.contrast, 2)

    def test_full_request_must_fit_even_if_clipping_leaves_equal_counts(self):
        for event, background, invalid in (
            ((-1, 4_000_000), (4_000_000, 8_000_000), "event"),
            ((0, 4_000_000), (4_000_000, 8_000_001), "background"),
            ((0, 4_000_000), (8_000_000, 12_000_000), "background"),
        ):
            with self.subTest(event=event, background=background):
                catalog, state, windows = self.build(event_ns=event, background_ns=background)
                result = compute_rms_contrast(catalog, state, windows)
                invalid_window = getattr(result, invalid)
                self.assertEqual(invalid_window.reasons, ("out_of_bounds",))
                self.assertIsNone(invalid_window.span)
                self.assertIsNone(invalid_window.rms)
                self.assertIsNone(result.contrast)

    def test_disjoint_valid_windows_on_different_segments_are_unavailable(self):
        catalog, state, windows = self.build(segments=[
            {"id": "left", "samples": [2, 2, 6, 6]},
            {"id": "right", "start_ns": 4_000_000, "samples": [7, 7, 9, 9]},
        ])
        result = compute_rms_contrast(catalog, state, windows)
        self.assertEqual(result.reasons, ("cross_segment",))
        self.assertEqual(result.event.rms, 2)
        self.assertEqual(result.background.rms, 1)
        self.assertIsNone(result.contrast)

    def test_cross_segment_and_recording_bounds_are_independent(self):
        for event, expected in (
            ((-1, 8_000_000), ("cross_segment", "out_of_bounds")),
            ((0, 8_000_000), ("cross_segment",)),
            ((3_000_000, 5_000_000), ("cross_segment",)),
            ((4_000_000, 6_000_000), ("cross_segment",)),
            ((-1, 4_000_000), ("out_of_bounds",)),
        ):
            with self.subTest(event=event):
                catalog, state, windows = self.build(
                    event_ns=event, background_ns=[8_000_000, 12_000_000], segments=[
                        {"id": "left", "samples": [2, 2, 6, 6]},
                        {"id": "right", "start_ns": 6_000_000, "samples": [7, 9, 7, 7, 9, 9]},
                    ],
                )
                result = compute_rms_contrast(catalog, state, windows)
                self.assertEqual(result.event.reasons, expected)
                self.assertEqual(result.reasons, expected)
                self.assertIsNone(result.event.span)

    def test_oversized_window_reports_intersecting_mask_without_scanning_samples(self):
        count = 1_000_001
        edge = count * 1_000_000
        catalog, state, windows = self.build(
            dtype="float64", event_ns=[0, edge], background_ns=[edge, 2 * edge],
            masks=[["main", 5, 6]], segments=[{"id": "main", "samples": [-1, "NaN"], "repeat": count}],
        )
        result = compute_rms_contrast(catalog, state, windows)
        self.assertEqual(result.event.reasons, ("too_large", "masked_samples"))
        self.assertEqual(result.background.reasons, ("too_large",))
        self.assertEqual(result.reasons, ("too_large", "masked_samples"))
        self.assertIsNone(result.event.mean)
        self.assertIsNone(result.contrast)

    def test_overlap_detection_is_scoped_to_selected_source_and_channel(self):
        catalog, state, windows = self.build(segments=[
            FIXTURES["defaults"]["segments"][0],
            {"id": "another_source", "source": "other", "samples": [2, 2, 6, 6]},
            {"id": "another_channel", "channel": "XX.SYN..EHN", "samples": [2, 2, 6, 6]},
        ])
        self.assertEqual(compute_rms_contrast(catalog, state, windows).contrast, 2)
        wrong = replace(windows.background, channel="XX.SYN..EHN")
        result = compute_rms_contrast(catalog, state, replace(windows, background=wrong))
        self.assertEqual(result.reasons, ("wrong_source_or_channel",))
        self.assertIsNone(result.background.span)

    def test_reports_all_applicable_reasons_in_order_with_no_partial_stats(self):
        catalog, state, windows = self.build(
            dtype="float64", segments=[{"id": "main", "samples": [2, 2, "NaN", 6, 7, 7, 9, 9]}],
            masks=[["main", 2, 3]], background_ns=[3_000_000, 8_000_000],
        )
        result = compute_rms_contrast(catalog, state, windows)
        self.assertEqual(result.reasons, ("unequal_length", "overlapping_windows", "masked_samples", "nonfinite_samples"))
        self.assertEqual(result.event.reasons, ("masked_samples", "nonfinite_samples"))
        self.assertIsNone(result.event.rms)
        self.assertIsNone(result.event.mean)
        self.assertIsNone(result.contrast)

    def test_provenance_coverage_identity_and_verification_are_independent(self):
        catalog, state, windows = self.build(
            segments=[{"id": "main", "samples": [2, 2, 6, 6, 99, 7, 7, 9, 9]}],
            background_ns=[5_000_000, 9_000_000],
        )
        reference = self.evidence(windows.source_hash, windows.channel, 0, 9_000_000)
        wrong_trace = replace(reference, channel="XX.SYN..EHN")
        partial = replace(reference, stop_ns=5_000_000)
        for epoch, counts, expected_status, expected_units, qualified in (
            (None, reference, "unknown", "raw counts", True),
            (EpochEvidence(reference), partial, "unverified", "raw units unknown", True),
            (EpochEvidence(partial, "Independently checked"), wrong_trace, "unverified", "raw units unknown", True),
            (EpochEvidence(wrong_trace, "Independently checked"), None, "unknown", "raw units unknown", True),
            (EpochEvidence(reference, "Independently checked"), None,
             "independently_verified", "raw units unknown", False),
        ):
            with self.subTest(epoch=epoch, counts=counts):
                result = compute_rms_contrast(catalog, state, replace(windows, epoch_evidence=epoch, counts_evidence=counts))
                self.assertEqual(result.epoch_status, expected_status)
                self.assertEqual(result.units, expected_units)
                self.assertEqual(result.contrast, 2)
                self.assertEqual(result.qualified, qualified)

    def test_lossless_wide_integers_still_allow_measurement(self):
        middle = 2**54
        catalog, state, windows = self.build(dtype="uint64", segments=[
            {"id": "main", "samples": [middle - 8, middle - 8, middle + 8, middle + 8, 7, 7, 9, 9]},
        ])
        result = compute_rms_contrast(catalog, state, windows)
        self.assertEqual(result.event.mean, middle)
        self.assertEqual(result.event.rms, 8)
        self.assertEqual(result.contrast, 8)

    def test_mean_survives_overflow_and_extreme_cancellation(self):
        catalog, state, windows = self.build(
            dtype="float64", event_ns=[0, 6_000_000], background_ns=[6_000_000, 12_000_000],
            segments=[{"id": "main", "samples": [1e308, 1e308, -1e308, -1e308, 5e-300, 5e-300, -1, 1, -1, 1, -1, 1]}],
        )
        result = compute_rms_contrast(catalog, state, windows)
        self.assertTrue(math.isclose(result.event.mean, 1e-299 / 6, rel_tol=1e-12))
        self.assertTrue(math.isclose(result.event.rms, 1e308 * math.sqrt(2 / 3), rel_tol=1e-12))

    def test_mean_survives_an_unrepresentable_sum_at_a_power_of_two(self):
        value = math.ldexp(1, 1023)
        catalog, state, windows = self.build(dtype="float64", segments=[
            {"id": "main", "samples": [value, value, 0, 0, -1, 1, -1, 1]},
        ])
        result = compute_rms_contrast(catalog, state, windows)
        self.assertEqual(result.event.mean, value / 2)
        self.assertEqual(result.event.rms, value / 2)
        self.assertEqual(result.contrast, value / 2)

    def test_nonconstant_subnormal_variation_cannot_become_an_exact_zero(self):
        tiniest = math.ulp(0.0)
        catalog, state, windows = self.build(dtype="float64", segments=[
            {"id": "main", "samples": [0, 0, 0, tiniest, -1, 1, -1, 1]},
        ])
        result = compute_rms_contrast(catalog, state, windows)
        self.assertEqual(result.event.reasons, ("numeric_failure",))
        self.assertIsNone(result.contrast)

    def test_positive_contrast_underflow_does_not_report_exact_zero(self):
        catalog, state, windows = self.build(dtype="float64", segments=[
            {"id": "main", "samples": [-1e-300, -1e-300, 1e-300, 1e-300, -1e300, -1e300, 1e300, 1e300]},
        ])
        result = compute_rms_contrast(catalog, state, windows)
        self.assertEqual(result.reasons, ("numeric_failure",))
        self.assertIsNone(result.contrast)
        self.assertEqual(result.event.rms, 1e-300)
        self.assertEqual(result.background.rms, 1e300)

    def test_contrast_uses_unrounded_rms_for_subnormal_windows(self):
        tiniest = math.ulp(0.0)
        for scale, background_rms in ((1.0, 1.5), (tiniest, 2 * tiniest)):
            with self.subTest(scale=scale):
                catalog, state, windows = self.build(dtype="float64", segments=[
                    {"id": "main", "samples": [multiplier * scale for multiplier in (-2, 2, -2, 2, 0, 3, 0, 3)]},
                ])
                result = compute_rms_contrast(catalog, state, windows)
                self.assertEqual(result.event.rms, 2 * scale)
                self.assertEqual(result.background.rms, background_rms)
                self.assertEqual(result.contrast, 4 / 3)

    def test_contrast_accepts_largest_and_smallest_positive_float64_values(self):
        largest = float.fromhex("0x1.fffffffffffffp+1023")
        smallest = math.ulp(0.0)
        for event, background, contrast in (
            (largest, 1.0, largest),
            (math.ldexp(1, -51), math.ldexp(1, 1023), smallest),
        ):
            with self.subTest(event=event, background=background):
                catalog, state, windows = self.build(dtype="float64", segments=[
                    {"id": "main", "samples": [-event, event, -event, event, -background, background, -background, background]},
                ])
                result = compute_rms_contrast(catalog, state, windows)
                self.assertEqual(result.reasons, ())
                self.assertEqual(result.contrast, contrast)


class AppliedRmsTests(unittest.TestCase):
    build = RmsTests.build
    evidence = staticmethod(RmsTests.evidence)
    def single(self):
        from seisslate.review import ReviewHistory

        catalog, _, windows = self.build()
        state = ReviewState.for_single_channel(windows.source_hash, windows.channel)
        return catalog, ReviewHistory(state), windows

    def test_applied_decision_and_mask_invalidation_are_separate_revisions(self):
        from seisslate.review import mask_spans, set_rms_review
        from seisslate.rms import RmsReview

        catalog, history, windows = self.single()
        state = history.current.state
        review = RmsReview(windows, "accepted", "Analyst", "Windows chosen")
        history.preview_rms_review(set_rms_review(state, catalog, review), "Review")
        self.assertIsNone(history.current.state.rms.baseline)
        applied = history.apply_rms(catalog)
        self.assertEqual(applied.state.rms.disposition, "accepted")
        self.assertTrue(history.replay_rms(catalog).ready)

        span = applied.state.rms.baseline.event.span
        masked = mask_spans(applied.state, (span,), catalog=catalog, reason="Transient")
        history.preview(masked, "Mask")
        with self.assertRaisesRegex(ValueError, "apply_rms"):
            history.apply()
        changed = history.apply_rms(catalog)
        self.assertEqual(changed.state.rms.disposition, "candidate")
        self.assertIn("masked_samples", changed.state.rms.baseline.reasons)
        self.assertFalse(history.replay_rms(catalog).ready)
        self.assertEqual(history.undo(), applied)
        self.assertTrue(history.replay_rms(catalog).ready)

    def test_single_channel_identity_cannot_change_or_have_paired_state(self):
        from seisslate.review import Receiver

        _, history, windows = self.single()
        state = history.current.state
        with self.assertRaisesRegex(ValueError, "source or mode"):
            history.preview(ReviewState.for_single_channel(windows.source_hash, "another"), "Switch channel")
        with self.assertRaisesRegex(ValueError, "paired"):
            replace(state, receivers=(Receiver("original", windows.channel, windows.source_hash),))
        self.assertEqual(state.single_channel_identity, (windows.source_hash, windows.channel))

    def test_changing_windows_or_epoch_clears_prior_review(self):
        from seisslate.review import set_rms_review, set_rms_windows
        from seisslate.rms import RmsReview

        catalog, history, windows = self.single()
        state = history.current.state
        review = set_rms_review(state, catalog, RmsReview(windows, "accepted", "A", "Checked"))
        history.preview_rms_review(review, "Apply")
        applied = history.apply_rms(catalog).state
        for changed in (
            replace(windows, background=None),
            replace(windows, epoch_evidence=EpochEvidence(
                self.evidence(windows.source_hash, windows.channel, 0, 8_000_000), verification="Checked",
            )),
        ):
            with self.subTest(changed=changed):
                preview = set_rms_windows(applied, catalog, changed)
                self.assertEqual(preview.rms.disposition, "candidate")
                self.assertIsNone(preview.rms.baseline)
                self.assertEqual(history.current.state, applied)


    def test_generic_preview_cannot_carry_acceptance_across_changed_rms_evidence(self):
        from seisslate.review import set_rms_review
        from seisslate.rms import RmsReview

        catalog, history, windows = self.single()
        # Unchanged windows can receive a new disposition after a candidate selection.
        candidate = set_rms_review(history.current.state, catalog, RmsReview(windows))
        history.preview(candidate, "Select candidate windows")
        history.apply_rms(catalog)
        reviewed = RmsReview(windows, "accepted", "Analyst", "Reviewed the windows")
        history.preview(set_rms_review(history.current.state, catalog, reviewed), "Accept")
        accepted = history.apply_rms(catalog)

        reference = self.evidence(windows.source_hash, windows.channel, 0, 8_000_000)
        for changed in (
            replace(windows, background=None),
            replace(windows, epoch_evidence=EpochEvidence(reference)),
            replace(windows, counts_evidence=reference),
        ):
            with self.subTest(changed=changed):
                carried = replace(accepted.state, rms=replace(reviewed, windows=changed))
                with self.assertRaisesRegex(ValueError, "explicit RMS review"):
                    history.preview(carried, "Change previously reviewed support")
                self.assertIsNone(history.draft)
                self.assertIs(history.current, accepted)

        reviewed_again = RmsReview(replace(windows, background=None), "accepted", "Analyst", "Reassessed")
        changed = set_rms_review(accepted.state, catalog, reviewed_again)
        history.preview_rms_review(changed, reviewed_again.reason)
        applied = history.apply_rms(catalog)
        self.assertEqual(applied.state.rms.disposition, "accepted")
        self.assertEqual(applied.state.rms.baseline.reasons, ("missing_window",))
        self.assertFalse(history.replay_rms(catalog).ready)


if __name__ == "__main__":
    unittest.main()
