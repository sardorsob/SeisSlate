from dataclasses import FrozenInstanceError, replace
import unittest

from seisslate.sources import SampleRef, SampleSpan


POINT = SampleRef("fingerprint", "XX.TEST..EHZ", "segment", 8)


class PickDecisionTests(unittest.TestCase):
    def test_inclusive_bounds_allow_an_explicit_zero_width_bracket(self):
        from seisslate.evidence import PickDecision

        decision = PickDecision(
            "B1", "original", POINT, "accepted", lower_ref=POINT, upper_ref=POINT,
            reason="Onset is visible", reviewer="Analyst",
        )
        self.assertEqual((decision.lower_ref, decision.upper_ref), (POINT, POINT))
        with self.assertRaises(FrozenInstanceError):
            setattr(decision, "reason", "Later judgment")

    def test_bounds_require_both_endpoints_containing_the_point_on_its_grid(self):
        from seisslate.evidence import PickDecision

        lower = replace(POINT, sample_index=7)
        upper = replace(POINT, sample_index=9)
        for pair in (
            (POINT, None), (None, POINT), (upper, lower), (upper, upper), (lower, lower),
            (replace(lower, source_hash="other"), upper),
            (lower, replace(upper, channel="XX.OTHER..EHZ")),
            (lower, replace(upper, segment_id="later")),
        ):
            with self.subTest(bounds=pair), self.assertRaises(ValueError):
                PickDecision("B1", "original", POINT, "candidate", lower_ref=pair[0], upper_ref=pair[1])

    def test_endpoint_at_inspection_stop_is_outside_the_view(self):
        from seisslate.evidence import PickDecision

        with self.assertRaises(ValueError):
            PickDecision(
                "B1", "original", POINT, "candidate", lower_ref=POINT, upper_ref=replace(POINT, sample_index=9),
                inspection_span=SampleSpan(POINT.segment_id, 7, 9),
            )

    def test_no_point_requires_an_unresolved_decision_with_time_and_reason(self):
        from seisslate.evidence import PickDecision

        for fields in (
            {"status": "candidate", "intended_utc_ns": 15},
            {"status": "unresolved"},
            {"status": "unresolved", "intended_utc_ns": True},
            {"status": "unresolved", "intended_utc_ns": 15, "reason": " \t "},
            {"status": "unresolved", "intended_utc_ns": 15, "lower_ref": POINT, "upper_ref": POINT},
        ):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                PickDecision("B1", "original", None, **({"reason": "No arrival", **fields}))

    def test_point_acceptance_and_unresolved_decisions_require_human_reasons(self):
        from seisslate.evidence import PickDecision

        for status in ("accepted", "unresolved"):
            with self.subTest(status=status), self.assertRaises(ValueError):
                PickDecision("B1", "original", POINT, status, reviewer="Analyst", reason=" \t ")


if __name__ == "__main__":
    unittest.main()
