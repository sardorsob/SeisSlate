from copy import copy
from dataclasses import FrozenInstanceError, replace
import unittest

import numpy as np

from seisslate.sources import SampleRef, SampleSpan, Segment, SourceCatalog


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


class RawEvidenceTests(unittest.TestCase):
    def setUp(self):
        from seisslate.evidence import PickDecision, RawEvidenceView
        from seisslate.example import load_example

        self.catalog, self.history = load_example()
        self.initial = self.history.current.state
        pick = self.initial.blow("B01").original.reviewed_ref
        self.segment = self.catalog.segment(pick)
        self.span = SampleSpan(self.segment.segment_id, 355, 366)
        self.bounds = self.segment.ref(359), self.segment.ref(361)
        self.decision = PickDecision(
            "B01", "original", pick, "accepted", *self.bounds, reason="Visible onset", reviewer="Analyst",
            inspection_span=self.span,
        )
        self.view = RawEvidenceView(
            self.segment.source_hash, self.segment.channel, self.span,
            self.segment.time_ns(355), self.segment.time_ns(366), pick, self.bounds,
        )
        self.versions = (("SeisSlate", "test"), ("NumPy", np.__version__))
        self.capture_time = 1_800_000_000_123_456_789

    def stage(self):
        from seisslate.review import set_pick_decision

        state = set_pick_decision(self.history.current.state, self.catalog, self.decision)
        self.history.preview(state, "Review exact samples")
        return self.history.confirm_current_pick_evidence(self.catalog, self.decision, self.view)

    def seal(self, confirmation, *, catalog=None, view=None):
        return self.history.apply_with_evidence(
            self.catalog if catalog is None else catalog, confirmation, self.view if view is None else view,
            captured_utc_ns=self.capture_time, software_versions=self.versions,
        )

    def modified_catalog(self, *, changed_bytes=False, changed_grid=False, changed_source=False):
        source = next(source for source in self.catalog.sources if source.sha256 == self.segment.source_hash)
        samples = self.segment.data.copy()
        if changed_bytes:
            samples[355] += 1
        source_hash = "0" * 64 if changed_source else source.sha256
        segment = Segment(
            source_hash, self.segment.channel, self.segment.segment_id,
            self.segment.start_ns + int(changed_grid), self.segment.sampling_rate_hz, samples,
        )
        updated_source = replace(source, sha256=source_hash, segments=(segment,))
        other_sources = tuple(item for item in self.catalog.sources if source.sha256 != item.sha256)
        return SourceCatalog((updated_source, *other_sources))

    def test_capture_records_one_linked_revision_and_restores_it_with_undo_redo(self):
        confirmation = self.stage()
        applied = self.seal(confirmation)
        (decision,) = applied.state.pick_decisions
        (record,) = applied.state.raw_evidence
        self.assertEqual(decision.evidence_id, record.evidence_id)
        self.assertEqual(record.applied_revision_id, applied.revision_id)
        self.assertEqual(record.confirmation, confirmation)
        self.assertEqual(record.confirmation.decision.reviewer, "Analyst")
        self.assertEqual(record.confirmation.samples.sample_count, 11)
        self.assertEqual(record.confirmation.samples.period_numerator, 1_000_000)
        self.assertEqual(record.confirmation.samples.period_denominator, 1)
        self.assertEqual(record.captured_utc_ns, self.capture_time)
        self.assertEqual(record.software_versions, self.versions)
        self.assertEqual(applied.reason, "Review exact samples")
        self.assertIsNone(self.history.draft)
        self.assertEqual(self.history.undo().state.raw_evidence, ())
        self.assertEqual(self.history.redo(), applied)

    def test_capture_requires_software_version_pairs_and_exact_recorded_receiver(self):
        confirmation = self.stage()
        proper_versions = self.versions
        draft = self.history.draft
        self.versions = ("ab",)
        with self.assertRaises(ValueError):
            self.seal(confirmation)
        self.assertIs(self.history.draft, draft)
        self.versions = proper_versions
        (record,) = self.seal(confirmation).state.raw_evidence
        with self.assertRaises(ValueError):
            replace(confirmation, view=replace(self.view, channel="XX.OTHER..EHZ"))
        with self.assertRaises(ValueError):
            replace(record, software_versions=(("numpy", "2.0"), ("numpy", "2.1")))

    def test_failed_recapture_keeps_the_same_draft_and_revision_id(self):
        confirmation = self.stage()
        draft = self.history.draft
        for catalog, view in (
            (self.catalog, replace(self.view, display_stop_ns=self.view.display_stop_ns + 1)),
            (self.modified_catalog(changed_bytes=True), self.view),
        ):
            with self.subTest(catalog=catalog, view=view), self.assertRaises(ValueError):
                self.seal(confirmation, catalog=catalog, view=view)
            self.assertIs(self.history.draft, draft)
            self.assertEqual(self.history.current.state, self.initial)
            self.assertEqual(self.history.next_revision_id, 1)
        with self.assertRaises(ValueError):
            self.history.apply()
        self.assertIs(self.history.draft, draft)
        self.assertEqual(self.seal(confirmation).revision_id, 1)

    def test_confirmation_cannot_outlive_its_draft_or_move_to_another_history(self):
        from seisslate.review import ReviewHistory

        confirmation = self.stage()
        draft = self.history.draft
        original = self.history
        self.history = copy(original)
        with self.assertRaises(ValueError):
            self.seal(confirmation)
        self.history = original

        self.history.cancel()
        self.history.preview(draft.state, draft.reason)
        with self.assertRaises(ValueError):
            self.seal(confirmation)
        self.assertEqual(self.history.current.state, self.initial)

        confirmation = self.stage()
        draft = self.history.draft
        self.history.preview(draft.state, draft.reason)
        with self.assertRaises(ValueError):
            self.seal(confirmation)

        confirmation = self.stage()
        draft = self.history.draft
        other = ReviewHistory(self.initial)
        other.preview(draft.state, draft.reason)
        self.history = other
        with self.assertRaises(ValueError):
            self.seal(confirmation)
        self.assertEqual(self.history.current.state, self.initial)

    def test_replay_reports_raw_grid_source_method_and_view_checks_separately(self):
        from seisslate.evidence import inspect_raw_evidence

        missing = self.history.replay_pick_evidence(self.catalog, "B01", "original")
        self.assertEqual(missing.raw_samples, "not_recorded")
        self.assertFalse(missing.attestation_current)
        (record,) = self.seal(self.stage()).state.raw_evidence
        current = inspect_raw_evidence(self.catalog, record, view=self.view)
        self.assertTrue(current.attestation_current)
        self.assertEqual(current.source_identity, "source_match")
        self.assertEqual(current.raw_samples, "raw_samples_match")
        self.assertTrue(current.grid_matches)
        self.assertTrue(current.view_restored)

        modified = inspect_raw_evidence(self.modified_catalog(changed_bytes=True), record, view=self.view)
        self.assertEqual(modified.source_identity, "source_match")
        self.assertEqual(modified.raw_samples, "raw_samples_differ")
        self.assertTrue(modified.grid_matches)
        self.assertFalse(modified.attestation_current)

        new_grid = inspect_raw_evidence(self.modified_catalog(changed_grid=True), record, view=self.view)
        self.assertFalse(new_grid.grid_matches)
        self.assertEqual(new_grid.raw_samples, "raw_samples_match")
        self.assertFalse(new_grid.attestation_current)

        new_source = inspect_raw_evidence(self.modified_catalog(changed_source=True), record)
        self.assertEqual(new_source.source_identity, "source_differ")
        self.assertEqual(new_source.raw_samples, "not_checked")

        unsupported = inspect_raw_evidence(self.catalog, replace(record, method_id="raw-v99"), view=self.view)
        self.assertFalse(unsupported.method_supported)
        self.assertEqual(unsupported.raw_samples, "unsupported_method")
        self.assertIsNone(unsupported.view_restored)
        self.assertFalse(unsupported.attestation_current)

        changed_view = inspect_raw_evidence(self.catalog, record, view=replace(
            self.view, display_stop_ns=self.view.display_stop_ns + 1,
        ))
        self.assertFalse(changed_view.view_restored)
        self.assertEqual(changed_view.raw_samples, "raw_samples_match")
        self.assertFalse(changed_view.attestation_current)

    def test_raw_digest_match_does_not_validate_impossible_recorded_views(self):
        from seisslate.evidence import inspect_raw_evidence

        (record,) = self.seal(self.stage()).state.raw_evidence
        for view in (
            replace(self.view, display_start_ns=self.view.display_start_ns + 1),
            replace(self.view, display_stop_ns=self.view.display_stop_ns + 1),
            replace(self.view, ordinate_scale="normalized"),
        ):
            changed = replace(record, confirmation=replace(record.confirmation, view=view))
            with self.subTest(view=view):
                replay = inspect_raw_evidence(self.catalog, changed, view=view)
                self.assertEqual(replay.raw_samples, "raw_samples_match")
                self.assertTrue(replay.grid_matches)
                self.assertFalse(replay.recorded_view_valid)
                self.assertFalse(replay.view_restored)
                self.assertFalse(replay.attestation_current)
                self.assertFalse(inspect_raw_evidence(self.catalog, changed).attestation_current)

    def test_masks_stale_attestation_only_when_a_changed_span_overlaps_support(self):
        from seisslate.review import mask_spans, restore_spans, set_pick_decision

        old_confirmation = self.stage()
        applied = self.seal(old_confirmation)
        outside = (SampleSpan(self.span.segment_id, 366, 370),)
        unrelated = mask_spans(applied.state, outside, catalog=self.catalog, reason="Separate artefact")
        self.history.preview(unrelated, "Mask after evidence")
        clean = self.history.apply()
        self.assertTrue(self.history.replay_pick_evidence(self.catalog, "B01", "original").attestation_current)

        touching = (SampleSpan(self.span.segment_id, 355, 356),)
        masked = mask_spans(clean.state, touching, catalog=self.catalog, reason="Suspected clipping")
        self.history.preview(masked, "Mask within evidence")
        self.history.apply()
        stale = self.history.replay_pick_evidence(self.catalog, "B01", "original")
        self.assertTrue(stale.masks_changed)
        self.assertEqual(stale.raw_samples, "raw_samples_match")
        self.assertFalse(stale.attestation_current)
        restored = restore_spans(masked, touching, catalog=self.catalog, reason="Samples reinspected")
        self.history.preview(restored, "Restore within evidence")
        self.history.apply()
        self.assertEqual(self.history.replay_pick_evidence(self.catalog, "B01", "original"), stale)
        self.history.undo()
        self.history.undo()
        self.assertTrue(self.history.replay_pick_evidence(self.catalog, "B01", "original").attestation_current)
        self.history.redo()
        self.assertEqual(self.history.replay_pick_evidence(self.catalog, "B01", "original"), stale)
        self.history.redo()
        reconsidered = set_pick_decision(self.history.current.state, self.catalog, self.decision)
        self.history.preview(reconsidered, "Reconsider review")
        with self.assertRaises(ValueError):
            self.seal(old_confirmation)
        self.assertEqual(self.history.current.state, restored)
        self.assertEqual(self.history.replay_pick_evidence(self.catalog, "B01", "original"), stale)
        rechecked = self.seal(self.stage())
        self.assertEqual(len(rechecked.state.raw_evidence), 2)
        self.assertNotEqual(rechecked.state.raw_evidence[1].evidence_id, applied.state.raw_evidence[0].evidence_id)
        self.assertTrue(self.history.replay_pick_evidence(self.catalog, "B01", "original").attestation_current)

    def test_applied_history_rejects_historical_replacement_and_discarded_branch_links(self):
        from seisslate.review import ReviewHistory, move_pick

        applied = self.seal(self.stage())
        (record,) = applied.state.raw_evidence
        tampered = replace(applied.state, raw_evidence=(replace(record, captured_utc_ns=0),))
        revision = replace(applied, revision_id=12, state=tampered)
        with self.assertRaises(ValueError):
            ReviewHistory(self.initial, revisions=(*self.history.revisions, revision))
        self.history.undo()
        changed = move_pick(self.initial, self.catalog, "B01", "original", self.segment.ref(361))
        self.history.preview(changed, "Create a new branch")
        branch = self.history.apply()
        self.assertGreater(branch.revision_id, applied.revision_id)
        self.assertEqual(branch.state.raw_evidence, ())
        self.assertFalse(self.history.can_redo)
        self.assertEqual(self.history.replay_pick_evidence(self.catalog, "B01", "original").raw_samples, "not_recorded")

    def test_gap_decision_applies_without_evidence_and_inspected_missing_point_can_be_captured(self):
        from seisslate.evidence import PickDecision
        from seisslate.review import set_pick_decision

        gap = PickDecision(
            "B07", "original", None, "unresolved", intended_utc_ns=self.segment.stop_ns,
            reason="Receiver gap", reviewer="Analyst",
        )
        pending = set_pick_decision(self.initial, self.catalog, gap)
        self.history.preview(pending, "Record missing arrival")
        self.assertEqual(self.history.apply().state.pick_decisions, (gap,))
        self.assertEqual(self.history.replay_pick_evidence(self.catalog, "B07", "original").raw_samples, "not_recorded")

        self.decision = replace(
            gap, intended_utc_ns=self.segment.time_ns(360), reason="No distinct onset", inspection_span=self.span,
        )
        self.view = replace(self.view, point=None, bounds=None)
        (record,) = self.seal(self.stage()).state.raw_evidence
        self.assertIsNone(record.confirmation.decision.reviewed_ref)
        self.assertEqual(record.confirmation.decision.intended_utc_ns, self.segment.time_ns(360))


class RawDigestTests(unittest.TestCase):
    def test_raw_v1_frame_has_a_fixed_digest_across_endianness(self):
        from seisslate.evidence import raw_digest

        expected = "9ced5c38532f3c3669954ce0756e0ec1a9f4c7783595d9cdb5d9118823d36d47"
        for dtype in ("<i2", ">i2"):
            with self.subTest(dtype=dtype):
                self.assertEqual(raw_digest(np.array([-2, 0, 513], dtype=dtype)), expected)
        self.assertNotEqual(raw_digest(np.array([-2, 0, 513], dtype="<i4")), expected)


if __name__ == "__main__":
    unittest.main()
