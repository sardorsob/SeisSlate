from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
from obspy import read

from seisslate.sources import SampleSpan, Segment, Source, SourceCatalog, load_source


FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "synthetic"
EPOCH = 946_684_800_000_000_000
MS = 1_000_000


class ReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.paths = tuple(FIXTURE / f"{name}.mseed" for name in ("original", "calibration"))
        cls.catalog = SourceCatalog(tuple(load_source(path) for path in cls.paths))
        cls.original, cls.calibration = (source.segments[0] for source in cls.catalog.sources)

    def setUp(self):
        from seisslate import review

        self.api = review
        self.state = review.ReviewState(
            (
                self.blow("B01", 360, 400, 13.1, 31.1),
                self.blow("B02", 815, 800, 26.85, 17.35),
                self.blow("B07", None, 2145, 28.1, 16.1),
            ),
            receivers=(
                review.Receiver("original", self.original.channel, self.original.source_hash),
                review.Receiver("calibration", self.calibration.channel, self.calibration.source_hash),
            ),
            timing_evidence="Synthetic common clock",
        )

    def blow(self, blow_id, original_ms, calibration_ms, original_m, calibration_m):
        original_ns = None if original_ms is None else EPOCH + original_ms * MS
        return self.api.Blow(
            blow_id,
            self.api.import_pick(self.catalog, self.original.channel, original_ns),
            self.api.import_pick(self.catalog, self.calibration.channel, EPOCH + calibration_ms * MS),
            original_m,
            calibration_m,
        )

    def test_off_grid_import_requires_explicit_candidate_and_keeps_reported_utc(self):
        requested = EPOCH + 360 * MS + 400_000
        unresolved = self.api.import_pick(self.catalog, self.original.channel, requested)
        self.assertEqual(unresolved.imported_ns, requested)
        self.assertIsNone(unresolved.original_ref)
        self.assertIsNone(unresolved.reviewed_ref)

        chosen = self.original.ref(361)
        resolved = self.api.import_pick(self.catalog, self.original.channel, requested, chosen)
        self.assertEqual(resolved.imported_ns, requested)
        self.assertEqual(resolved.original_ref, chosen)
        self.assertEqual(resolved.reviewed_ref, chosen)
        with self.assertRaises(ValueError):
            self.api.import_pick(self.catalog, self.original.channel, requested, self.original.ref(400))

    def test_nudge_draft_changes_one_onset_and_cancel_preserves_applied_revision(self):
        source_bytes = [path.read_bytes() for path in self.paths]
        history = self.api.ReviewHistory(self.state)
        draft = self.api.nudge_pick(history.current.state, self.catalog, "B01", "original", 1)
        history.preview(draft, "Nudge first arrival")
        self.assertEqual(history.current.state, self.state)
        picked, second, missing = history.draft.state.blows
        self.assertEqual(picked.original.imported_ns, EPOCH + 360 * MS)
        self.assertEqual(picked.original.original_ref, self.original.ref(360))
        self.assertEqual(picked.original.reviewed_ref, self.original.ref(361))
        self.assertEqual(picked.calibration, self.state.blows[0].calibration)
        self.assertEqual((second, missing), self.state.blows[1:])
        history.cancel()
        self.assertIsNone(history.draft)
        self.assertEqual(history.current.state, self.state)
        self.assertEqual([path.read_bytes() for path in self.paths], source_bytes)

    def test_mask_union_and_partial_restore_keep_half_open_sample_boundaries(self):
        segment_id = self.original.segment_id
        first = SampleSpan(segment_id, 350, 361)
        second = SampleSpan(segment_id, 355, 365)
        masked = self.api.mask_spans(self.state, (first, second))
        self.assertEqual(masked.masks, (SampleSpan(segment_id, 350, 365),))
        self.assertTrue(masked.is_masked(self.original.ref(360)))
        self.assertFalse(masked.is_masked(self.original.ref(365)))
        self.assertFalse(masked.is_masked(self.calibration.ref(360)))
        restored = self.api.restore_spans(masked, (SampleSpan(segment_id, 359, 362),))
        self.assertEqual(restored.masks, (SampleSpan(segment_id, 350, 359), SampleSpan(segment_id, 362, 365)))
        self.assertFalse(restored.is_masked(self.original.ref(360)))
        self.assertTrue(restored.is_masked(self.original.ref(362)))
        self.assertEqual(restored.blows, self.state.blows)

    def test_reasoned_interval_actions_record_only_samples_that_changed(self):
        segment_id = self.original.segment_id
        legacy = self.api.mask_spans(self.state, (SampleSpan(segment_id, 353, 355),))
        selection = (SampleSpan(segment_id, 351, 358),)
        masked = self.api.mask_spans(legacy, selection, catalog=self.catalog, reason="Suspected clipping")
        self.assertEqual(masked.masks, (SampleSpan(segment_id, 351, 358),))
        (mask,) = masked.interval_actions
        self.assertEqual((mask.action, mask.reason, mask.spans), (
            "mask", "Suspected clipping", (SampleSpan(segment_id, 351, 353), SampleSpan(segment_id, 355, 358)),
        ))
        self.assertEqual(self.api.mask_spans(masked, selection, catalog=self.catalog, reason="Again"), masked)

        restored = self.api.restore_spans(
            masked, (SampleSpan(segment_id, 356, 361),), catalog=self.catalog, reason="Review cleared tail",
        )
        self.assertEqual(restored.masks, (SampleSpan(segment_id, 351, 356),))
        previous, restore = restored.interval_actions
        self.assertEqual(previous, mask)
        self.assertEqual((restore.action, restore.reason, restore.spans), (
            "restore", "Review cleared tail", (SampleSpan(segment_id, 356, 358),),
        ))
        self.assertEqual(self.api.restore_spans(
            restored, (SampleSpan(segment_id, 356, 361),), catalog=self.catalog, reason="Again",
        ), restored)

    def test_reasoned_interval_commands_reject_blanks_and_invalid_catalog_spans(self):
        valid = SampleSpan(self.original.segment_id, 350, 360)
        for action in (self.api.mask_spans, self.api.restore_spans):
            for span, reason in (
                (valid, " \t "),
                (SampleSpan("absent", 350, 360), "Note"),
                (SampleSpan(self.original.segment_id, 2149, 2151), "Note"),
            ):
                with self.subTest(action=action.__name__, span=span, reason=reason), self.assertRaises(ValueError):
                    action(self.state, (span,), catalog=self.catalog, reason=reason)

    def test_pick_move_preserves_same_point_decision_and_replaces_moved_decision_atomically(self):
        from seisslate.evidence import PickDecision

        point = self.original.ref(360)
        decision = PickDecision(
            "B01", "original", point, "accepted", lower_ref=self.original.ref(359), upper_ref=point,
            reason="Sharp onset", reviewer="Analyst",
        )
        accepted = self.api.set_pick_decision(self.state, self.catalog, decision)
        self.assertEqual(self.api.move_pick(accepted, self.catalog, "B01", "original", point), accepted)

        next_point = self.original.ref(361)
        nudged = self.api.nudge_pick(accepted, self.catalog, "B01", "original", 1)
        self.assertEqual(nudged.blows[0].original.reviewed_ref, next_point)
        self.assertEqual(nudged.pick_decisions, ())
        self.assertEqual(nudged.blows[0].original.original_ref, point)

        replacement = replace(decision, reviewed_ref=next_point, lower_ref=next_point, upper_ref=next_point)
        replaced = self.api.move_pick(accepted, self.catalog, "B01", "original", next_point, decision=replacement)
        self.assertEqual(replaced.blows, nudged.blows)
        self.assertEqual(replaced.pick_decisions, (replacement,))
        with self.assertRaises(ValueError):
            self.api.move_pick(accepted, self.catalog, "B01", "original", next_point, decision=decision)
        cleared = self.api.move_pick(replaced, self.catalog, "B01", "original", None)
        self.assertIsNone(cleared.blows[0].original.reviewed_ref)
        self.assertEqual(cleared.pick_decisions, ())
        self.assertEqual(cleared.blows[0].original.original_ref, point)

    def test_decision_targets_and_catalog_references_are_validated(self):
        from seisslate.evidence import PickDecision

        point = self.original.ref(360)
        decision = PickDecision("B01", "original", point, "candidate")
        accepted = self.api.set_pick_decision(self.state, self.catalog, decision)
        with self.assertRaises(ValueError):
            replace(accepted, pick_decisions=(decision, decision))
        for bad in (
            replace(decision, blow_id="missing"),
            replace(decision, role="calibration"),
            replace(decision, reviewed_ref=self.original.ref(359)),
            replace(decision, lower_ref=point, upper_ref=replace(point, sample_index=999_999)),
        ):
            with self.subTest(decision=bad), self.assertRaises(ValueError):
                self.api.set_pick_decision(self.state, self.catalog, bad)
        with self.assertRaises(ValueError):
            replace(decision, role="receiver")

    def test_missing_unresolved_intent_can_record_an_available_inspection_span(self):
        from seisslate.evidence import PickDecision

        decision = PickDecision(
            "B07", "original", None, "unresolved", intended_utc_ns=EPOCH + 2145 * MS,
            reason="No distinct onset", reviewer="Analyst",
        )
        unresolved = self.api.set_pick_decision(self.state, self.catalog, decision)
        self.assertEqual(unresolved.pick_decisions, (decision,))
        self.assertEqual(unresolved.blows, self.state.blows)

        inspected = replace(decision, inspection_span=SampleSpan(self.original.segment_id, 2140, 2150))
        inspected_state = self.api.set_pick_decision(unresolved, self.catalog, inspected)
        self.assertEqual(inspected_state.pick_decisions, (inspected,))
        with self.assertRaises(ValueError):
            self.api.set_pick_decision(unresolved, self.catalog, replace(
                inspected, inspection_span=SampleSpan(self.calibration.segment_id, 2140, 2150),
            ))
        with self.assertRaisesRegex(ValueError, "intended UTC"):
            self.api.set_pick_decision(unresolved, self.catalog, replace(
                inspected, inspection_span=SampleSpan(self.original.segment_id, 10, 20),
            ))

    def test_gap_intent_retains_the_role_and_utc_without_a_fabricated_sample(self):
        from seisslate.evidence import PickDecision

        intended = self.original.stop_ns
        self.assertEqual(self.catalog.lookup(self.original.channel, intended).status, "gap")
        decision = PickDecision(
            "B07", "original", None, "unresolved", intended_utc_ns=intended,
            reason="Receiver gap", reviewer="Analyst",
        )
        state = self.api.set_pick_decision(self.state, self.catalog, decision)
        (record,) = state.pick_decisions
        self.assertEqual((record.role, record.intended_utc_ns, record.reason), ("original", intended, "Receiver gap"))
        self.assertIsNone(record.reviewed_ref)
        self.assertIsNone(record.inspection_span)
        self.assertIsNone(record.evidence_id)
        with self.assertRaisesRegex(ValueError, "intended UTC"):
            self.api.set_pick_decision(state, self.catalog, replace(
                decision, inspection_span=SampleSpan(self.original.segment_id, 0, 10),
            ))

    def test_no_point_inspection_uses_unrounded_half_open_time_support(self):
        from seisslate.evidence import PickDecision

        segment = Segment("fractional", "XX.ORIG..EHZ", "fractional:0", 0, 400_000_000, np.arange(6))
        catalog = SourceCatalog((Source(Path("fractional.mseed"), "fractional", 0, (segment,)),))
        state = self.api.ReviewState(
            (self.api.Blow("missing", self.api.Pick(None), self.api.Pick(None), None, None),),
            receivers=(self.api.Receiver("original", segment.channel, segment.source_hash),),
        )
        # At a 2.5 ns period, sample 1 displays at 2 ns. Its exact boundary is 2.5 ns.
        for start, stop, intended, inside in (
            (1, 4, 2, False), (1, 4, 3, True), (1, 4, 9, True), (1, 4, 10, False), (0, 1, 2, True),
        ):
            with self.subTest(start=start, stop=stop, intended=intended):
                decision = PickDecision(
                    "missing", "original", None, "unresolved", intended_utc_ns=intended,
                    reason="No distinct onset", inspection_span=SampleSpan(segment.segment_id, start, stop),
                )
                if inside:
                    reviewed = self.api.set_pick_decision(state, catalog, decision)
                    self.assertEqual(reviewed.pick_decisions, (decision,))
                else:
                    with self.assertRaisesRegex(ValueError, "intended UTC"):
                        self.api.set_pick_decision(state, catalog, decision)

    def test_interval_history_rejects_actions_that_claim_a_no_op_changed_samples(self):
        from seisslate.evidence import IntervalAction

        span = SampleSpan(self.original.segment_id, 350, 355)
        state = self.api.mask_spans(self.state, (span,))
        forged = replace(state, interval_actions=(IntervalAction("mask", (span,), "Already masked"),))
        history = self.api.ReviewHistory(state)
        with self.assertRaises(ValueError):
            history.preview(forged, "Mask again")
        self.assertIsNone(history.draft)
        self.assertEqual(history.current.state, state)

    def test_decision_and_interval_actions_follow_the_applied_history(self):
        from seisslate.evidence import PickDecision

        history = self.api.ReviewHistory(self.state)
        point = self.original.ref(360)
        decision = PickDecision("B01", "original", point, "candidate", reason="Visible onset", reviewer="Analyst")
        staged = self.api.set_pick_decision(self.state, self.catalog, decision)
        span = (SampleSpan(self.original.segment_id, 350, 352),)
        staged = self.api.mask_spans(staged, span, catalog=self.catalog, reason="Spurious precursor")
        preview = history.preview(staged, "Waveform review")
        self.assertEqual(history.current.state, self.state)
        history.cancel()
        self.assertEqual(history.apply().state, self.state)

        history.preview(staged, "Waveform review")
        applied = history.apply()
        self.assertEqual(applied, preview)
        restaged = self.api.restore_spans(staged, span, catalog=self.catalog, reason="Precursor verified")
        history.preview(restaged, "Restore interval")
        restored = history.apply()
        self.assertEqual(history.undo(), applied)
        self.assertEqual(history.redo(), restored)
        self.assertEqual(history.undo(), applied)
        branch = self.api.move_pick(staged, self.catalog, "B01", "original", None)
        history.preview(branch, "Clear arrival")
        branched = history.apply()
        self.assertGreater(branched.revision_id, restored.revision_id)
        self.assertEqual(branched.state.interval_actions, applied.state.interval_actions)
        self.assertEqual(branched.state.pick_decisions, ())
        self.assertFalse(history.can_redo)
        with self.assertRaises(ValueError):
            history.preview(replace(branched.state, interval_actions=()), "Discard historical action")

    def test_decision_does_not_assert_pairing_fit_or_group_review(self):
        from seisslate.analysis import analyze
        from seisslate.evidence import PickDecision
        from seisslate.groups import load_group_csv

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "groups.csv"
            path.write_text("group_id,start_utc,end_utc,duration_s\nG1,2000-01-01T00:00:00Z,,1\n")
            groups = load_group_csv(path)
        first, *others = self.state.blows
        state = replace(self.state, blows=(replace(first, group_id="G1"), *others), groups=groups)
        decision = PickDecision(
            "B01", "original", self.original.ref(360), "unresolved", reason="Emergent onset", reviewer="Analyst",
        )
        reviewed = self.api.set_pick_decision(state, self.catalog, decision)
        self.assertEqual(analyze(self.catalog, state, reviewed), analyze(self.catalog, state, state))
        self.assertFalse(reviewed.groups[0].reviewed)
        self.assertEqual(reviewed.blows[0].pair_status, "confirmed")
        reviewed_group = self.api.set_group_reviewed(reviewed, "G1", True)
        self.assertEqual(reviewed_group.pick_decisions, (decision,))

    def test_save_omits_preview_but_keeps_applied_decisions_beyond_cursor(self):
        from seisslate.evidence import PickDecision
        from seisslate.session import load_session, save_session

        history = self.api.ReviewHistory(self.state)
        decision = PickDecision("B01", "original", self.original.ref(360), "candidate")
        state = self.api.set_pick_decision(self.state, self.catalog, decision)
        history.preview(state, "Propose sample")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "session.json"
            save_session(path, self.catalog, history)
            _, reopened = load_session(path)
            self.assertEqual(reopened.current.state, self.state)
            history.apply()
            history.undo()
            save_session(path, self.catalog, history)
            _, reopened = load_session(path)
            self.assertEqual(reopened.current.state, self.state)
            self.assertEqual(reopened.redo().state.pick_decisions, (decision,))

    def test_missing_reviewed_pick_can_be_supplied_without_changing_its_partner(self):
        blow = self.state.blows[2]
        self.assertEqual(blow.original, self.api.Pick(None))
        with self.assertRaises(ValueError):
            self.api.nudge_pick(self.state, self.catalog, "B07", "original", 1)
        offered = self.catalog.sources[0].segments[1].ref(0)
        changed = self.api.move_pick(self.state, self.catalog, "B07", "original", offered)
        self.assertEqual(changed.blows[2].original, self.api.Pick(None, None, offered))
        self.assertEqual(changed.blows[2].calibration, blow.calibration)
        with self.assertRaises(ValueError):
            self.api.move_pick(self.state, self.catalog, "B07", "original", self.calibration.ref(0))

    def test_nudge_does_not_cross_segment_boundary(self):
        at_end = self.api.move_pick(self.state, self.catalog, "B01", "original", self.original.ref(2149))
        with self.assertRaises(ValueError):
            self.api.nudge_pick(at_end, self.catalog, "B01", "original", 1)
        self.assertEqual(at_end.blows[0].original.reviewed_ref.sample_index, 2149)

    def test_resolved_pick_stays_on_its_recording_when_channel_names_match(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "same-channel.mseed"
            alternate_trace = read(str(self.paths[0]), format="MSEED")[0]
            alternate_trace.data[0] += 1
            alternate_trace.write(str(path), format="MSEED")
            alternate = load_source(path)
        catalog = SourceCatalog((*self.catalog.sources, alternate))
        state = replace(self.state, receivers=(
            replace(self.state.receivers[0], source_hash=""),
            self.state.receivers[1],
        ))
        with self.assertRaises(ValueError):
            self.api.move_pick(state, catalog, "B01", "original", alternate.segments[0].ref(360))

    def test_pair_and_fit_exclusion_have_independent_reasons_and_keep_picks(self):
        ambiguous = self.api.set_pair_status(self.state, "B02", "ambiguous", "Two close candidates")
        omitted = self.api.set_inclusion(ambiguous, "B02", False, "Excluded from numerical fit")
        blow = omitted.blows[1]
        self.assertEqual(blow.pair_status, "ambiguous")
        self.assertEqual(blow.pair_reason, "Two close candidates")
        self.assertFalse(blow.included)
        self.assertEqual(blow.reason, "Excluded from numerical fit")
        self.assertEqual(blow.original, self.state.blows[1].original)
        self.assertEqual(blow.calibration, self.state.blows[1].calibration)

    def test_apply_undo_redo_and_branch_restore_complete_states(self):
        history = self.api.ReviewHistory(self.state)
        moved = self.api.nudge_pick(self.state, self.catalog, "B01", "original", 1)
        history.preview(moved, "Nudge original")
        first = history.apply()
        masked = self.api.mask_spans(moved, (SampleSpan(self.original.segment_id, 361, 362),))
        history.preview(masked, "Mask arrival")
        second = history.apply()
        self.assertEqual(history.current.state, masked)
        self.assertEqual(history.undo(), first)
        self.assertEqual(history.redo(), second)
        history.undo()
        branched = self.api.set_inclusion(moved, "B02", False, "Fit review")
        history.preview(branched, "Change fit inclusion")
        branch = history.apply()
        self.assertGreater(branch.revision_id, second.revision_id)
        self.assertEqual(history.redo(), branch)
        self.assertEqual(history.revisions, (history.revisions[0], first, branch))
        self.assertEqual(history.undo().state, moved)
        self.assertEqual(history.undo().state, self.state)
        reopened = self.api.ReviewHistory(
            self.state, revisions=history.revisions, cursor=history.cursor, next_revision_id=history.next_revision_id
        )
        self.assertEqual(reopened.redo().state, moved)
        self.assertEqual(reopened.redo().state, branched)

    def test_new_blow_and_geometry_commit_preserve_existing_blow_ids(self):
        empty = replace(self.state, blows=())
        created = self.api.Blow("N01", self.api.Pick(None), self.api.Pick(None), None, None)
        added = self.api.add_blow(empty, created)
        placed = self.api.set_geometry(added, "N01", 0, 44.2)
        self.assertEqual(placed.blows, (replace(created, distance_original_m=0, distance_calibration_m=44.2),))
        with self.assertRaises(ValueError):
            self.api.add_blow(placed, created)
        with self.assertRaises(ValueError):
            self.api.set_geometry(placed, "N01", -1, 44.2)
        with self.assertRaises(ValueError):
            self.api.set_geometry(placed, "N01", 1, float("inf"))


if __name__ == "__main__":
    unittest.main()
