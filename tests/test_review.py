from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from obspy import read

from seisslate.sources import SampleSpan, SourceCatalog, load_source


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
