import csv
import json
import os
import shutil
import subprocess
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "synthetic"
HISTORICAL = FIXTURE.with_name("review-evidence")


class SessionTest(unittest.TestCase):
    def setUp(self):
        from seisslate import session
        from seisslate.example import load_example

        self.api = session
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.folder = Path(directory.name) / "Test folder é"
        self.folder.mkdir()
        self.copies = self.folder / "recordings"
        shutil.copytree(FIXTURE, self.copies)
        self.catalog, self.history = load_example(self.copies)
        self.path = self.folder / "Pilot review.json"

    def apply_nudge(self):
        from seisslate.review import nudge_pick

        state = nudge_pick(self.history.current.state, self.catalog, "B01", "original", 1)
        self.history.preview(state, "Nudge first onset")
        return self.history.apply()

    def apply_evidence(self):
        from seisslate.evidence import PickDecision, RawEvidenceView
        from seisslate.review import mask_spans, set_pick_decision
        from seisslate.sources import SampleSpan

        state = self.history.current.state
        point = state.blow("B01").original.reviewed_ref
        segment = self.catalog.segment(point)
        span = SampleSpan(segment.segment_id, 355, 366)
        bounds = segment.ref(359), segment.ref(361)
        decision = PickDecision(
            "B01", "original", point, "accepted", *bounds, reason="Clear onset", reviewer="Reviewer é",
            inspection_span=span,
        )
        state = set_pick_decision(state, self.catalog, decision)
        state = mask_spans(
            state, (SampleSpan(segment.segment_id, 350, 351),), catalog=self.catalog, reason="Separate transient",
        )
        self.history.preview(state, "Waveform review")
        view = RawEvidenceView(
            point.source_hash, point.channel, span, segment.time_ns(355), segment.time_ns(366), point, bounds,
        )
        confirmation = self.history.confirm_current_pick_evidence(self.catalog, decision, view)
        return self.history.apply_with_evidence(
            self.catalog, confirmation, view, captured_utc_ns=1_800_000_000_000_000_000,
            software_versions=(("SeisSlate", "test"), ("NumPy", "test")),
        )

    def test_v3_round_trips_full_evidence_in_redo_and_omits_draft(self):
        from seisslate.review import set_inclusion

        evidence = self.apply_evidence()
        initial = self.history.undo()
        self.history.preview(set_inclusion(initial.state, "B01", False, "Draft-only edit"), "Draft-only revision")
        self.api.save_session(self.path, self.catalog, self.history)
        saved = json.loads(self.path.read_bytes())
        self.assertEqual(set(saved), {
            "schema_version", "mode", "method_version", "sources", "revisions", "cursor", "next_revision_id",
        })
        self.assertEqual((saved["schema_version"], saved["mode"]), (3, "paired"))
        self.assertTrue(all(revision["state"]["rms"] is None for revision in saved["revisions"]))
        self.assertNotIn(b"Draft-only", self.path.read_bytes())
        catalog, reopened = self.api.load_session(self.path)
        self.assertEqual(reopened.current, initial)
        self.assertEqual(reopened.next_revision_id, self.history.next_revision_id)
        self.assertEqual(reopened.redo(), evidence)
        self.assertTrue(reopened.replay_pick_evidence(catalog, "B01", "original").attestation_current)
        self.assertEqual(reopened.current.state.interval_actions[0].reason, "Separate transient")

    def test_v3_rejects_unknown_and_malformed_contents_in_retained_redo(self):
        self.apply_evidence()
        self.history.undo()
        self.api.save_session(self.path, self.catalog, self.history)
        original = self.path.read_bytes()
        base = ("revisions", 1, "state")
        record = base + ("raw_evidence", 0)
        confirmation = record + ("confirmation",)
        decision = base + ("pick_decisions", 0)
        drop = object()
        for fields, invalid, message in (
            (("schema_version",), 4, "Unsupported session schema version"),
            (("schema_version",), True, "schema version"),
            (("mode",), drop, "fields"),
            (("mode",), "unknown-mode", "Unsupported session mode"),
            (("mode",), "single-channel", "single-channel"),
            (("mode",), False, "session mode"),
            (("rms",), None, "fields"),
            (base + ("rms",), drop, "fields"),
            (base + ("rms",), {}, "RMS"),
            (base + ("pick_decisions",), True, "decisions"),
            (decision + ("role",), "calibration", "reviewed sample"),
            (decision + ("evidence_id",), "dangling", "linked raw evidence"),
            (decision + ("upper_ref", "sample_index"), 358, "Inclusive bounds"),
            (decision + ("lower_ref", "sample_index"), True, "sample index"),
            (record + ("applied_revision_id",), 2, "applied revision"),
            (confirmation + ("samples", "sample_count"), True, "sample count"),
            (confirmation + ("view", "bounds"), [None, None], "bounds"),
            (confirmation + ("samples", "extra"), 2, "fields"),
            (record + ("software_versions",), ["ab"], "version"),
        ):
            with self.subTest(fields=fields, invalid=invalid):
                raw = json.loads(original)
                node = raw
                for field in fields[:-1]:
                    node = node[field]
                if invalid is drop:
                    del node[fields[-1]]
                else:
                    node[fields[-1]] = invalid
                self.path.write_text(json.dumps(raw), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    self.api.load_session(self.path)

        duplicate = original.replace(b'"sample_count": 11,', b'"sample_count": 11, "sample_count": 11,', 1)
        self.assertNotEqual(duplicate, original)
        self.path.write_bytes(duplicate)
        with self.assertRaisesRegex(ValueError, "Duplicate JSON fields"):
            self.api.load_session(self.path)

    def test_v3_preserves_digest_mismatch_and_unsupported_method_for_replay(self):
        self.apply_evidence()
        self.api.save_session(self.path, self.catalog, self.history)
        original = self.path.read_bytes()
        for method_id, expected in (("raw-v1", "raw_samples_differ"), ("raw-v99", "unsupported_method")):
            with self.subTest(method_id=method_id):
                raw = json.loads(original)
                record = raw["revisions"][-1]["state"]["raw_evidence"][0]
                record["confirmation"]["samples"]["digest"] = "0" * 64
                record["method_id"] = method_id
                self.path.write_text(json.dumps(raw), encoding="utf-8")
                catalog, history = self.api.load_session(self.path)
                replay = history.replay_pick_evidence(catalog, "B01", "original")
                self.assertEqual(replay.raw_samples, expected)
                self.assertFalse(replay.attestation_current)

    def test_v3_rejects_a_renamed_evidence_id_that_would_collide_with_a_future_capture(self):
        self.apply_evidence()
        self.api.save_session(self.path, self.catalog, self.history)
        original = self.path.read_bytes()
        for invalid in ("raw-2", "other-1"):
            with self.subTest(evidence_id=invalid):
                raw = json.loads(original)
                state = raw["revisions"][1]["state"]
                state["raw_evidence"][0]["evidence_id"] = invalid
                state["pick_decisions"][0]["evidence_id"] = invalid
                self.path.write_text(json.dumps(raw), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "evidence ID"):
                    self.api.load_session(self.path)

    def test_fresh_process_recomputes_applied_revision_and_replays_undo_redo(self):
        self.apply_nudge()
        self.api.save_session(self.path, self.catalog, self.history)
        result = subprocess.run(
            [sys.executable, "-B", "-c", """
import json
import sys
from seisslate.analysis import analyze
from seisslate.session import load_session

catalog, history = load_session(sys.argv[1])
result = analyze(catalog, history.revisions[0].state, history.current.state)
print(json.dumps([
    history.current.revision_id, result.reviewed.slope_m_per_s, result.reviewed.intercept_m,
    result.reviewed.blow_ids, history.undo().revision_id, history.redo().revision_id,
]))
""", str(self.path)],
            check=True, capture_output=True, text=True,
        )
        revision, slope, intercept, included, undo_id, redo_id = json.loads(result.stdout)
        self.assertEqual((revision, undo_id, redo_id), (1, 0, 1))
        self.assertEqual(included, ["B01", "B02", "B03", "B08"])
        self.assertAlmostEqual(slope, 148750 / 293, delta=1e-9)
        self.assertAlmostEqual(intercept, 546 / 293, delta=1e-9)

    def test_moving_sibling_recordings_and_reviews_preserves_reopen_without_relink(self):
        reviews = self.folder / "reviews"
        reviews.mkdir()
        path = reviews / self.path.name
        self.apply_nudge()
        self.api.save_session(path, self.catalog, self.history)
        moved = self.folder.rename(self.folder.with_name("Moved project"))
        relocated_path = moved / reviews.name / path.name
        catalog, history = self.api.load_session(relocated_path)
        self.assertEqual(history.current, self.history.current)
        self.assertEqual(catalog.sources[0].path, (moved / self.copies.name / "original.mseed").resolve())
        saved = json.loads(relocated_path.read_text(encoding="utf-8"))
        self.assertEqual(saved["sources"][0]["path"], "../recordings/original.mseed")

    def test_missing_source_can_be_relinked_to_same_hash_in_unicode_path(self):
        source = self.catalog.sources[0]
        self.api.save_session(self.path, self.catalog, self.history)
        moved_path = source.path.rename(self.folder / "relinked démarrage.mseed")
        with self.assertRaises(self.api.MissingSourceError) as raised:
            self.api.load_session(self.path)
        self.assertEqual(raised.exception.expected_hash, source.sha256)
        self.assertEqual(raised.exception.path, source.path)
        with self.assertRaises(self.api.SourceMismatchError):
            self.api.load_session(self.path, relink={source.sha256: self.catalog.sources[1].path})
        catalog, history = self.api.load_session(self.path, relink={source.sha256: moved_path})
        self.assertEqual(catalog.sources[0].path, moved_path.resolve())
        self.assertEqual(history.current, self.history.current)

    def test_wrong_source_hash_is_rejected(self):
        source = self.catalog.sources[0]
        self.api.save_session(self.path, self.catalog, self.history)
        source.path.write_bytes(b"different recording")
        with self.assertRaises(self.api.SourceMismatchError) as raised:
            self.api.load_session(self.path)
        self.assertEqual(raised.exception.expected_hash, source.sha256)

    def test_windows_network_recording_path_requires_explicit_relink(self):
        source = self.catalog.sources[0]
        self.api.save_session(self.path, self.catalog, self.history)
        saved = json.loads(self.path.read_text(encoding="utf-8"))
        saved["sources"][0]["path"] = r"\\unsigned-server\recordings\original.mseed"
        self.path.write_text(json.dumps(saved), encoding="utf-8")
        with patch.object(self.api, "load_source", wraps=self.api.load_source) as reader:
            with self.assertRaises(self.api.MissingSourceError) as raised:
                self.api.load_session(self.path)
        reader.assert_not_called()
        self.assertEqual(raised.exception.expected_hash, source.sha256)

    def test_invalid_method_type_and_sample_reference_are_rejected(self):
        self.api.save_session(self.path, self.catalog, self.history)
        document = self.path.read_text(encoding="utf-8")
        for fields, invalid in (
            (("method_version",), "unknown-method"),
            (("cursor",), True),
            (("revisions", 0, "state", "blows", 0, "original", "reviewed_ref", "sample_index"), 99999),
            (("revisions", 0, "state", "blows", 0, "distance_original_m"), "13.1"),
        ):
            with self.subTest(fields=fields):
                invalid_document = json.loads(document)
                item = invalid_document
                for field in fields[:-1]:
                    item = item[field]
                item[fields[-1]] = invalid
                self.path.write_text(json.dumps(invalid_document), encoding="utf-8")
                with self.assertRaises(ValueError):
                    self.api.load_session(self.path)

    def test_failed_atomic_replace_leaves_previous_session_and_cleans_temporary_file(self):
        self.api.save_session(self.path, self.catalog, self.history)
        prior = self.path.read_bytes()
        paths = set(self.folder.iterdir())
        self.apply_nudge()
        with patch.object(os, "replace", side_effect=PermissionError("Destination busy")):
            with self.assertRaises(PermissionError):
                self.api.save_session(self.path, self.catalog, self.history)
        self.assertEqual(self.path.read_bytes(), prior)
        self.assertEqual(set(self.folder.iterdir()), paths)
        _, reopened = self.api.load_session(self.path)
        self.assertEqual(reopened.current.revision_id, 0)

    def test_save_rejects_geometry_that_would_fail_to_reopen(self):
        from seisslate.review import ReviewHistory

        state = self.history.current.state
        invalid_blow = replace(state.blows[0], distance_original_m=-1)
        invalid_state = replace(state, blows=(invalid_blow, *state.blows[1:]))
        with self.assertRaises(ValueError):
            self.api.save_session(self.path, self.catalog, ReviewHistory(invalid_state))
        self.assertFalse(self.path.exists())

    def test_writers_reject_recording_paths_and_filesystem_aliases(self):
        source = self.catalog.sources[0]
        prior = source.path.read_bytes()
        alias = self.folder / "hard link"
        os.link(source.path, alias)
        for write in (self.api.save_session, self.api.export_csv):
            for path in (source.path, alias):
                with self.subTest(write=write.__name__, path=path):
                    with self.assertRaises(ValueError):
                        write(path, self.catalog, self.history)
        self.assertEqual(source.path.read_bytes(), prior)

    def test_csv_changes_with_applied_revision_and_distinguishes_exclusion_from_missing(self):
        from seisslate.review import set_inclusion

        self.apply_nudge()
        edited = set_inclusion(self.history.current.state, "B01", False, "=1+1")
        self.history.preview(edited, "Temporary fit comparison")
        path = self.folder / "review.csv"
        self.api.export_csv(path, self.catalog, self.history)
        with path.open(newline="", encoding="utf-8") as file:
            rows = {row["blow_id"]: row for row in csv.DictReader(file)}
        self.assertEqual((rows["B01"]["revision_id"], rows["B01"]["lag_ns"]), ("1", "-39000000"))
        self.assertEqual(rows["B01"]["effective_included"], "true")
        self.history.apply()
        self.api.export_csv(path, self.catalog, self.history)
        with path.open(newline="", encoding="utf-8") as file:
            rows = {row["blow_id"]: row for row in csv.DictReader(file)}
        excluded = rows["B01"]
        self.assertEqual((excluded["revision_id"], excluded["lag_ns"]), ("2", "-39000000"))
        self.assertEqual((excluded["requested_included"], excluded["effective_included"]), ("false", "false"))
        self.assertEqual(excluded["reason"], "'=1+1")
        self.assertTrue(excluded["residual_m"])
        missing = rows["B07"]
        self.assertEqual((missing["lag_ns"], missing["residual_m"]), ("", ""))
        self.assertEqual(missing["status"], "missing")

    def test_evidence_csv_reports_applied_targets_and_separates_staleness_from_raw_match(self):
        from seisslate.evidence import PickDecision
        from seisslate.review import mask_spans, set_pick_decision
        from seisslate.sources import SampleSpan

        captured = self.apply_evidence()
        record = captured.state.raw_evidence[0]
        segment = self.catalog.segment(record.confirmation.view.span.segment_id)
        stale = mask_spans(
            captured.state, (SampleSpan(segment.segment_id, 359, 360),),
            catalog=self.catalog, reason="Nearby artefact",
        )
        unresolved = PickDecision(
            "B07", "original", None, "unresolved", intended_utc_ns=segment.stop_ns,
            reason=" =1+1", reviewer="\t=unknown",
        )
        state = set_pick_decision(stale, self.catalog, unresolved)
        self.history.preview(state, "Record a gap and mask")
        exported = self.folder / "evidence.csv"

        def rows():
            self.api.export_evidence_csv(exported, self.catalog, self.history)
            with exported.open(newline="", encoding="utf-8") as file:
                return {(row["blow_id"], row["role"]): row for row in csv.DictReader(file)}

        draft_rows = rows()
        self.assertEqual(draft_rows[("B07", "original")]["decision_status"], "legacy-unassessed")
        self.assertEqual(draft_rows[("B01", "original")]["attestation_current"], "true")
        self.history.apply()
        applied = rows()
        row = applied[("B01", "original")]
        self.assertEqual((row["revision_id"], row["evidence_revision_id"], row["evidence_id"]), ("2", "1", "raw-1"))
        self.assertEqual((row["decision_status"], row["decision_reason"], row["reviewer"]),
                         ("accepted", "Clear onset", "Reviewer é"))
        self.assertEqual((row["imported_utc_ns"], row["original_sample_utc_ns"], row["reviewed_sample_utc_ns"]),
                         ("946684800360000000",) * 3)
        self.assertEqual((row["original_sample_index"], row["reviewed_sample_index"]), ("360", "360"))
        self.assertEqual(row["reviewed_source_hash"], self.catalog.sources[0].sha256)
        self.assertEqual(row["reviewed_segment_id"], segment.segment_id)
        self.assertEqual(row["reviewed_channel"], segment.channel)
        self.assertEqual((row["lower_bound_sample_index"], row["upper_bound_sample_index"]), ("359", "361"))
        self.assertEqual((row["lower_bound_utc_ns"], row["upper_bound_utc_ns"]),
                         ("946684800359000000", "946684800361000000"))
        self.assertEqual((row["evidence_method_id"], row["evidence_span_start"], row["evidence_span_stop_exclusive"]),
                         ("raw-v1", "355", "366"))
        self.assertEqual(row["evidence_raw_digest"], record.confirmation.samples.digest)
        self.assertEqual((row["source_identity"], row["raw_samples"], row["grid_matches"]),
                         ("source_match", "raw_samples_match", "true"))
        self.assertEqual((row["recorded_view_valid"], row["view_restoration"], row["mask_stale"],
                          row["attestation_current"]), ("true", "not_checked", "true", "false"))
        self.assertEqual(applied[("B01", "calibration")]["decision_status"], "legacy-unassessed")
        gap = applied[("B07", "original")]
        self.assertEqual((gap["reviewed_sample_index"], gap["evidence_id"], gap["attestation_current"]), ("", "", ""))
        self.assertEqual((gap["decision_status"], gap["raw_samples"], gap["intended_utc_ns"]),
                         ("unresolved", "not_recorded", str(segment.stop_ns)))
        self.assertEqual((gap["decision_reason"], gap["reviewer"]), ("' =1+1", "'\t=unknown"))
        self.history.undo()
        self.assertEqual(rows(), draft_rows)
        self.history.redo()
        self.assertEqual(rows(), applied)

    def test_evidence_export_replay_uses_the_same_revision_if_live_cursor_changes(self):
        captured = self.apply_evidence()
        output = self.folder / "evidence.csv"
        writer = csv.writer

        def change_cursor(file):
            self.history.undo()
            return writer(file)

        with patch.object(csv, "writer", side_effect=change_cursor):
            self.api.export_evidence_csv(output, self.catalog, self.history)
        self.assertEqual(self.history.current.revision_id, 0)
        with output.open(newline="", encoding="utf-8") as file:
            row = next(csv.DictReader(file))
        self.assertEqual(row["revision_id"], str(captured.revision_id))
        self.assertEqual((row["evidence_id"], row["decision_status"]), ("raw-1", "accepted"))
        self.assertEqual((row["raw_samples"], row["attestation_current"]), ("raw_samples_match", "true"))

    def test_evidence_export_from_legacy_review_keeps_exact_fit_bytes(self):
        source = HISTORICAL / "v2.session.json"
        original = source.read_bytes()
        catalog, history = self.api.load_session(source)
        evidence = self.folder / "evidence.csv"
        fit = self.folder / "fit.csv"
        self.api.export_evidence_csv(evidence, catalog, history, session_path=source)
        self.api.export_csv(fit, catalog, history, session_path=source)
        self.assertEqual(source.read_bytes(), original)
        self.assertEqual(fit.read_bytes(), (HISTORICAL / "applied.fit.csv").read_bytes())
        with evidence.open(newline="", encoding="utf-8") as file:
            rows = {(row["blow_id"], row["role"]): row for row in csv.DictReader(file)}
        pick = rows[("B01", "original")]
        self.assertEqual((pick["decision_status"], pick["evidence_id"], pick["raw_samples"]),
                         ("legacy-unassessed", "", "not_recorded"))
        self.assertEqual((pick["original_sample_index"], pick["reviewed_sample_index"], pick["reviewed_sample_utc_ns"]),
                         ("360", "361", "946684800361000000"))

    def test_exports_protect_active_session_path_and_filesystem_aliases(self):
        self.api.save_session(self.path, self.catalog, self.history)
        original = self.path.read_bytes()
        hardlink = self.folder / "review hardlink.csv"
        symlink = self.folder / "review symlink.csv"
        os.link(self.path, hardlink)
        aliases = [self.path, hardlink]
        try:
            symlink.symlink_to(self.path)
        except OSError as error:
            if sys.platform != "win32" or getattr(error, "winerror", None) != 1314:
                raise
        else:
            aliases.append(symlink)
        for export in (self.api.export_csv, self.api.export_evidence_csv):
            for path in aliases:
                with self.subTest(export=export.__name__, path=path), self.assertRaisesRegex(ValueError, "active.*session"):
                    export(path, self.catalog, self.history, session_path=self.path)
                self.assertEqual(path.read_bytes(), original)
        self.apply_nudge()
        self.api.save_session(self.path, self.catalog, self.history)
        self.assertNotEqual(self.path.read_bytes(), original)

    def test_evidence_csv_keeps_previous_bytes_if_atomic_replace_fails(self):
        output = self.folder / "evidence.csv"
        output.write_bytes(b"Previous evidence")
        paths = set(self.folder.iterdir())
        with patch.object(os, "replace", side_effect=PermissionError("Destination busy")):
            with self.assertRaises(PermissionError):
                self.api.export_evidence_csv(output, self.catalog, self.history)
        self.assertEqual(output.read_bytes(), b"Previous evidence")
        self.assertEqual(set(self.folder.iterdir()), paths)

    def test_evidence_csv_checks_destination_aliases_again_before_replacement(self):
        self.api.save_session(self.path, self.catalog, self.history)
        output = self.folder / "evidence.csv"
        original = self.path.read_bytes()
        fsync = os.fsync

        def place_alias(descriptor):
            fsync(descriptor)
            os.link(self.path, output)

        with patch.object(os, "fsync", side_effect=place_alias):
            with self.assertRaisesRegex(ValueError, "active.*session"):
                self.api.export_evidence_csv(output, self.catalog, self.history, session_path=self.path)
        self.assertEqual(self.path.read_bytes(), original)
        self.assertTrue(output.samefile(self.path))
        self.assertEqual(list(self.folder.glob(".seisslate-*")), [])

    def test_evidence_csv_protects_source_recordings_and_group_logs_in_redo(self):
        from seisslate.groups import load_group_csv
        from seisslate.review import add_groups

        group_log = self.folder / "Synthetic log.csv"
        group_log.write_text("group_id,start_utc,end_utc,duration_s\nG1,2000-01-01T00:00:00Z,,1\n", encoding="utf-8")
        with_log = add_groups(self.history.current.state, load_group_csv(group_log))
        self.history.preview(with_log, "Retain group log")
        self.history.apply()
        self.history.undo()
        source = self.catalog.sources[0].path
        for path, message in ((source, "recordings"), (group_log, "group log")):
            with self.subTest(path=path):
                original = path.read_bytes()
                with self.assertRaisesRegex(ValueError, message):
                    self.api.export_evidence_csv(path, self.catalog, self.history)
                self.assertEqual(path.read_bytes(), original)


class HistoricalSessionTest(unittest.TestCase):
    def test_frozen_v3_paired_fixture_resaves_without_wire_or_replay_changes(self):
        from seisslate.session import load_session, save_session

        fixture = HISTORICAL / "v3-paired.session.json"
        original = fixture.read_bytes()
        document = json.loads(original)
        state_keys = {
            "blows", "masks", "receivers", "timing_evidence", "groups", "pick_decisions", "interval_actions",
            "raw_evidence", "rms",
        }
        self.assertEqual((document["schema_version"], document["mode"], document["cursor"]), (3, "paired", 1))
        for revision in document["revisions"]:
            self.assertEqual(set(revision["state"]), state_keys)
            self.assertIsNone(revision["state"]["rms"])
        catalog, history = load_session(fixture)
        self.assertEqual(tuple(revision.revision_id for revision in history.revisions), (0, 1))
        decision = history.current.state.pick_decisions[0]
        self.assertEqual((decision.status, decision.reviewed_ref.sample_index, decision.evidence_id), ("accepted", 360, "raw-1"))
        self.assertTrue(history.replay_pick_evidence(catalog, "B01", "original").attestation_current)
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            copied = folder / "review-evidence"
            copied.mkdir()
            shutil.copytree(FIXTURE, folder / "synthetic")
            reopened_catalog, reopened_history = load_session(fixture, relink={
                source.sha256: folder / "synthetic" / source.path.name for source in catalog.sources
            })
            rewritten = copied / fixture.name
            save_session(rewritten, reopened_catalog, reopened_history)
            self.assertEqual(json.loads(rewritten.read_bytes()), document)
        self.assertEqual(fixture.read_bytes(), original)

    def test_initial_snapshot_accepts_baseline_masks_but_rejects_recorded_actions(self):
        from seisslate.session import load_session

        fixture = HISTORICAL / "v3-paired.session.json"
        document = json.loads(fixture.read_bytes())
        initial = document["revisions"][0]["state"]
        applied = document["revisions"][1]["state"]
        with TemporaryDirectory() as directory:
            path = Path(directory) / fixture.name
            relink = {
                source["sha256"]: FIXTURE / Path(source["path"]).name for source in document["sources"]
            }
            # The existing action is falsely carried forward from a baseline with empty masks.
            initial["interval_actions"] = applied["interval_actions"]
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "initial.*interval actions"):
                load_session(path, relink=relink)

            initial["interval_actions"] = applied["interval_actions"] = []
            initial["masks"] = applied["masks"]
            path.write_text(json.dumps(document), encoding="utf-8")
            _, history = load_session(path, relink=relink)
            self.assertEqual(len(history.revisions[0].state.masks), 1)

    def test_upgraded_v1_and_v2_reopen_in_fresh_process_with_exact_fit_exports(self):
        from seisslate.session import load_session, save_session

        with TemporaryDirectory() as directory:
            folder = Path(directory)
            for version in (1, 2):
                with self.subTest(version=version):
                    source = HISTORICAL / f"v{version}.session.json"
                    original = source.read_bytes()
                    catalog, history = load_session(source)
                    upgraded = folder / f"v{version}-upgraded.json"
                    save_session(upgraded, catalog, history)
                    self.assertEqual(json.loads(upgraded.read_bytes())["schema_version"], 3)
                    self.assertEqual(load_session(upgraded)[1].revisions, history.revisions)
                    self.assertEqual(source.read_bytes(), original)
                    subprocess.run(
                        [sys.executable, "-B", "-c", """
import sys
from pathlib import Path
from seisslate.session import export_csv, load_session

catalog, history = load_session(sys.argv[1])
assert (history.cursor, history.current.revision_id, history.next_revision_id) == (1, 1, 4)
assert history.current.state.blow('B01').original.reviewed_ref.sample_index == 361
out = Path(sys.argv[1]).parent
export_csv(out / 'applied.fit.csv', catalog, history)
assert history.redo().revision_id == 3
export_csv(out / 'redo.fit.csv', catalog, history)
""", str(upgraded)],
                        check=True, capture_output=True, text=True,
                    )
                    for filename in ("applied.fit.csv", "redo.fit.csv"):
                        self.assertEqual((folder / filename).read_bytes(), (HISTORICAL / filename).read_bytes())

    def test_historical_sessions_restore_applied_edits_and_unapplied_redo(self):
        from seisslate.session import load_session

        for version in (1, 2):
            with self.subTest(version=version):
                path = HISTORICAL / f"v{version}.session.json"
                original_bytes = path.read_bytes()
                self.assertEqual(json.loads(original_bytes)["schema_version"], version)
                catalog, history = load_session(path)
                self.assertEqual(
                    tuple(source.path for source in catalog.sources),
                    (FIXTURE / "original.mseed", FIXTURE / "calibration.mseed"),
                )
                self.assertEqual(tuple(revision.revision_id for revision in history.revisions), (0, 1, 3))
                self.assertEqual((history.cursor, history.current.revision_id, history.next_revision_id), (1, 1, 4))
                self.assertTrue(history.can_undo)
                self.assertTrue(history.can_redo)
                self.assertIsNone(history.draft)

                applied = history.current.state
                blow = applied.blow("B01")
                self.assertEqual(blow.original.imported_ns, 946_684_800_360_000_000)
                self.assertEqual(blow.original.original_ref.sample_index, 360)
                self.assertEqual(blow.original.reviewed_ref.sample_index, 361)
                self.assertEqual(blow.calibration.reviewed_ref.sample_index, 400)
                self.assertTrue(applied.is_masked(blow.original.original_ref))
                self.assertFalse(applied.is_masked(blow.original.reviewed_ref))
                self.assertFalse(applied.blow("B03").included)

                initial = history.undo().state
                self.assertEqual(initial.blow("B01").original.reviewed_ref.sample_index, 360)
                self.assertEqual(initial.masks, ())
                self.assertTrue(initial.blow("B03").included)
                self.assertEqual(initial.groups, ())
                self.assertEqual(history.redo().state, applied)
                redo = history.redo().state
                self.assertEqual(redo.blow("B01").original, blow.original)
                self.assertEqual(redo.blow("B01").calibration.reviewed_ref.sample_index, 401)
                self.assertEqual(redo.masks, applied.masks)
                self.assertFalse(history.can_redo)
                if version == 1:
                    self.assertEqual(applied.receivers, ())
                    self.assertEqual(applied.groups, ())
                    self.assertEqual(redo.groups, ())
                    self.assertIsNone(redo.blow("B01").group_id)
                else:
                    self.assertEqual(blow.group_id, "review-start")
                    (group,) = applied.groups
                    (reviewed_group,) = redo.groups
                    self.assertEqual((group.source_file, group.source_row), ("hammer-groups.csv", 2))
                    self.assertFalse(group.reviewed)
                    self.assertEqual(reviewed_group, replace(group, reviewed=True))
                self.assertEqual(path.read_bytes(), original_bytes)

    def test_historical_fit_csv_matches_writer_bytes_at_applied_cursor_and_after_redo(self):
        from seisslate.session import export_csv, load_session

        with TemporaryDirectory() as directory:
            path = Path(directory) / "fit.csv"
            for version in (1, 2):
                catalog, history = load_session(HISTORICAL / f"v{version}.session.json")
                for state in ("applied", "redo"):
                    with self.subTest(version=version, state=state):
                        export_csv(path, catalog, history)
                        expected = HISTORICAL / f"{state}.fit.csv"
                        self.assertEqual(path.read_bytes(), expected.read_bytes())
                    if state == "applied":
                        history.redo()

    def test_historical_readers_enforce_version_specific_keys_in_redo(self):
        from seisslate.session import load_session

        with TemporaryDirectory() as directory:
            path = Path(directory) / "review.json"
            for version in (1, 2):
                for fields, key, value in (((), "groups", []), (("blows", 0), "group_id", None)):
                    with self.subTest(version=version, key=key):
                        fixture = HISTORICAL / f"v{version}.session.json"
                        document = json.loads(fixture.read_bytes())
                        state = document["revisions"][-1]["state"]
                        for field in fields:
                            state = state[field]
                        if version == 1:
                            state[key] = value
                        else:
                            del state[key]
                        path.write_text(json.dumps(document), encoding="utf-8")
                        relink = {
                            source["sha256"]: FIXTURE / Path(source["path"]).name
                            for source in document["sources"]
                        }
                        with self.assertRaisesRegex(ValueError, "fields"):
                            load_session(path, relink=relink)


if __name__ == "__main__":
    unittest.main()
