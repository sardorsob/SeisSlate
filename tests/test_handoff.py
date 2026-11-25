import csv
import json
import os
import shutil
import subprocess
import sys
import unittest
from dataclasses import replace
from fractions import Fraction
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

from seisslate import session
from seisslate.review import ReviewHistory, ReviewState, set_rms_review
from seisslate.rms import EpochEvidence, RmsEvidence, RmsReview, RmsWindow, RmsWindows
from seisslate.sources import SourceCatalog, load_source


class HandoffTest(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.bundle = self.root / "sender"
        recordings = self.bundle / "recordings"
        reviews = self.bundle / "reviews"
        recordings.mkdir(parents=True)
        reviews.mkdir()
        self.recording = recordings / "receiver é.mseed"
        self.fixture = Path(__file__).resolve().parents[1] / "fixtures" / "synthetic"
        shutil.copyfile(self.fixture / "original.mseed", self.recording)
        self.catalog = SourceCatalog((load_source(self.recording),))
        self.segment = self.catalog.segments[0]
        self.identity = self.segment.source_hash, self.segment.channel
        self.history = ReviewHistory(ReviewState.for_single_channel(*self.identity))
        self.session_path = reviews / "review.json"
        self.manifest_path = self.bundle / "handoff.json"
        self.csv_path = self.bundle / "rms.csv"
        self.metadata = self.bundle / "instrument.txt"

    def apply(self, *, evidence=True, checked=True, invalid=False):
        def grid(index):
            return self.segment.start_ns + index * self.segment.period_ns

        event = RmsWindow(*self.identity, grid(355) - Fraction(1, 3), grid(365))
        background = RmsWindow(*self.identity, grid(375), grid(386 if invalid else 385))
        self.metadata.write_bytes(b"Synthetic instrument evidence")
        reference = RmsEvidence(
            *self.identity, grid(350), grid(500), "=instrument record",
            sha256(self.metadata.read_bytes()).hexdigest(), "+synthetic",
        ) if evidence else None
        review = RmsReview(
            RmsWindows(*self.identity, event, background, EpochEvidence(reference, "@reviewed") if reference else None,
                       reference),
            "accepted", "@Reviewer", "  =review complete",
        )
        state = set_rms_review(self.history.current.state, self.catalog, review)
        self.history.preview_rms_review(state, "Applied selection")
        return self.history.apply_rms(self.catalog, checked_evidence=(reference,) if reference and checked else ())

    def rms_row(self, catalog=None, history=None):
        session.export_rms_csv(self.csv_path, catalog or self.catalog, history or self.history,
                               session_path=self.session_path)
        with self.csv_path.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(len(rows), 1)
        return rows[0]

    def test_rms_export_reports_exact_applied_intent_and_replay_qualification(self):
        applied = self.apply()
        self.history.preview(replace(applied.state, rms=replace(applied.state.rms, reviewer="Draft")), "Draft")
        row = self.rms_row()
        self.assertEqual(row["revision_id"], "1")
        self.assertEqual(row["disposition"], "accepted")
        self.assertEqual(row["reviewer"], "'@Reviewer")
        self.assertEqual(row["decision_reason"], "'  =review complete")
        self.assertEqual(row["event_intent_start_utc_ns_denominator"], "3")
        self.assertEqual(Fraction(int(row["event_intent_start_utc_ns_numerator"]), 3),
                         applied.state.rms.windows.event.start_ns)
        self.assertEqual(row["current_event_span_start"], "355")
        self.assertEqual(row["current_event_span_stop_exclusive"], "365")
        self.assertEqual(row["current_samples_per_window"], "10")
        self.assertEqual(row["epoch_evidence_locator"], "'=instrument record")
        self.assertEqual(row["epoch_verification"], "'@reviewed")
        self.assertEqual(row["counts_evidence_provenance"], "'+synthetic")
        self.assertEqual(row["baseline_epoch_status"], "independently_verified")
        self.assertEqual(row["current_epoch_status"], "unverified")
        self.assertEqual(row["baseline_units"], "raw counts")
        self.assertEqual(row["current_units"], "raw units unknown")
        self.assertEqual(row["current_qualified"], "true")
        self.assertEqual(row["replay_status"], "numeric_match")
        self.assertEqual(row["ready"], "true")
        self.assertEqual(float(row["contrast_delta"]), 0.0)
        self.assertEqual(self.history.draft.state.rms.reviewer, "Draft")

    def test_rms_export_keeps_accepted_unavailable_numbers_blank(self):
        self.apply(invalid=True)
        row = self.rms_row()
        self.assertEqual(row["disposition"], "accepted")
        self.assertEqual(row["current_reasons"], "unequal_length")
        self.assertEqual(row["replay_status"], "numeric_match")
        self.assertEqual(row["ready"], "false")
        self.assertEqual(row["current_contrast"], "")
        self.assertEqual(row["contrast_delta"], "")
        self.assertEqual(row["current_duration_ns_numerator"], "")

    def test_rms_export_distinguishes_mismatch_from_human_disposition(self):
        applied = self.apply()
        session.save_session(self.session_path, self.catalog, self.history)
        document = json.loads(self.session_path.read_bytes())
        document["revisions"][1]["state"]["rms"]["baseline"]["contrast"] *= 2
        self.session_path.write_text(json.dumps(document), encoding="utf-8")
        row = self.rms_row(*session.load_session(self.session_path))
        self.assertEqual(row["disposition"], "accepted")
        self.assertEqual(row["replay_status"], "numeric_mismatch")
        self.assertEqual(row["ready"], "false")
        self.assertEqual(float(row["current_contrast"]), applied.state.rms.baseline.contrast)
        self.assertEqual(float(row["contrast_delta"]), -applied.state.rms.baseline.contrast)

    def test_manifest_relocates_through_cli_and_fresh_process(self):
        self.apply()
        session.save_session(self.session_path, self.catalog, self.history)
        create = self.run_cli("create", self.manifest_path, self.session_path, "--log", self.metadata)
        self.assertEqual(create.returncode, 0, create.stderr)
        raw = json.loads(self.manifest_path.read_bytes())
        self.assertEqual(raw["session"]["path"], "reviews/review.json")
        self.assertEqual(raw["sources"][0]["path"], "recordings/receiver é.mseed")
        self.assertEqual(raw["logs"][0]["path"], "instrument.txt")
        self.assertEqual(raw["logs"][0]["sha256"], sha256(self.metadata.read_bytes()).hexdigest())
        self.assertNotIn(str(self.root), self.manifest_path.read_text(encoding="utf-8"))
        destination = self.bundle.rename(self.root / "recipient")
        verify = self.run_cli("verify", destination / "handoff.json")
        self.assertEqual(verify.returncode, 0, verify.stderr)
        self.assertIn("numeric_match", verify.stdout)
        self.assertIn("unverified", verify.stdout)

    def run_cli(self, *args):
        checkout = Path(__file__).resolve().parents[1]
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(checkout), env.get("PYTHONPATH"))))
        return subprocess.run([sys.executable, "-B", "-m", "seisslate.handoff", *map(str, args)],
                              cwd=self.root, env=env, capture_output=True, text=True)

    def test_manifest_rejects_wrong_missing_and_escaped_recordings(self):
        self.apply()
        session.save_session(self.session_path, self.catalog, self.history)
        session.write_handoff_manifest(self.manifest_path, self.session_path)
        document = json.loads(self.manifest_path.read_bytes())
        self.assertEqual(document["logs"], [])
        original = self.recording.read_bytes()
        self.recording.write_bytes((self.fixture / "calibration.mseed").read_bytes())
        with self.assertRaises(session.SourceMismatchError):
            session.load_handoff(self.manifest_path)
        self.recording.unlink()
        with self.assertRaises(session.MissingSourceError):
            session.load_handoff(self.manifest_path)
        outside = self.root / "outside.mseed"
        outside.write_bytes(original)
        self.recording.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "handoff directory"):
            session.load_handoff(self.manifest_path)
        self.recording.unlink()
        self.recording.write_bytes(original)
        document["sources"] = []
        self.manifest_path.write_text(json.dumps(document), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "absent from the handoff manifest"):
            session.load_handoff(self.manifest_path)

    def test_manifest_checks_session_and_only_explicit_log_bytes(self):
        self.apply()
        session.save_session(self.session_path, self.catalog, self.history)
        session.write_handoff_manifest(self.manifest_path, self.session_path, logs=[self.metadata])
        self.metadata.write_bytes(b"Wrong instrument evidence")
        with self.assertRaisesRegex(ValueError, "Log SHA-256 mismatch"):
            session.load_handoff(self.manifest_path)
        self.metadata.unlink()
        with self.assertRaisesRegex(ValueError, "Missing handoff log"):
            session.load_handoff(self.manifest_path)
        session.write_handoff_manifest(self.manifest_path, self.session_path)
        session.load_handoff(self.manifest_path)
        with self.session_path.open("ab") as file:
            file.write(b"\n")
        with self.assertRaisesRegex(ValueError, "Session SHA-256 mismatch"):
            session.load_handoff(self.manifest_path)

    def test_unsupported_rms_method_remains_visible_and_unready_after_handoff(self):
        self.apply()
        session.save_session(self.session_path, self.catalog, self.history)
        document = json.loads(self.session_path.read_bytes())
        document["revisions"][1]["state"]["rms"]["method_version"] = "future-rms-v99"
        self.session_path.write_text(json.dumps(document), encoding="utf-8")
        session.write_handoff_manifest(self.manifest_path, self.session_path)
        row = self.rms_row(*session.load_handoff(self.manifest_path))
        self.assertEqual(row["method_version"], "future-rms-v99")
        self.assertEqual(row["replay_status"], "unsupported_method")
        self.assertEqual(row["ready"], "false")
        self.assertEqual(row["current_contrast"], "")
        self.assertTrue(row["baseline_contrast"])
        verify = self.run_cli("verify", self.manifest_path)
        self.assertNotEqual(verify.returncode, 0)
        self.assertIn("unsupported_method", verify.stdout)

    def test_rms_export_and_manifest_cannot_overwrite_inputs_while_active_session_save_works(self):
        self.apply()
        session.save_session(self.session_path, self.catalog, self.history)
        initial = self.session_path.read_bytes()
        self.history.undo()
        session.save_session(self.session_path, self.catalog, self.history)
        self.assertNotEqual(self.session_path.read_bytes(), initial)
        self.history.redo()
        session.save_session(self.session_path, self.catalog, self.history)
        self.assertEqual(self.session_path.read_bytes(), initial)
        for path in (self.session_path, self.recording, self.metadata):
            before = path.read_bytes()
            alias = self.bundle / "alias"
            for make_alias in (os.link, os.symlink):
                make_alias(path, alias)
                with self.subTest(protected=path.name, alias=make_alias.__name__):
                    with self.assertRaises(ValueError):
                        session.export_rms_csv(alias, self.catalog, self.history, session_path=self.session_path)
                    with self.assertRaises(ValueError):
                        session.write_handoff_manifest(alias, self.session_path, logs=[self.metadata])
                    self.assertEqual(path.read_bytes(), before)
                    self.assertEqual(alias.read_bytes(), before)
                alias.unlink()

    def test_historical_provenance_cannot_claim_absent_evidence(self):
        self.apply(evidence=False)
        session.save_session(self.session_path, self.catalog, self.history)
        original = self.session_path.read_bytes()
        for field, status in (
            ("units", "raw counts"), ("epoch_status", "independently_verified"),
            ("epoch_status", "conflicting"), ("epoch_status", "unverified"),
        ):
            with self.subTest(field=field, status=status):
                document = json.loads(original)
                document["revisions"][1]["state"]["rms"]["baseline"][field] = status
                self.session_path.write_text(json.dumps(document), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "baseline provenance"):
                    session.load_session(self.session_path)

    def test_historical_provenance_allows_unchecked_retained_evidence(self):
        self.apply(checked=False)
        session.save_session(self.session_path, self.catalog, self.history)
        _, history = session.load_session(self.session_path)
        baseline = history.current.state.rms.baseline
        self.assertEqual(baseline.epoch_status, "unverified")
        self.assertEqual(baseline.units, "raw units unknown")

    def test_historical_provenance_rejects_insufficient_trace_coverage_or_verification(self):
        self.apply()
        session.save_session(self.session_path, self.catalog, self.history)
        original = self.session_path.read_bytes()
        for record_name, field, replacement in (
            ("counts_evidence", "channel", "other"),
            ("counts_evidence", "stop_ns", {"numerator": self.segment.time_ns(360), "denominator": 1}),
            ("epoch_evidence", "verification", ""),
            ("epoch_evidence", "conflicting", True),
        ):
            with self.subTest(record=record_name, field=field):
                document = json.loads(original)
                document["revisions"][1]["state"]["rms"]["windows"][record_name][field] = replacement
                self.session_path.write_text(json.dumps(document), encoding="utf-8")
                message = "selected source and channel" if field == "channel" else "baseline provenance"
                with self.assertRaisesRegex(ValueError, message):
                    session.load_session(self.session_path)

    def test_historical_conflict_status_requires_matching_unavailability_reason(self):
        applied = self.apply()
        windows = applied.state.rms.windows
        review = replace(applied.state.rms, windows=replace(
            windows, epoch_evidence=replace(windows.epoch_evidence, conflicting=True),
        ), baseline=None)
        self.history.preview_rms_review(set_rms_review(applied.state, self.catalog, review), "Record conflict")
        self.history.apply_rms(self.catalog)
        session.save_session(self.session_path, self.catalog, self.history)
        catalog, history = session.load_session(self.session_path)
        replay = history.replay_rms(catalog)
        self.assertEqual(replay.current.epoch_status, "conflicting")
        self.assertEqual(replay.current.reasons, ("epoch_conflict",))
        self.assertFalse(replay.ready)

        document = json.loads(self.session_path.read_bytes())
        baseline = document["revisions"][2]["state"]["rms"]["baseline"]
        baseline["reasons"] = ["zero_background"]
        self.session_path.write_text(json.dumps(document), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "conflict reason"):
            session.load_session(self.session_path)

    def test_paired_handoff_retains_explicit_group_log_and_exact_fit_bytes_on_reopen(self):
        from seisslate.example import load_example
        from seisslate.groups import load_group_csv
        from seisslate.review import add_groups, nudge_pick

        fixture = self.bundle / "paired"
        shutil.copytree(self.fixture, fixture)
        catalog, history = load_example(fixture)
        log = self.bundle / "group.csv"
        log.write_text("group_id,start_utc,end_utc,duration_s\nG1,2000-01-01T00:00:00Z,,1\n", encoding="utf-8")
        state = add_groups(history.current.state, load_group_csv(log))
        history.preview(nudge_pick(state, catalog, "B01", "original", 1), "Apply group and pick")
        history.apply()
        session.save_session(self.session_path, catalog, history)
        session.export_csv(self.csv_path, catalog, history, session_path=self.session_path)
        before = self.csv_path.read_bytes()
        session.write_handoff_manifest(self.manifest_path, self.session_path, logs=[log])
        relocated = self.bundle.rename(self.root / "recipient")
        catalog, history = session.load_handoff(relocated / self.manifest_path.name)
        output = relocated / "replayed-fit.csv"
        session.export_csv(output, catalog, history)
        self.assertEqual(output.read_bytes(), before)
        self.assertEqual(history.current.state.groups[0].source_sha256, sha256((relocated / log.name).read_bytes()).hexdigest())

    def paired_fixture(self, name="v3-paired.session.json"):
        shutil.copytree(self.fixture, self.bundle / "synthetic")
        original = self.fixture.with_name("review-evidence") / name
        shutil.copyfile(original, self.session_path)

    def verify_paired_fixture(self):
        created = self.run_cli("create", self.manifest_path, self.session_path)
        self.assertEqual(created.returncode, 0, created.stderr)
        return self.run_cli("verify", self.manifest_path)

    def paired_row(self, verified):
        rows = [line for line in verified.stdout.splitlines() if line.startswith("B01 original ")]
        self.assertEqual(len(rows), 1, verified.stdout)
        return rows[0]

    def test_paired_cli_verifies_current_evidence_and_human_decision(self):
        self.paired_fixture()
        verified = self.verify_paired_fixture()
        self.assertEqual(verified.returncode, 0, verified.stderr)
        row = self.paired_row(verified)
        self.assertIn("decision accepted", row)
        self.assertIn("raw_samples_match", row)
        self.assertIn("attestation current", row)
        self.assertIn("ready true", row)

    def test_paired_cli_marks_legacy_missing_evidence_unassessed(self):
        self.paired_fixture("v1.session.json")
        verified = self.verify_paired_fixture()
        self.assertEqual(verified.returncode, 0, verified.stderr)
        row = self.paired_row(verified)
        self.assertIn("decision legacy-unassessed", row)
        self.assertIn("attestation unassessed", row)
        self.assertIn("ready false", row)

    def test_paired_cli_fails_for_unsupported_or_differing_recorded_evidence(self):
        self.paired_fixture()
        original = self.session_path.read_bytes()
        for field, replacement, status in (
            ("method", "raw-v99", "unsupported_method"),
            ("digest", "0" * 64, "raw_samples_differ"),
        ):
            with self.subTest(field=field):
                document = json.loads(original)
                record = document["revisions"][1]["state"]["raw_evidence"][0]
                if field == "method":
                    record["method_id"] = replacement
                else:
                    record["confirmation"]["samples"]["digest"] = replacement
                self.session_path.write_text(json.dumps(document), encoding="utf-8")
                verified = self.verify_paired_fixture()
                self.assertNotEqual(verified.returncode, 0, verified.stdout)
                row = self.paired_row(verified)
                self.assertIn("decision accepted", row)
                self.assertIn(status, row)
                self.assertIn("ready false", row)

    def test_paired_cli_reports_stale_attestation_despite_matching_raw_bytes(self):
        from seisslate.review import mask_spans
        from seisslate.sources import SampleSpan

        self.paired_fixture()
        catalog, history = session.load_session(self.session_path)
        active = history.current.state
        span = active.raw_evidence[0].confirmation.view.span
        state = mask_spans(active, (SampleSpan(span.segment_id, span.start, span.start + 1),),
                           catalog=catalog, reason="Mask interference within evidence support")
        history.preview(state, "Stage mask")
        history.apply()
        session.save_session(self.session_path, catalog, history)
        verified = self.verify_paired_fixture()
        self.assertEqual(verified.returncode, 0, verified.stderr)
        row = self.paired_row(verified)
        self.assertIn("decision accepted", row)
        self.assertIn("raw_samples_match", row)
        self.assertIn("attestation stale", row)
        self.assertIn("ready false", row)

    def test_unknown_rms_method_cannot_claim_absent_historical_provenance(self):
        self.apply(evidence=False)
        session.save_session(self.session_path, self.catalog, self.history)
        document = json.loads(self.session_path.read_bytes())
        rms = document["revisions"][1]["state"]["rms"]
        rms["method_version"] = "future-rms-v99"
        rms["baseline"]["units"] = "raw counts"
        self.session_path.write_text(json.dumps(document), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "baseline provenance"):
            session.load_session(self.session_path)
