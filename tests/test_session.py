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


if __name__ == "__main__":
    unittest.main()
