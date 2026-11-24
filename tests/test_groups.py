import hashlib
import json
import unittest
from dataclasses import replace
from os import fsync
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from seisslate.analysis import analyze
from seisslate.example import load_example
from seisslate.review import Receiver, ReviewState, nudge_pick
from seisslate.session import export_csv, load_session, save_session
from seisslate.sources import Segment, Source, SourceCatalog


HEADER = "group_id,start_utc,end_utc,duration_s\n"
ROW = "G1,1970-01-01T00:00:00Z,,8\n"


class GroupTest(unittest.TestCase):
    def setUp(self):
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.path = Path(folder.name) / "Field log.csv"

    def read_csv(self, content=HEADER + ROW):
        from seisslate.groups import load_group_csv

        self.path.write_bytes(content.encode("utf-8"))
        return load_group_csv(self.path)

    def test_preserves_exact_bounds_raw_values_and_csv_locator(self):
        content = (
            "\ufeffgroup_id,start_utc,duration_s,time_basis,source_position_id,"
            "original_channel,calibration_channel,timing_note,geometry_note\r\n"
            'G1, 1970-01-01T00:00:00.000000007Z ,0.000000009,logger UTC,P4,O,C,"clock\r\n'
            'note",survey note\r\n'
            "G2,1970-01-01T00:00:00+00:00,0.250,,,,,,\r\n"
        )
        first, second = self.read_csv(content)
        self.assertEqual((first.group_id, first.start_ns, first.stop_ns), ("G1", 7, 16))
        self.assertEqual(first.start_utc, " 1970-01-01T00:00:00.000000007Z ")
        self.assertEqual((first.end_utc, first.duration_s, first.time_basis), (None, "0.000000009", "logger UTC"))
        self.assertEqual((first.source_position_id, first.original_channel, first.calibration_channel), ("P4", "O", "C"))
        self.assertEqual((first.timing_note, first.geometry_note), ("clock\r\nnote", "survey note"))
        self.assertEqual((first.source_file, first.source_sha256, first.source_row), (
            "Field log.csv", hashlib.sha256(content.encode("utf-8")).hexdigest(), 2,
        ))
        self.assertEqual((second.source_row, second.time_basis, second.reviewed), (4, "UTC", False))

    def test_duration_and_end_can_share_headers_with_one_value_per_row(self):
        first, second = self.read_csv(HEADER + ROW + "G2,1969-12-31T23:59:59.99Z,1970-01-01T00:00:00Z,\n")
        self.assertEqual((first.start_ns, first.stop_ns), (0, 8_000_000_000))
        self.assertEqual((second.start_ns, second.stop_ns), (-10_000_000, 0))
        self.assertEqual(second.end_utc, "1970-01-01T00:00:00Z")
        self.assertIsNone(second.duration_s)

    def test_invalid_rows_report_the_row_without_returning_a_partial_batch(self):
        for bad_row in (
            "G1,1970-01-01T00:00:01Z,,1\n",  # duplicate
            ",1970-01-01T00:00:01Z,,1\n",
            "G2,1970-01-01T00:00:01,,1\n",
            "G2,1970-02-31T00:00:01Z,,1\n",
            "G2,1970-01-01T00:00:01Z,1970-01-01T00:00:01Z,\n",
            "G2,1970-01-01T00:00:01Z,1970-01-01T00:00:03Z,1\n",
            "G2,1970-01-01T00:00:01Z,,0\n",
            "G2,1970-01-01T00:00:01Z,,-1\n",
            "G2,1970-01-01T00:00:01Z,,0.0000000001\n",
            "G2,1970-01-01T00:00:01Z,,NaN\n",
            "G2,1970-01-01T00:00:01Z,,1e10000\n",
            "G2,9999-12-31T23:59:59Z,,1\n",
            "G2,1970-01-01T00:00:01Z,,1000000000000\n",
            "G2,1970-01-01T00:00:01Z,,1,unexpected\n",
            'G2,1970-01-01T00:00:01Z,,"1\n',
        ):
            with self.subTest(row=bad_row), self.assertRaisesRegex(ValueError, r"[Ll]ine 3"):
                self.read_csv(HEADER + ROW + bad_row)

    def test_malformed_header_and_empty_csv_are_rejected(self):
        for content in ("", "group_id,start_utc\n", HEADER,
                        "group_id,start_utc,duration_s,typo\n", "group_id,group_id,start_utc,duration_s\n"):
            with self.subTest(content=content), self.assertRaisesRegex(ValueError, r"[Ll]ine 1"):
                self.read_csv(content)

    def test_invalid_utf8_after_bom_uses_physical_line_number(self):
        from seisslate.groups import load_group_csv

        self.path.write_bytes(("\ufeff" + HEADER + ROW).encode("utf-8") + b"\xff")
        with self.assertRaisesRegex(ValueError, r"[Ll]ine 3:.*UTF-8"):
            load_group_csv(self.path)

    def test_import_stages_only_group_metadata_and_is_undoable(self):
        from seisslate.review import add_groups, set_group_reviewed

        catalog, history = load_example()
        baseline = history.current.state
        before_fit = analyze(catalog, baseline, baseline)
        groups = self.read_csv()
        state = add_groups(baseline, groups)
        history.preview(state, "Import group metadata")
        self.assertEqual(history.current.state, baseline)
        history.cancel()
        self.assertEqual(history.current.state, baseline)
        history.preview(state, "Import group metadata")
        history.apply()
        self.assertEqual(history.current.state, replace(baseline, groups=groups))
        self.assertEqual(analyze(catalog, baseline, state), before_fit)
        self.assertEqual(history.undo().state, baseline)
        self.assertEqual(history.redo().state, state)

        duplicate_batch = self.read_csv(HEADER + ROW.replace("G1", "G2") + ROW)
        with self.assertRaisesRegex(ValueError, r"[Ll]ine 3.*G1"):
            add_groups(history.current.state, duplicate_batch)
        self.assertEqual(history.current.state, state)
        changed = set_group_reviewed(state, "G1", True)
        history.preview(changed, "Mark reviewed")
        history.apply()
        self.assertTrue(history.current.state.groups[0].reviewed)
        self.assertFalse(history.undo().state.groups[0].reviewed)

    def test_recorded_coverage_preserves_gaps_overlap_and_receiver_shift(self):
        from seisslate.groups import inspect_group

        def segment(source_hash, channel, segment_id, start_ns, rate, size):
            return Segment(source_hash, channel, segment_id, start_ns, rate, np.zeros(size))

        original = Source(Path("original"), "a", 1, (
            segment("a", "XX.ORI..EHZ", "a1", 0, 2, 4),
            segment("a", "XX.ORI..EHZ", "a2", 2_500_000_000, 2, 4),
            segment("a", "XX.ORI..EHZ", "a3", 4_250_000_000, 2, 2),
        ))
        calibration = Source(Path("calibration"), "b", 1, (
            segment("b", "XX.CAL..EHZ", "b1", 100_000_000, 2, 16),
        ))
        catalog = SourceCatalog((original, calibration))
        state = ReviewState((), receivers=(
            Receiver("original", "XX.ORI..EHZ", "a"),
            Receiver("calibration", "XX.CAL..EHZ", "b"),
        ))
        group, gap, full = self.read_csv(
            "group_id,start_utc,duration_s,original_channel,calibration_channel,timing_note\n"
            "G1,1970-01-01T00:00:00Z,8,XX.ORI..EHZ,Alias,field log\n"
            "gap,1970-01-01T00:00:02Z,0.5,,,\n"
            "full,1970-01-01T00:00:00.1Z,1.9,,,\n"
        )
        result = inspect_group(catalog, state, group)
        self.assertEqual((result.original.channel_id, result.original.sha256), ("XX.ORI..EHZ", "a"))
        self.assertEqual((result.original.coverage, result.original.channel_status), ("conflict", "match"))
        self.assertEqual(result.original.intervals, ((0, 2_000_000_000), (2_500_000_000, 5_250_000_000)))
        self.assertEqual((result.calibration.channel_id, result.calibration.sha256), ("XX.CAL..EHZ", "b"))
        self.assertEqual((result.calibration.coverage, result.calibration.channel_status), ("partial", "conflict"))
        self.assertEqual(result.calibration.intervals, ((100_000_000, 8_000_000_000),))
        self.assertEqual((result.timing_status, result.geometry_status), ("unverified", "missing"))
        self.assertEqual(inspect_group(catalog, state, gap).original.coverage, "none")
        self.assertEqual(inspect_group(catalog, state, full).original.coverage, "full")

    def test_role_with_same_channel_in_two_recordings_requires_explicit_fingerprint(self):
        from seisslate.groups import inspect_group

        catalog = SourceCatalog(tuple(
            Source(Path(source_hash), source_hash, 1, (
                Segment(source_hash, "X", source_hash, 0, 1, np.zeros(8)),
            )) for source_hash in ("a", "b")
        ))
        (group,) = self.read_csv()
        state = ReviewState((), receivers=(Receiver("original", "X"),))
        result = inspect_group(catalog, state, group)
        self.assertEqual((result.original.coverage, result.original.channel_status), ("conflict", "conflict"))
        self.assertEqual(result.original.intervals, ())
        self.assertEqual((result.calibration.coverage, result.calibration.channel_status), ("none", "missing"))

    def test_legacy_session_preserves_applied_cursor_next_id_and_redo(self):
        catalog, history = load_example()
        edited = nudge_pick(history.current.state, catalog, "B01", "original", 1)
        history.preview(edited, "Nudge first onset")
        history.apply()
        second_edit = nudge_pick(edited, catalog, "B01", "calibration", 1)
        history.preview(second_edit, "Nudge second onset")
        history.apply()
        history.undo()
        path = self.path.with_suffix(".json")
        save_session(path, catalog, history)
        document = json.loads(path.read_text())
        document["schema_version"] = 1
        # Version 1 predates both fields. Every revision, including redo, used this shape.
        for revision in document["revisions"]:
            revision["state"].pop("groups", None)
            for blow in revision["state"]["blows"]:
                blow.pop("group_id", None)
        path.write_text(json.dumps(document))
        _, reopened = load_session(path)
        self.assertEqual(reopened.revisions, history.revisions)
        self.assertEqual(reopened.next_revision_id, 3)
        self.assertEqual(reopened.current.state.groups, ())
        self.assertTrue(reopened.can_undo)
        self.assertTrue(reopened.can_redo)
        self.assertEqual(reopened.undo().state, history.revisions[0].state)
        self.assertEqual(reopened.redo().state, edited)
        self.assertEqual(reopened.redo().state, second_edit)
        self.assertIsNone(reopened.current.state.blows[0].group_id)

    def test_v2_group_and_explicit_blow_link_survive_reopen_and_reject_bad_links(self):
        from seisslate.review import add_groups, set_group_reviewed

        catalog, history = load_example()
        groups = self.read_csv()
        state = add_groups(history.current.state, groups)
        history.preview(state, "Import group metadata")
        history.apply()
        linked = replace(state.blows[0], group_id="G1")
        state = set_group_reviewed(replace(state, blows=(linked, *state.blows[1:])), "G1", True)
        history.preview(state, "Link observation and review group")
        history.apply()
        path = self.path.with_suffix(".json")
        save_session(path, catalog, history)
        self.path.unlink()  # Original log is referenced by hash and locator, with no dependency on its path.
        _, reopened = load_session(path)
        self.assertEqual(reopened.current.state, state)
        self.assertEqual(reopened.undo().state.groups, groups)
        self.assertEqual(reopened.redo().state.blows[0].group_id, "G1")
        document = json.loads(path.read_text())
        self.assertEqual(document["schema_version"], 2)

        for field, bad_value in (("group_id", "missing"), ("group_id", 5)):
            with self.subTest(field=field, value=bad_value):
                document["revisions"][-1]["state"]["blows"][0][field] = bad_value
                path.write_text(json.dumps(document))
                with self.assertRaisesRegex(ValueError, r"[Gg]roup"):
                    load_session(path)

    def test_session_and_export_protect_imported_group_logs_in_redo_history(self):
        from seisslate.review import add_groups

        catalog, history = load_example()
        history.preview(add_groups(history.current.state, self.read_csv()), "Import group metadata")
        history.apply()
        history.undo()
        self.assertTrue(history.can_redo)
        contents = self.path.read_bytes()
        other = self.path.parent / "reviews" / self.path.name
        other.parent.mkdir()

        for writer in (save_session, export_csv):
            with self.subTest(writer=writer.__name__):
                self.path.write_bytes(contents)
                with self.assertRaisesRegex(ValueError, "group log"):
                    writer(self.path, catalog, history)
                self.assertEqual(self.path.read_bytes(), contents)

                other.write_bytes(b"unrelated file with the same basename")
                writer(other, catalog, history)
                self.assertNotEqual(other.read_bytes(), b"unrelated file with the same basename")

    def test_export_rechecks_for_group_log_immediately_before_replacement(self):
        from seisslate.review import add_groups

        catalog, history = load_example()
        history.preview(add_groups(history.current.state, self.read_csv()), "Import group metadata")
        history.apply()
        contents = self.path.read_bytes()
        output = self.path.with_name("renamed log.csv")
        output.write_bytes(b"unrelated file")

        def place_source_at_destination(descriptor):
            fsync(descriptor)
            output.write_bytes(contents)

        with patch("seisslate.session.os.fsync", side_effect=place_source_at_destination):
            with self.assertRaisesRegex(ValueError, "group log"):
                export_csv(output, catalog, history)
        self.assertEqual(output.read_bytes(), contents)
        self.assertEqual(set(self.path.parent.iterdir()), {self.path, output})
