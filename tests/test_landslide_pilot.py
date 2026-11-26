"""Exercise the relocatable, blinded landslide comparison pack."""

import csv
from hashlib import sha256
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from obspy import Stream, Trace, UTCDateTime, read
from pyrocko.gui.snuffler.marker import Marker

from tools import landslide_pilot


class LandslidePilotTest(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.epoch = UTCDateTime("2022-09-15T05:00:00Z")
        self.source = self.root / "source.mseed"
        traces = [
            Trace(
                data=np.sin(np.arange(10_000) / 10).astype(np.float64),
                header={
                    "network": "XX", "station": "TEST", "channel": "BHZ",
                    "sampling_rate": 20, "starttime": self.epoch,
                },
            ),
            Trace(
                data=np.arange(100, dtype=np.float64),
                header={
                    "network": "XX", "station": "SHORT", "channel": "BHZ",
                    "sampling_rate": 20, "starttime": self.epoch + 230,
                },
            ),
        ]
        Stream(traces).write(str(self.source), format="MSEED")
        self.manifest = self.root / "case.json"
        self.manifest.write_text(json.dumps({
            "case_id": "synthetic-control",
            "sources": [{"path": self.source.name, "sha256": sha256(self.source.read_bytes()).hexdigest()}],
            "channels": ["XX.TEST..BHZ", "XX.SHORT..BHZ", "XX.ABSNT..BHZ"],
            "start_utc": str(self.epoch + 50),
            "stop_utc": str(self.epoch + 450),
            "provenance": {"fixture": "simulated"},
        }), encoding="utf-8")
        self.pack = self.root / "pack"

    def test_preparation_writes_named_supported_waveforms_and_blind_table(self):
        landslide_pilot.prepare(self.manifest, self.pack)
        with (self.pack / "review.csv").open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        table = {(row["channel"], row["band"]): row for row in rows}
        self.assertEqual(len(table), 6)
        self.assertEqual(table["XX.ABSNT..BHZ", "LP"]["onset_status"], "unavailable")
        self.assertEqual(table["XX.SHORT..BHZ", "LP"]["end_status"], "insufficient_support")
        self.assertEqual(table["XX.TEST..BHZ", "LP"]["end_status"], "unreviewed")
        self.assertEqual(table["XX.TEST..BHZ", "LP"]["end_reason"], "")

        prepared = read(str(self.pack / "snuffler.mseed"))
        by_id = {trace.id: trace for trace in prepared}
        row = table["XX.TEST..BHZ", "LP"]
        self.assertEqual(by_id[row["snuffler_nslc"]].stats.starttime, self.epoch + 200)
        self.assertEqual(by_id[row["snuffler_nslc"]].stats.npts, 2000)
        self.assertNotIn(table["XX.SHORT..BHZ", "LP"]["snuffler_nslc"], by_id)
        self.assertEqual(sha256(self.source.read_bytes()).hexdigest(),
                         json.loads(self.manifest.read_text())["sources"][0]["sha256"])

    def test_score_snaps_unique_samples_and_reopens_after_pack_relocation(self):
        landslide_pilot.prepare(self.manifest, self.pack)
        table = self.root / "completed.csv"
        marker_file = self.root / "completed.markers"
        with (self.pack / "review.csv").open(newline="", encoding="utf-8") as file:
            contents = csv.DictReader(file)
            rows = list(contents)
            fields = contents.fieldnames
        row = next(row for row in rows if row["channel"] == "XX.TEST..BHZ" and row["band"] == "LP")
        row.update(onset_status="identifiable", end_status="identifiable",
                   onset_reason="sustained increase", end_reason="return to background")
        partial = next(row for row in rows if row["channel"] == "XX.TEST..BHZ" and row["band"] == "HF")
        partial.update(onset_status="ambiguous", onset_reason="earlier burst",
                       end_status="identifiable", end_reason="clear decay")
        with table.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        markers = [Marker([tuple(record["snuffler_nslc"].split("."))],
                          float(self.epoch + seconds), float(self.epoch + seconds), kind=kind)
                   for record, seconds, kind in [(row, 225.012, 1), (row, 230.008, 2),
                                                  (partial, 231.013, 2)]]
        Marker.save_markers(markers, str(marker_file), fdigits=9)

        result = self.root / "baseline.json"
        landslide_pilot.score(self.pack, table, marker_file, result)
        relocated = self.root / "relocated with spaces"
        shutil.move(self.pack, relocated)
        self.source.unlink()
        worksheet = landslide_pilot.reopen(relocated, result)
        result = worksheet.result("XX.TEST..BHZ", "LP")
        self.assertEqual(result.onset.sample.sample_index, 4500)
        self.assertEqual(result.end.sample.sample_index, 4600)
        self.assertEqual(result.duration_seconds, 5)
        one_sided = worksheet.result("XX.TEST..BHZ", "HF")
        self.assertEqual(one_sided.onset.status, "ambiguous")
        self.assertEqual(one_sided.end.sample.sample_index, 4620)
        self.assertIsNone(one_sided.duration_seconds)
        payload = json.loads((self.root / "baseline.json").read_text())
        baseline = next(row for row in payload["evidence"]["baseline"] if row["duration_status"] == "eligible")
        tampered = self.root / "tampered.json"
        baseline["observed_signal_duration_seconds"] = 5.5
        tampered.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Baseline evidence does not reproduce"):
            landslide_pilot.reopen(relocated, tampered)
        baseline["observed_signal_duration_seconds"] = 5
        baseline["marker_utc"]["onset"] = "2022-09-15 05:03:46.000000000"
        tampered.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Stored marker time does not reproduce"):
            landslide_pilot.reopen(relocated, tampered)
        payload["method"] = "Qt worksheet"
        tampered.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Evidence fields do not match the review method"):
            landslide_pilot.reopen(relocated, tampered)


    def test_rejects_midpoint_raw_track_and_source_overwrite(self):
        landslide_pilot.prepare(self.manifest, self.pack)
        table = self.pack / "review.csv"
        markers = self.root / "invalid.markers"
        output = self.root / "refused.json"
        for trace_id, seconds, digits in [
            ("XX.TEST.LP.BHZ", 225.025, 9), ("XX.TEST.LP.BHZ", 225, 1), ("XX.TEST.RW.BHZ", 225, 9),
        ]:
            with self.subTest(trace_id=trace_id, digits=digits):
                marker = Marker([tuple(trace_id.split("."))],
                                float(self.epoch + seconds), float(self.epoch + seconds), kind=1)
                Marker.save_markers([marker], str(markers), fdigits=digits)
                with self.assertRaisesRegex(ValueError, "supported sample|prepared band trace"):
                    landslide_pilot.score(self.pack, table, markers, output)
                self.assertFalse(output.exists())

        Marker.save_markers([], str(markers))
        source_bytes = self.source.read_bytes()
        with self.assertRaises(FileExistsError):
            landslide_pilot.score(self.pack, table, markers, self.source)
        self.assertEqual(self.source.read_bytes(), source_bytes)

    def test_reveal_requires_independent_review_then_preserves_source_strings(self):
        landslide_pilot.prepare(self.manifest, self.pack)
        table = self.pack / "review.csv"
        markers = self.root / "blank.markers"
        Marker.save_markers([], str(markers))
        initial = self.root / "initial.json"
        landslide_pilot.score(self.pack, table, markers, initial)
        published = self.root / "reference.csv"
        published.write_text(
            "Network,Station,LocationCode,Channel,detect_HF,detect_LP,"
            "starttimeHF_UTC,endtimeHF_UTC,starttimeLP_UTC,endtimeLP_UTC\n"
            "XX,TEST,  ,BHZ,1,0,2022-09-15T05:03:45.019Z,2022-09-15T05:03:50.013Z,,\n",
            encoding="utf-8",
        )
        output = self.root / "reveal.json"
        with self.assertRaisesRegex(ValueError, "Finish the independent review"):
            landslide_pilot.reveal(self.pack, initial, published, output)
        self.assertFalse(output.exists())

        completed_table = self.root / "complete.csv"
        with table.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            records = list(reader)
            fields = reader.fieldnames
        for record in records:
            for boundary in ("onset", "end"):
                if record[f"{boundary}_status"] == "unreviewed":
                    record[f"{boundary}_status"] = "ambiguous"
                    record[f"{boundary}_reason"] = "No single defensible boundary"
        with completed_table.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
        completed = self.root / "completed.json"
        landslide_pilot.score(self.pack, completed_table, markers, completed, previous=initial)
        initial_bytes = initial.read_bytes()
        landslide_pilot.reveal(self.pack, completed, published, output)
        payload = json.loads(output.read_text())
        row, = payload["published_rows"]
        self.assertEqual(row["LocationCode"], "  ")
        self.assertEqual(row["starttimeHF_UTC"], "2022-09-15T05:03:45.019Z")
        self.assertEqual(initial.read_bytes(), initial_bytes)

    def test_qt_preview_apply_undo_and_save_valid_replay(self):
        from PyQt5 import QtCore, QtTest, QtWidgets

        from tools.landslide_worksheet import LandslideWorksheet

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        landslide_pilot.prepare(self.manifest, self.pack)
        output = self.root / "qt review.json"
        pack = landslide_pilot.open_pack(self.pack)
        window = LandslideWorksheet(pack, output)
        self.addCleanup(window.close)
        window.resize(1240, 730)
        window.show()
        row_index = next(index for index, row in enumerate(pack.rows)
                         if row.channel == "XX.TEST..BHZ" and row.band == "LP")
        window.table.selectRow(row_index)
        app.processEvents()
        self.assertEqual(window.table.horizontalScrollBar().maximum(), 0)
        axis = window.axes["LP"]

        def click_at(elapsed):
            axis.set_xlim(elapsed - 2, elapsed + 2)
            window.canvas.draw()
            x, y = axis.transData.transform((elapsed, 0))
            QtTest.QTest.mouseClick(window.canvas, QtCore.Qt.LeftButton,
                                   pos=QtCore.QPoint(round(x), round(window.canvas.height() - y)))

        click_at(175)
        self.assertIsNone(window.worksheet.result("XX.TEST..BHZ", "LP").onset.sample)
        window.reason.setText("clear sustained increase")
        window.apply_button.click()
        self.assertEqual(window.worksheet.result("XX.TEST..BHZ", "LP").onset.sample.sample_index, 4500)
        window.later_button.click()
        click_at(50)  # LP has no processing support in this part of the review window.
        self.assertIsNone(window.pending_sample)
        window.reason.setText("failed re-pick")
        window.apply_button.click()
        self.assertEqual(window.worksheet.result("XX.TEST..BHZ", "LP").onset.sample.sample_index, 4500)
        click_at(175)
        window.later_button.click()
        window.reason.setText("inspect neighboring sample")
        window.apply_button.click()
        self.assertEqual(window.worksheet.result("XX.TEST..BHZ", "LP").onset.sample.sample_index, 4501)

        window.boundary.setCurrentText("End")
        window.status.setCurrentText("ambiguous")
        window.reason.setText("multiple plausible endings")
        window.apply_button.click()
        self.assertEqual(window.worksheet.result("XX.TEST..BHZ", "LP").end.status, "ambiguous")
        window.undo_button.click()
        self.assertEqual(window.worksheet.result("XX.TEST..BHZ", "LP").end.status, "unreviewed")
        window.save_button.click()
        replay = landslide_pilot.reopen(self.pack, output)
        self.assertEqual(replay.to_dict(), window.worksheet.to_dict())


if __name__ == "__main__":
    unittest.main()
