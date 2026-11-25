"""Native single-channel review through the Qt workbench."""

import csv
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, skipUnless
from unittest.mock import patch

import numpy as np


HAS_DESKTOP = all(find_spec(name) is not None for name in ("pyrocko", "PyQt5", "matplotlib"))


@skipUnless(HAS_DESKTOP, "Install SeisSlate with the desktop extra")
class RmsWorkbenchTest(TestCase):
    def test_open_select_preview_apply_mask_undo_and_reopen(self):
        from PyQt5.QtCore import QPoint, Qt
        from PyQt5.QtTest import QTest
        from PyQt5.QtWidgets import QApplication, QFileDialog, QInputDialog, QMessageBox
        from pyrocko import io
        from pyrocko.gui.snuffler.marker import Marker
        from pyrocko.trace import Trace

        from seisslate.ui import launch_workbench

        with TemporaryDirectory() as directory:
            wave = Path(directory) / "single channel.mseed"
            session = Path(directory) / "RMS review é.json"
            origin = 946684800
            nslc = ("XX", "RAW", "", "EHZ")
            io.save([Trace(*nslc, tmin=origin, deltat=0.01,
                           ydata=np.array([1, 5, 7, 9, 11, 13, 30, 34], dtype=np.int32)),
                     Trace("XX", "RAW", "", "EHN", tmin=origin, deltat=0.01,
                           ydata=np.arange(8, dtype=np.int32))], str(wave))

            def exercise(wb):
                action = getattr(wb, "open_single_action", None)
                self.assertIsNotNone(action, "The desktop needs an Open single channel action.")
                with patch.object(QFileDialog, "getOpenFileName", return_value=(str(wave), "")):
                    with patch.object(QInputDialog, "getItem", return_value=("XX.RAW..EHZ", True)):
                        action.trigger()
                QApplication.processEvents()
                self.assertEqual(wb.state().mode, "single-channel")
                self.assertEqual([s.channel for s in wb.segments], ["XX.RAW..EHZ"])
                self.assertFalse(wb.buttons["earlier"].isEnabled())
                self.assertFalse(wb.buttons["import_groups"].isEnabled())
                rms = wb.rms_panel

                def press(button):
                    self.assertTrue(button.isEnabled(), button.text())
                    button.setFocus()
                    QTest.keyClick(button, Qt.Key_Space)
                    QApplication.processEvents()

                def button(name):
                    press(wb.buttons[name])

                def marker(start, stop):
                    marker = Marker([nslc], origin + start, origin + stop)
                    wb.viewer.add_marker(marker)
                    marker.selected = True
                    return marker

                def disposition(value, reason):
                    rms.disposition.setCurrentIndex(rms.disposition.findData(value))
                    rms.reviewer.setText("Analyst")
                    wb.reason.setText(reason)

                base = wb.history.current
                # Snuffler uses Qt logical pixels at both scale factors.
                viewer = wb.viewer
                viewer.set_time_range(origin - 0.01, origin + 0.09)
                QApplication.processEvents()
                low, high = viewer.get_time_range()
                track = next(index for index in range(2) if nslc in viewer.get_nslc_ids_for_track(index))
                y = round(viewer.height() * (track + 0.5) / 2)
                start = QPoint(round((origin + 0.001 - low) * viewer.width() / (high - low)), y)
                stop = QPoint(round((origin + 0.019 - low) * viewer.width() / (high - low)), y)
                QTest.mouseMove(viewer, start)
                QTest.mouseDClick(viewer, Qt.LeftButton, pos=start)
                QTest.mouseRelease(viewer, Qt.LeftButton, pos=start)
                QTest.mousePress(viewer, Qt.LeftButton, pos=start)
                QTest.mouseMove(viewer, stop)
                QTest.mouseRelease(viewer, Qt.LeftButton, pos=stop)
                press(rms.event.use_marker)
                self.assertEqual((rms.event.start.text(), rms.event.stop.text()), ("0", "2"))
                self.assertEqual(wb.history.current, base)
                rms.background.start.setFocus()
                QTest.keyClicks(rms.background.start, "6")
                rms.background.stop.setFocus()
                QTest.keyClicks(rms.background.stop, "8")
                disposition("rejected", "Known incidental traffic")
                press(rms.preview_button)
                self.assertIn("Unavailable", rms.applied.text())
                self.assertIn("Contrast 1", rms.preview.text())
                self.assertIn("[6, 8)", rms.preview.text())
                self.assertIn("rejected", rms.preview.text())
                self.assertTrue(wb.buttons["apply"].isEnabled())
                rms.background.stop.setText("7")
                self.assertEqual(rms.disposition.currentData(), "candidate")
                self.assertFalse(wb.buttons["apply"].isEnabled())
                rms.background.stop.setText("8")
                disposition("rejected", "Known incidental traffic")
                press(rms.preview_button)
                button("apply")
                rejected = wb.history.current
                self.assertIn("rejected", rms.applied.text())
                self.assertIn("Contrast 1", rms.applied.text())
                self.assertIn("raw units unknown", rms.applied.text())
                self.assertIn("Raw", wb.source_label.text())
                self.assertEqual({mark.get_label() for mark in wb.managed_markers}, {"Event E", "Background B"})

                # A shorter exact span remains a distinct preview and does not alter the applied result.
                rms.background.stop.setText("7")
                disposition("accepted", "Analyst confirms half-open intent")
                press(rms.preview_button)
                self.assertEqual(wb.history.current, rejected)
                self.assertIn("Contrast 1", rms.applied.text())
                self.assertIn("Unequal window lengths", rms.preview.text())
                self.assertIn("accepted", rms.preview.text())
                button("cancel")
                self.assertEqual(rms.background.stop.text(), "8")

                rms.background.stop.setText("7")
                disposition("accepted", "Analyst confirms half-open intent")
                press(rms.preview_button)
                button("apply")
                accepted = wb.history.current
                self.assertIn("accepted", rms.applied.text())
                self.assertIn("Unavailable", rms.applied.text())
                self.assertIn("Unequal window lengths", rms.applied.text())
                button("undo")
                self.assertEqual(wb.history.current, rejected)
                button("redo")
                self.assertEqual(wb.history.current, accepted)

                marker(0.0, 0.01)
                wb.reason.setText("Impulse contamination")
                press(rms.mask_button)
                button("apply")
                self.assertIn("candidate", rms.applied.text())
                self.assertIn("Masked samples", rms.applied.text())
                self.assertIn("Mask", {mark.get_label() for mark in wb.managed_markers})
                np.testing.assert_array_equal(wb.segments[0].data, [1, 5, 7, 9, 11, 13, 30, 34])
                marker(0.0, 0.01)
                wb.reason.setText("Contamination resolved")
                press(rms.restore_button)
                button("apply")
                self.assertNotIn("Masked samples", rms.applied.text())
                self.assertFalse(wb.state().masks)
                button("undo")
                self.assertIn("Masked samples", rms.applied.text())
                button("undo")
                self.assertEqual(wb.history.current, accepted)

                with patch.object(QFileDialog, "getSaveFileName", return_value=(str(session), "")):
                    wb.save_action.trigger()
                self.assertTrue(session.is_file(), wb.status.text())
                with patch.object(QFileDialog, "getOpenFileName", return_value=(str(session), "")):
                    wb.open_session_action.trigger()
                self.assertEqual(wb.history.current, accepted)
                self.assertIn(rejected, wb.history.revisions)
                self.assertIn("accepted", rms.applied.text())
                self.assertIn("Unavailable", rms.applied.text())
                self.assertTrue(wb.export_rms_action.isEnabled())
                self.assertFalse(wb.export_evidence_action.isEnabled())
                wb.window.resize(1240, 900)
                QApplication.processEvents()
                self.assertEqual(rms.horizontalScrollBar().maximum(), 0)
                self.assertIn("Replay numeric match", rms.provenance.toPlainText())
                self.assertIn("SHA-256", rms.provenance.toPlainText())

                # A retained out-of-coverage waveform selection gets a reason, not a clipped span.
                marker(-0.002, 0.019)
                press(rms.event.use_marker)
                self.assertEqual(rms.event.start.text(), "")
                press(rms.preview_button)
                self.assertIn("outside recording coverage", rms.preview.text())
                clear = getattr(rms.event, "clear_button", None)
                self.assertIsNotNone(clear, "A retained marker intent with blank indices needs a clear action.")
                press(clear)
                press(rms.preview_button)
                self.assertIsNone(wb.history.draft.state.rms.windows.event)
                button("cancel")
                self.assertIsNone(wb.history.draft)

                rms.background.stop.setText("8")
                with patch.object(QMessageBox, "question", return_value=QMessageBox.Cancel) as switch:
                    wb.example_action.trigger()
                switch.assert_called_once()
                self.assertEqual(wb.state().mode, "single-channel")
                self.assertFalse(wb.save_action.isEnabled())
                button("cancel")

                rms.background.stop.setText("8")
                disposition("accepted", "Reviewed both equal spans")
                press(rms.preview_button)
                button("apply")
                self.assertIn("Eligible for evidence use yes", rms.applied.text())

                def export_selected(filename):
                    target = Path(directory) / filename
                    with patch.object(QFileDialog, "getSaveFileName", return_value=(str(target), "")):
                        wb.export_rms_action.trigger()
                    self.assertTrue(target.is_file(), wb.status.text())
                    with target.open(encoding="utf-8", newline="") as stream:
                        row, = csv.DictReader(stream)
                    return row

                latest = export_selected("accepted.csv")
                self.assertEqual((latest["disposition"], latest["ready"]), ("accepted", "true"))
                button("undo")
                unavailable = export_selected("unavailable.csv")
                self.assertEqual((unavailable["disposition"], unavailable["ready"]), ("accepted", "false"))
                self.assertEqual(unavailable["current_contrast"], "")
                button("undo")
                historical = export_selected("rejected.csv")
                self.assertEqual(historical["revision_id"], str(rejected.revision_id))
                self.assertEqual((historical["disposition"], historical["ready"]), ("rejected", "false"))
                self.assertEqual(float(historical["current_contrast"]), 1)
                button("redo")
                button("redo")
                with patch.object(QFileDialog, "getSaveFileName", return_value=(str(session), "")):
                    wb.save_action.trigger()
                wb.example_action.trigger()
                self.assertEqual(wb.state().mode, "paired")
                self.assertTrue(wb.buttons["later"].isEnabled())
                self.assertFalse(wb.export_rms_action.isVisible())
                self.assertTrue(wb.export_evidence_action.isVisible())

            launch_workbench(check=exercise)
