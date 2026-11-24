"""Exercise the synthetic review through the real Qt controls and Snuffler view."""

import csv
import importlib.util
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, skipUnless
from unittest.mock import patch


HAS_DESKTOP = all(importlib.util.find_spec(name) for name in ("pyrocko", "PyQt5"))


@skipUnless(HAS_DESKTOP, "Install SeisSlate with the desktop extra")
class WorkbenchTest(TestCase):
    def test_apply_mask_undo_and_reopen(self):
        from PyQt5.QtCore import QPoint, Qt
        from PyQt5.QtTest import QTest
        from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox

        from seisslate.ui import launch_workbench
        from seisslate.example import load_example

        catalog, history = load_example()

        def exercise(workbench):
            viewer = workbench.viewer
            QApplication.processEvents()

            def click(name):
                button = workbench.buttons[name]
                self.assertTrue(button.isEnabled(), name)
                QTest.mouseClick(button, Qt.LeftButton)
                QApplication.processEvents()

            canvas = workbench.canvas
            canvas.draw()
            row = next(row for row in workbench.result.reviewed.observations if row.blow_id == "B03")
            x, y = workbench.axes.transData.transform((row.lag_ns / 1e6, row.dx_m))
            # Convert Matplotlib's physical pixels to Qt's logical mouse coordinates.
            ratio = canvas.device_pixel_ratio
            height = canvas.get_width_height()[1]
            point = QPoint(round(x / ratio), round(height - y / ratio))
            QTest.mouseClick(canvas, Qt.LeftButton, pos=point)
            self.assertEqual(workbench.selected_blow_id, "B03")
            self.assertTrue(workbench.table.item(2, 0).isSelected())

            first_row = workbench.table.visualItemRect(workbench.table.item(0, 0)).center()
            QTest.mouseClick(workbench.table.viewport(), Qt.LeftButton, pos=first_row)
            QApplication.processEvents()
            self.assertEqual(workbench.selected_blow_id, "B01")
            low, high = viewer.get_time_range()
            self.assertLess(high - low, 0.1)
            self.assertLess(low, 946684800.360)
            self.assertGreater(high, 946684800.400)
            baseline_slope = workbench.result.reviewed.slope_m_per_s
            click("later")
            self.assertIsNotNone(history.draft)
            self.assertFalse(workbench.export_action.isEnabled())
            self.assertEqual(workbench.result.reviewed.slope_m_per_s, baseline_slope)
            click("cancel")
            self.assertIsNone(history.draft)
            self.assertEqual(history.current.state.blows[0].original.reviewed_ref.sample_index, 360)

            click("later")
            click("apply")
            moved = history.current.state.blows[0]
            nudged_slope = workbench.result.reviewed.slope_m_per_s
            self.assertEqual(moved.original.reviewed_ref.sample_index, 361)
            self.assertEqual(moved.calibration.reviewed_ref.sample_index, 400)
            self.assertNotEqual(nudged_slope, baseline_slope)
            self.assertEqual(workbench.table.item(0, 2).text(), "-39.000")

            click("later")
            click("apply")
            self.assertEqual(history.current.state.blows[0].original.reviewed_ref.sample_index, 362)
            click("undo")
            self.assertEqual(history.current.state.blows[0], moved)

            time_s = catalog.time_ns(moved.original.reviewed_ref) / 1e9
            nslc = tuple(moved.original.reviewed_ref.channel.split("."))
            track = next(index for index in range(2) if nslc in viewer.get_nslc_ids_for_track(index))
            y = round(viewer.height() * (track + 0.5) / 2)

            point_time = time_s + 0.0014
            point = QPoint(round((point_time - low) * viewer.width() / (high - low)), y)
            QTest.mouseMove(viewer, point)
            QTest.mouseDClick(viewer, Qt.LeftButton, pos=point)
            QTest.mouseRelease(viewer, Qt.LeftButton, pos=point)
            QTest.mouseClick(viewer, Qt.LeftButton, pos=point)
            with patch.object(QMessageBox, "question", return_value=QMessageBox.Ok) as confirm:
                click("point")
            confirm.assert_called_once()
            self.assertEqual(history.draft.state.blows[0].original.reviewed_ref.sample_index, 362)
            click("cancel")

            start = QPoint(round((time_s - 0.00035 - low) * viewer.width() / (high - low)), y)
            stop = QPoint(round((time_s + 0.00035 - low) * viewer.width() / (high - low)), y)
            QTest.mouseMove(viewer, start)
            QTest.mouseDClick(viewer, Qt.LeftButton, pos=start)
            QTest.mouseRelease(viewer, Qt.LeftButton, pos=start)
            QTest.mousePress(viewer, Qt.LeftButton, pos=start)
            QTest.mouseMove(viewer, stop)
            QTest.mouseRelease(viewer, Qt.LeftButton, pos=stop)
            interval, = workbench.overlay.get_selected_markers()
            self.assertEqual(list(interval.get_nslc_ids()), [nslc])
            self.assertLess(interval.get_tmin(), time_s)
            self.assertGreater(interval.get_tmax(), time_s)
            workbench.role.setCurrentIndex(workbench.role.findData("calibration"))
            click("mask")
            self.assertIsNone(history.draft)
            self.assertIn("Calibration", workbench.status.text())
            workbench.role.setCurrentIndex(workbench.role.findData("original"))
            click("mask")
            preview = list(workbench.managed_markers)
            self.assertTrue(any(marker.kind == 4 and marker.get_tmin() < marker.get_tmax() for marker in preview))
            click("apply")
            self.assertNotIn("B01", workbench.result.reviewed.blow_ids)
            self.assertEqual(history.current.state.blows[0].calibration, moved.calibration)
            click("undo")
            self.assertEqual(workbench.result.reviewed.slope_m_per_s, nudged_slope)
            self.assertIn("B01", workbench.result.reviewed.blow_ids)

            with TemporaryDirectory() as destination:
                path = Path(destination) / "Saved review é.json"
                with patch.object(QFileDialog, "getSaveFileName", return_value=(str(path), "")):
                    workbench.save_action.trigger()
                self.assertTrue(path.is_file(), workbench.status.text())
                with patch.object(QFileDialog, "getOpenFileName", return_value=(str(path), "")):
                    workbench.open_session_action.trigger()
                self.assertIsNot(workbench.history, history)
                self.assertEqual(workbench.history.current.state, history.current.state)
                self.assertEqual(workbench.result.reviewed.slope_m_per_s, nudged_slope)
                self.assertEqual(workbench.selected_blow_id, "B01")
                self.assertTrue(workbench.table.item(0, 0).isSelected())
                self.assertTrue(workbench.buttons["later"].isEnabled())
                self.assertTrue(workbench.buttons["undo"].isEnabled())
                self.assertTrue(workbench.buttons["redo"].isEnabled())
                exported = Path(destination) / "Applied review é.csv"
                with patch.object(QFileDialog, "getSaveFileName", return_value=(str(exported), "")):
                    workbench.export_action.trigger()
                with exported.open(newline="", encoding="utf-8") as table:
                    row = next(csv.DictReader(table))
                self.assertEqual(row["revision_id"], str(history.current.revision_id))
                self.assertEqual(row["lag_ns"], "-39000000")
                self.assertEqual(row["original_sample_index"], "361")

            screenshot = os.environ.get("SEISSLATE_TEST_SCREENSHOT")
            if screenshot:
                self.assertTrue(workbench.window.grab().save(screenshot))

        launch_workbench((catalog, history), check=exercise)

    def test_empty_window_exposes_open_actions(self):
        from seisslate.ui import launch_workbench

        def exercise(workbench):
            self.assertTrue(workbench.open_pair_action.isEnabled())
            self.assertTrue(workbench.open_session_action.isEnabled())
            self.assertFalse(workbench.save_action.isEnabled())
            self.assertFalse(workbench.buttons["later"].isEnabled())
            self.assertEqual(workbench.table.rowCount(), 0)
            self.assertFalse(workbench.viewer.acceptDrops())
            file_menu = next(action.menu() for action in workbench.window.menuBar().actions()
                             if action.text() == "&File")
            for action in file_menu.actions():
                if action.text().startswith("Open waveform "):
                    self.assertFalse(action.isEnabled(), action.text())

        launch_workbench(check=exercise)
