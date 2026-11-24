"""Exercise group review through the Qt workbench and its session tools."""

import hashlib
import importlib.util
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, skipUnless
from unittest.mock import patch


HAS_DESKTOP = all(importlib.util.find_spec(name) for name in ("pyrocko", "PyQt5"))
HEADER = (
    "group_id,start_utc,end_utc,duration_s,source_position_id,"
    "original_channel,calibration_channel,timing_note,geometry_note\n"
)
GROUP_CSV = HEADER + (
    "mismatch,2000-01-01T00:00:02.400Z,2000-01-01T00:00:02.700Z,,P08,"
    "XX.OTHER..EHZ,XX.CALB..EHZ,,\n"
    "gap,2000-01-01T00:00:02.100Z,,0.15,P07,XX.ORIG..EHZ,XX.CALB..EHZ,,"
    "Synthetic survey note <check>\n"
)


@skipUnless(HAS_DESKTOP, "Install SeisSlate with the desktop extra")
class GroupWorkbenchTest(TestCase):
    def test_group_import_cancel_apply_navigate_and_reopen(self):
        from PyQt5.QtCore import Qt
        from PyQt5.QtTest import QTest
        from PyQt5.QtWidgets import QApplication, QFileDialog, QInputDialog

        from seisslate.example import load_example
        from seisslate.ui import launch_workbench

        catalog, history = load_example()

        with TemporaryDirectory() as destination:
            root = Path(destination)
            group_csv = root / "Group review é.csv"
            group_csv.write_text(GROUP_CSV, encoding="utf-8")
            session_path = root / "Grouped review é.json"

            def exercise(workbench):
                def click(key):
                    button = workbench.buttons[key]
                    self.assertTrue(button.isVisible(), key)
                    self.assertTrue(button.isEnabled(), key)
                    QTest.mouseClick(button, Qt.LeftButton)
                    QApplication.processEvents()

                def load_csv():
                    with patch.object(QFileDialog, "getOpenFileName", return_value=(str(group_csv), "")):
                        click("import_groups")

                def assert_applied_unchanged():
                    self.assertEqual(history.current.state, initial.state)
                    self.assertEqual(workbench.result, initial_fit)
                    self.assertEqual(len(workbench.managed_markers), initial_markers)

                initial = history.current
                initial_fit = workbench.result
                initial_markers = len(workbench.managed_markers)
                self.assertEqual(workbench.review_tabs.currentIndex(), 0)
                QTest.mouseClick(workbench.review_tabs.tabBar(), Qt.LeftButton,
                                 pos=workbench.review_tabs.tabBar().tabRect(1).center())
                QApplication.processEvents()
                load_csv()
                self.assertEqual(workbench.group_table.rowCount(), 2)
                self.assertEqual(workbench.group_table.item(0, 0).text(), "gap")
                self.assertIn("Preview", workbench.group_table.item(0, 1).text())
                self.assertIn("Preview", workbench.status.text())
                self.assertEqual(history.draft.state.blows, initial.state.blows)
                assert_applied_unchanged()
                click("cancel")
                self.assertIsNone(history.draft)
                self.assertEqual(workbench.group_table.rowCount(), 0)
                assert_applied_unchanged()
                click("next_unresolved")
                self.assertIn("No unresolved hammer groups", workbench.status.text())

                load_csv()
                click("apply")
                self.assertEqual(len(history.current.state.groups), 2)
                self.assertEqual(history.current.state.blows, initial.state.blows)
                self.assertEqual(workbench.result, initial_fit)

                first = workbench.group_table.item(0, 0)
                QTest.mouseClick(workbench.group_table.viewport(), Qt.LeftButton,
                                 pos=workbench.group_table.visualItemRect(first).center())
                QApplication.processEvents()
                self.assertEqual(workbench.viewer.get_time_range(), (946684802.1, 946684802.25))
                detail = workbench.group_details.toPlainText()
                for expected in (
                    "XX.ORIG..EHZ", "XX.CALB..EHZ", "Partial", "Full", "Session clock note",
                    "Unverified", "Synthetic survey note <check>", "P07", "row 3", group_csv.name,
                    hashlib.sha256(group_csv.read_bytes()).hexdigest(), "2000-01-01T00:00:02.100Z",
                ):
                    self.assertIn(expected, detail)
                window = workbench.group_table.item(0, 2).text()
                self.assertIn("02.100000000Z", window)
                self.assertIn("02.250000000Z", window)
                header = workbench.group_table.horizontalHeader()
                self.assertLessEqual(header.sectionViewportPosition(1) + header.sectionSize(1),
                                     workbench.group_table.viewport().width())
                screenshot = os.environ.get("SEISSLATE_GROUP_TEST_SCREENSHOT")
                if screenshot:
                    self.assertTrue(workbench.window.grab().save(screenshot))

                click("review_group")
                self.assertFalse(next(g for g in history.current.state.groups if g.group_id == "gap").reviewed)
                click("apply")
                self.assertEqual(workbench.group_table.item(0, 1).text(), "Reviewed")
                self.assertEqual(workbench.result, initial_fit)
                previous = history.current
                click("next_unresolved")
                self.assertIs(history.current, previous)
                self.assertIsNone(history.draft)
                detail = workbench.group_details.toPlainText()
                self.assertIn("XX.OTHER..EHZ", detail)
                self.assertIn("Conflict", detail)
                self.assertIn("Missing", detail)
                self.assertEqual(workbench.viewer.get_time_range(), (946684802.4, 946684802.7))

                load_csv()
                self.assertIs(history.current, previous)
                self.assertIsNone(history.draft)
                self.assertIn("line 2", workbench.status.text().lower())
                self.assertIn("mismatch", workbench.status.text())
                group_csv.write_text(HEADER + "bad,2000-01-01T00:00:01,2000-01-01T00:00:02Z,,,,,,\n",
                                     encoding="utf-8")
                load_csv()
                self.assertIs(history.current, previous)
                self.assertIsNone(history.draft)
                self.assertIn("line 2", workbench.status.text().lower())

                click("undo")
                self.assertEqual(workbench.group_table.item(0, 1).text(), "Unresolved")
                click("undo")
                self.assertEqual(workbench.group_table.rowCount(), 0)
                assert_applied_unchanged()
                click("redo")
                click("redo")
                self.assertEqual(workbench.group_table.item(0, 1).text(), "Reviewed")
                workbench.group_table.selectRow(0)
                click("review_group")
                click("apply")
                self.assertEqual(workbench.group_table.item(0, 1).text(), "Unresolved")
                with patch.object(QInputDialog, "getText", return_value=("M01", True)) as name_prompt:
                    click("add_group_blow")
                self.assertIn("gap", str(name_prompt.call_args))
                click("apply")
                blow = history.current.state.blow("M01")
                self.assertEqual(blow.group_id, "gap")
                self.assertIsNone(blow.original.reviewed_ref)
                self.assertIsNone(blow.calibration.reviewed_ref)
                self.assertFalse(blow.included)

                workbench.unresolved_filter.setCurrentIndex(workbench.unresolved_filter.findData("blows"))
                click("next_unresolved")
                self.assertEqual(workbench.selected_blow_id, "B07")
                click("next_unresolved")
                self.assertEqual(workbench.selected_blow_id, "M01")
                self.assertEqual(workbench.review_tabs.currentIndex(), 0)
                self.assertEqual(workbench.viewer.get_time_range(), (946684802.1, 946684802.25))
                self.assertIn("gap", workbench.details.text())
                with patch.object(QInputDialog, "getText", return_value=("M02", True)):
                    click("add")
                click("apply")
                self.assertIsNone(history.current.state.blow("M02").group_id)

                with patch.object(QFileDialog, "getSaveFileName", return_value=(str(session_path), "")):
                    workbench.save_action.trigger()
                self.assertTrue(session_path.is_file(), workbench.status.text())
                group_csv.unlink()
                with patch.object(QFileDialog, "getOpenFileName", return_value=(str(session_path), "")):
                    workbench.open_session_action.trigger()
                self.assertIsNot(workbench.history, history)
                self.assertEqual(workbench.history.current.state, history.current.state)
                self.assertEqual(workbench.group_table.rowCount(), 2)
                self.assertEqual(workbench.result.reviewed.blow_ids, initial_fit.reviewed.blow_ids)

            launch_workbench((catalog, history), check=exercise)

    def test_group_navigation_handles_tiny_and_unplottable_windows(self):
        from PyQt5.QtCore import Qt
        from PyQt5.QtTest import QSignalSpy, QTest
        from PyQt5.QtWidgets import QApplication, QFileDialog

        from seisslate.example import load_example
        from seisslate.ui import launch_workbench

        with TemporaryDirectory() as destination:
            group_csv = Path(destination) / "Bounds.csv"
            group_csv.write_text(
                "group_id,start_utc,duration_s\n"
                "tiny,2000-01-01T00:00:01.234567891Z,0.000000001\n"
                "old,0001-01-01T00:00:00Z,1\n",
                encoding="utf-8",
            )

            def exercise(workbench):
                workbench.review_tabs.setCurrentIndex(1)
                with patch.object(QFileDialog, "getOpenFileName", return_value=(str(group_csv), "")):
                    QTest.mouseClick(workbench.buttons["import_groups"], Qt.LeftButton)
                QApplication.processEvents()
                self.assertIn("outside the waveform viewer", workbench.status.text())
                self.assertIn("0001-01-01T00:00:00.000000000Z", workbench.group_table.item(0, 2).text())

                QApplication.setActiveWindow(workbench.window)
                workbench.viewer.setFocus()
                QApplication.processEvents()

                def navigate(key, button):
                    clicked = QSignalSpy(workbench.buttons[button].clicked)
                    QTest.keyClick(workbench.viewer, key, Qt.ControlModifier)
                    self.assertTrue(len(clicked) or clicked.wait(1000), button)

                navigate(Qt.Key_Down, "next_unresolved")
                self.assertEqual(workbench.selected_group_id, "tiny")
                self.assertIn("Preview", workbench.status.text())
                self.assertIn("padded for display", workbench.status.text())
                low, high = workbench.viewer.get_time_range()
                self.assertLess(high - low, 0.11)
                self.assertLess(low, 946684801.2345679)
                self.assertGreater(high, 946684801.2345679)
                detail = workbench.group_details.toPlainText()
                self.assertIn("2000-01-01T00:00:01.234567891Z", detail)

                navigate(Qt.Key_Up, "previous_unresolved")
                self.assertEqual(workbench.selected_group_id, "old")
                self.assertEqual(workbench.viewer.get_time_range(), (low, high))
                self.assertIsNotNone(workbench.history.draft)

            launch_workbench(load_example(), check=exercise)
