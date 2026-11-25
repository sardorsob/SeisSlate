"""Exercise confirmation and Apply through the Qt raw evidence dock."""

import csv
import json
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, skipUnless
from unittest.mock import patch

import numpy as np


HAS_DESKTOP = all(find_spec(name) is not None for name in ("pyrocko", "PyQt5", "matplotlib"))


@skipUnless(HAS_DESKTOP, "Install SeisSlate with the desktop extra")
class EvidenceWorkbenchTest(TestCase):
    def test_apply_uses_the_displayed_raw_span_and_recovers_from_failed_capture(self):
        from PyQt5.QtCore import Qt
        from PyQt5.QtTest import QTest
        from PyQt5.QtWidgets import QApplication, QFileDialog, QScrollArea

        from seisslate.example import load_example
        from seisslate.review import mask_spans, restore_spans
        from seisslate.sources import SampleSpan
        from seisslate.ui import launch_workbench

        catalog, history = load_example()
        initial = history.current
        point = initial.state.blow("B01").original.reviewed_ref
        source = catalog.segment(point)

        def exercise(workbench):
            def activate(name, key=Qt.Key_Space):
                button = workbench.buttons[name]
                self.assertTrue(button.isEnabled(), name)
                button.setFocus()
                QTest.keyClick(button, key)
                QApplication.processEvents()

            workbench.select_blow("B01")
            baseline_fit = workbench.result.reviewed
            activate("raw_review")
            self.assertTrue(workbench.raw_view.isVisible())
            self.assertIn("processing not recorded", workbench.source_label.text())
            self.assertIn("B01", workbench.evidence_origin.text())
            self.assertIn("legacy-unassessed", workbench.evidence_summary.text())
            view = workbench.raw_view.view
            workbench.viewer.parent().lowpass_control.edited(30.0)
            self.assertEqual(workbench.viewer.lowpass, 30.0)
            self.assertIs(workbench.raw_view.view, view)
            np.testing.assert_array_equal(workbench.raw_view.axes.lines[0].get_ydata(),
                                          source.data[view.span.start:view.span.stop])

            self.assertEqual(workbench.evidence_status.currentData(), "candidate")
            workbench.evidence_status.setCurrentIndex(workbench.evidence_status.findData("accepted"))
            workbench.evidence_reviewer.setText("Field reviewer")
            workbench.reason.setText("First visible arrival")
            workbench.evidence_lower.setText(str(point.sample_index))
            workbench.evidence_upper.setText(str(point.sample_index))
            activate("show_raw_span")
            self.assertIn("zero width", workbench.raw_view.span_label.text())
            activate("confirm_raw_review")
            draft = history.draft
            self.assertIsNotNone(draft)
            self.assertFalse(workbench.export_action.isEnabled())
            self.assertFalse(workbench.export_evidence_action.isEnabled())
            self.assertEqual(draft.state.raw_evidence, ())
            self.assertIs(history.current, initial)
            workbench.select_blow("B02")
            self.assertEqual(workbench.selected_blow_id, "B01")
            self.assertFalse(workbench.role.isEnabled())
            workbench.table.selectRow(1)
            self.assertEqual(workbench.table.selectedItems()[0].text(), "B01")
            workbench.role.setCurrentIndex(workbench.role.findData("calibration"))
            self.assertEqual(workbench.role.currentData(), "original")
            self.assertIn("B01", workbench.status.text())
            with self.assertRaisesRegex(ValueError, "raw review preview first"):
                workbench.preview(initial.state, "Unrelated preview")
            self.assertIs(history.draft, draft)

            workbench.evidence_start.setText(str(view.span.start + 1))
            activate("show_raw_span")
            activate("apply")
            self.assertIn("confirm", workbench.status.text().lower())
            self.assertIs(history.current, initial)
            self.assertIs(history.draft, draft)
            activate("confirm_raw_review")
            draft = history.draft
            with patch.object(history, "apply_with_evidence", side_effect=ValueError("Capture test failure")):
                activate("apply")
            self.assertIn("Capture test failure", workbench.status.text())
            self.assertIs(history.current, initial)
            self.assertIs(history.draft, draft)
            self.assertEqual(workbench.reason.text(), "First visible arrival")

            confirmed_view = workbench.raw_view.view
            QTest.keyClick(workbench.raw_view.sample_control, Qt.Key_Up)
            workbench.viewer.parent().lowpass_control.edited(55.0)
            workbench.viewer.set_time_range(946_684_800.345, 946_684_800.422)
            self.assertIs(workbench.raw_view.view, confirmed_view)
            activate("apply")
            applied = history.current
            self.assertEqual(workbench.result.reviewed, baseline_fit)
            self.assertIn("Attestation current", workbench.details.text())
            self.assertIn("Observed lag", workbench.details.text())
            self.assertIn("Residual", workbench.details.text())
            self.assertIn("Fit requested yes", workbench.details.text())
            self.assertIn("effective yes", workbench.details.text())
            self.assertIn("Original sample 360", workbench.evidence_origin.text())
            self.assertIn("Reviewed sample 360", workbench.evidence_origin.text())
            self.assertIn("zero width", workbench.evidence_origin.text())
            self.assertIn(source.source_hash, workbench.evidence_provenance.text())
            workbench.window.resize(1240, 900)
            QApplication.processEvents()
            scroll = workbench.evidence_dock.findChild(QScrollArea)
            self.assertEqual(scroll.horizontalScrollBar().maximum(), 0)
            (record,) = applied.state.raw_evidence
            (decision,) = applied.state.pick_decisions
            self.assertEqual(applied.revision_id, record.applied_revision_id)
            self.assertEqual(decision.evidence_id, record.evidence_id)
            self.assertEqual(decision.reviewed_ref, point)
            self.assertEqual((decision.lower_ref, decision.upper_ref), (point, point))
            self.assertEqual(decision.reason, "First visible arrival")
            self.assertEqual(record.confirmation.view.span.start, view.span.start + 1)
            self.assertIn("Attestation current", workbench.evidence_summary.text())
            self.assertIsNone(history.draft)
            activate("undo")
            self.assertIs(history.current, initial)
            self.assertIn("legacy-unassessed", workbench.evidence_summary.text())
            activate("redo")
            self.assertIs(history.current, applied)
            self.assertIn("Attestation current", workbench.evidence_summary.text())

            # Apply regular review edits through the same controls as the evidence draft.
            inside = SampleSpan(source.segment_id, point.sample_index, point.sample_index + 1)
            outside = SampleSpan(source.segment_id, view.span.stop, view.span.stop + 1)
            for command, span, current in (
                (mask_spans, outside, True), (mask_spans, inside, False), (restore_spans, inside, False),
            ):
                revised = command(history.current.state, (span,), catalog=catalog, reason="Check mask staleness")
                workbench.preview(revised, "Check mask staleness")
                activate("apply")
                summary = workbench.evidence_summary.text()
                self.assertIn("Historical decision accepted", summary)
                self.assertIn("raw samples match", summary)
                self.assertIn("Attestation current" if current else "Attestation stale", summary)
                self.assertIs(history.current.state.raw_evidence[0], record)

            with TemporaryDirectory() as directory:
                path = Path(directory) / "Raw review é.json"
                with patch.object(QFileDialog, "getSaveFileName", return_value=(str(path), "")):
                    workbench.save_action.trigger()
                self.assertTrue(path.is_file(), workbench.status.text())
                self.assertEqual(json.loads(path.read_bytes())["schema_version"], 3)
                with patch.object(QFileDialog, "getOpenFileName", return_value=(str(path), "")):
                    workbench.open_session_action.trigger()
                self.assertIsNot(workbench.history, history)
                self.assertEqual(workbench.history.current, history.current)
                self.assertFalse(workbench.raw_view.isVisible())
                activate("raw_review")
                summary = workbench.evidence_summary.text()
                self.assertIn("Historical decision accepted by Field reviewer", summary)
                self.assertIn("Attestation stale", summary)
                self.assertIn("raw samples match", summary)
                self.assertIn("View restored", summary)

                file_menu = next(action.menu() for action in workbench.window.menuBar().actions()
                                 if action.text() == "&File")
                self.assertIn(workbench.export_evidence_action, file_menu.actions())
                self.assertEqual(workbench.export_evidence_action.text(), "Export evidence CSV…")
                session_bytes = path.read_bytes()
                for action, filename in ((workbench.export_action, "fit.csv"),
                                         (workbench.export_evidence_action, "evidence.csv")):
                    with patch.object(QFileDialog, "getSaveFileName", return_value=(str(path), "")):
                        action.trigger()
                    self.assertIn("active review session", workbench.status.text().lower())
                    self.assertEqual(path.read_bytes(), session_bytes)
                    destination = Path(directory) / filename
                    with patch.object(QFileDialog, "getSaveFileName", return_value=(str(destination), "")):
                        action.trigger()
                    self.assertTrue(destination.is_file(), workbench.status.text())
                with (Path(directory) / "fit.csv").open(newline="", encoding="utf-8") as file:
                    self.assertNotIn("decision_status", next(csv.DictReader(file)))
                with (Path(directory) / "evidence.csv").open(newline="", encoding="utf-8") as file:
                    accepted = next(csv.DictReader(file))
                self.assertEqual((accepted["blow_id"], accepted["role"]), ("B01", "original"))
                self.assertEqual(accepted["decision_status"], "accepted")
                self.assertEqual(accepted["raw_samples"], "raw_samples_match")
                self.assertEqual(accepted["attestation_current"], "false")

        launch_workbench((catalog, history), check=exercise)

    def test_missing_arrival_can_be_previewed_and_cancelled_or_recorded_with_its_reason(self):
        from PyQt5.QtCore import Qt
        from PyQt5.QtTest import QTest

        from seisslate.example import load_example
        from seisslate.ui import launch_workbench

        catalog, history = load_example()
        initial = history.current

        def exercise(workbench):
            def click(name):
                button = workbench.buttons[name]
                self.assertTrue(button.isEnabled(), name)
                QTest.mouseClick(button, Qt.LeftButton)

            workbench.select_blow("B07")
            click("mark_unresolved")
            self.assertEqual(workbench.evidence_status.currentData(), "unresolved")
            workbench.evidence_intended.setText("2000-01-01T00:00:02.175000000Z")
            click("show_raw_span")
            self.assertIsNone(workbench.raw_view.view)
            self.assertIn("unavailable", workbench.evidence_notice.text().lower())
            click("confirm_raw_review")
            self.assertIsNone(history.draft)
            self.assertIn("reason", workbench.status.text().lower())
            workbench.reason.setText("Recording gap at intended arrival")
            click("confirm_raw_review")
            self.assertIsNotNone(history.draft)
            click("cancel")
            self.assertIs(history.current, initial)
            self.assertIsNone(history.draft)

            click("confirm_raw_review")
            click("apply")
            (decision,) = history.current.state.pick_decisions
            self.assertEqual((decision.blow_id, decision.role), ("B07", "original"))
            self.assertEqual(decision.status, "unresolved")
            self.assertIsNone(decision.reviewed_ref)
            self.assertIsNone(decision.inspection_span)
            self.assertEqual(decision.intended_utc_ns, 946_684_802_175_000_000)
            self.assertEqual(decision.reason, "Recording gap at intended arrival")
            self.assertEqual(history.current.state.raw_evidence, ())
            self.assertIn("unavailable", workbench.evidence_summary.text().lower())

            # A no-point decision can also record genuinely inspected acquisition support.
            workbench.evidence_intended.setText("2000-01-01T00:00:02.145000000Z")
            workbench.evidence_reviewer.setText("Field reviewer")
            workbench.reason.setText("Inspected samples without an identifiable onset")
            click("show_raw_span")
            self.assertIsNotNone(workbench.raw_view.view)
            self.assertIsNone(workbench.raw_view.view.point)
            click("confirm_raw_review")
            click("apply")
            (record,) = history.current.state.raw_evidence
            self.assertIsNone(record.confirmation.decision.reviewed_ref)
            self.assertEqual(record.confirmation.decision.intended_utc_ns, 946_684_802_145_000_000)
            self.assertIn("Attestation current", workbench.evidence_summary.text())

        launch_workbench((catalog, history), check=exercise)

    def test_unconfirmed_raw_review_is_cleared_on_blow_or_receiver_change(self):
        from seisslate.example import load_example
        from seisslate.ui import launch_workbench

        catalog, history = load_example()
        initial = history.current

        def exercise(workbench):
            workbench.buttons["raw_review"].click()
            self.assertIn("B01 · Original", workbench.evidence_origin.text())
            workbench.table.selectRow(1)
            self.assertEqual(workbench.selected_blow_id, "B02")
            self.assertFalse(workbench.evidence_dock.isVisible())
            self.assertIsNone(workbench.raw_view.view)
            self.assertFalse(workbench.buttons["confirm_raw_review"].isEnabled())

            workbench.buttons["raw_review"].click()
            self.assertIn("B02 · Original", workbench.evidence_origin.text())
            workbench.role.setCurrentIndex(workbench.role.findData("calibration"))
            self.assertFalse(workbench.evidence_dock.isVisible())
            self.assertFalse(workbench.buttons["confirm_raw_review"].isEnabled())
            self.assertIs(history.current, initial)
            self.assertIsNone(history.draft)

            workbench.buttons["raw_review"].click()
            self.assertIn("B02 · Calibration", workbench.evidence_origin.text())
            self.assertEqual(workbench.raw_view.view.point, initial.state.blow("B02").calibration.reviewed_ref)

        launch_workbench((catalog, history), check=exercise)

    def test_undo_removing_an_unconfirmed_raw_review_target_clears_the_dock(self):
        from PyQt5.QtWidgets import QInputDialog

        from seisslate.example import load_example
        from seisslate.ui import launch_workbench

        catalog, history = load_example()
        initial = history.current

        def exercise(workbench):
            with patch.object(QInputDialog, "getText", return_value=("Temporary blow", True)):
                workbench.buttons["add"].click()
            workbench.buttons["apply"].click()
            workbench.select_blow("Temporary blow")
            workbench.buttons["raw_review"].click()
            self.assertIn("Temporary blow", workbench.evidence_origin.text())
            self.assertTrue(workbench.evidence_dock.isVisible())

            workbench.buttons["undo"].click()
            self.assertIs(history.current, initial)
            self.assertEqual(workbench.selected_blow_id, "B01")
            self.assertFalse(workbench.evidence_dock.isVisible())
            self.assertIsNone(workbench.raw_view.view)
            self.assertIn("Applied revision", workbench.status.text())

        launch_workbench((catalog, history), check=exercise)

    def test_blow_queue_visits_affected_receivers_and_preserves_pair_failures(self):
        from dataclasses import replace

        from seisslate.evidence import PickDecision, RawEvidenceView
        from seisslate.example import load_example
        from seisslate.review import ReviewHistory, mask_spans, set_pair_status, set_pick_decision
        from seisslate.sources import SampleSpan
        from seisslate.ui import launch_workbench

        catalog, original = load_example()
        selected_ids = {"B01", "B02", "B03", "B04", "B07", "B08"}
        state = original.current.state
        history = ReviewHistory(replace(state, blows=tuple(blow for blow in state.blows if blow.blow_id in selected_ids)))
        stale_span = None
        for blow in history.current.state.blows:
            for role in ("original", "calibration"):
                point = getattr(blow, role).reviewed_ref
                if point is None or blow.blow_id == "B03":
                    continue
                segment = catalog.segment(point)
                span = SampleSpan(point.segment_id, point.sample_index - 1, point.sample_index + 1)
                view = RawEvidenceView(point.source_hash, point.channel, span, segment.time_ns(span.start),
                                       segment.time_ns(span.stop), point, None)
                status = "candidate" if (blow.blow_id, role) == ("B08", "calibration") else "accepted"
                decision = PickDecision(blow.blow_id, role, point, status, reason="Clear arrival",
                                        reviewer="Reviewer", inspection_span=span)
                history.preview(set_pick_decision(history.current.state, catalog, decision), "Review pick")
                confirmation = history.confirm_current_pick_evidence(catalog, decision, view)
                history.apply_with_evidence(catalog, confirmation, view, captured_utc_ns=1,
                                            software_versions=(("SeisSlate", "test"),))
                if (blow.blow_id, role) == ("B02", "calibration"):
                    stale_span = SampleSpan(span.segment_id, span.start, span.start + 1)
        state = set_pair_status(history.current.state, "B04", "ambiguous", "Pairing needs review")
        self.assertIsNotNone(stale_span)
        state = mask_spans(state, (stale_span,), catalog=catalog, reason="Changed support near onset")
        history.preview(state, "Apply review changes")
        applied = history.apply()

        def exercise(workbench):
            workbench.unresolved_filter.setCurrentIndex(workbench.unresolved_filter.findData("blows"))
            workbench.select_blow("B01")
            for expected in (("B02", "calibration"), ("B03", "original"), ("B03", "calibration"),
                             ("B04", "original"), ("B07", "original"), ("B08", "calibration"),
                             ("B02", "calibration")):
                workbench.buttons["next_unresolved"].click()
                self.assertEqual((workbench.selected_blow_id, workbench.role.currentData()), expected)
            workbench.buttons["previous_unresolved"].click()
            self.assertEqual((workbench.selected_blow_id, workbench.role.currentData()), ("B08", "calibration"))
            self.assertIs(history.current, applied)
            self.assertIsNone(history.draft)

            baseline_fit = workbench.result.reviewed
            workbench.buttons["raw_review"].click()
            self.assertEqual(workbench.evidence_status.currentData(), "candidate")
            workbench.evidence_status.setCurrentIndex(workbench.evidence_status.findData("accepted"))
            workbench.evidence_reviewer.setText("Reviewer")
            workbench.reason.setText("Confirmed candidate against original samples")
            workbench.buttons["confirm_raw_review"].click()
            self.assertIsNotNone(history.draft)
            workbench.buttons["apply"].click()
            self.assertIsNone(history.draft)
            self.assertIn("Historical decision accepted", workbench.evidence_summary.text())
            self.assertEqual(workbench.result.reviewed, baseline_fit)
            workbench.buttons["next_unresolved"].click()
            workbench.buttons["previous_unresolved"].click()
            self.assertEqual((workbench.selected_blow_id, workbench.role.currentData()), ("B07", "original"))

        launch_workbench((catalog, history), check=exercise)
