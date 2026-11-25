"""Snuffler workbench for paired, source-referenced seismic review."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from importlib.metadata import PackageNotFoundError, version
from math import nextafter
from pathlib import Path
from platform import python_version
from time import time_ns

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from pyrocko import config, pile, snuffle, util
from pyrocko.gui.qt_compat import qc, qg, qw
from pyrocko.gui.snuffler.marker import Marker
from pyrocko.gui.snuffler.snuffling import Snuffling
from pyrocko.trace import Trace

from .analysis import Analysis, analyze
from .evidence import PickDecision, RawConfirmation, RawEvidenceView, require_reason
from .example import load_example
from .groups import GroupWindow, inspect_group, load_group_csv
from .raw_view import RAW_ORDINATE_SCALE, RawView
from .review import (
    Blow, Pick, Receiver, ReviewHistory, ReviewState, Revision, Role, add_blow, add_groups, mask_spans,
    move_pick, nudge_pick, restore_spans, set_geometry, set_inclusion,
    set_group_reviewed, set_pair_status, set_pick_decision,
)
from .sources import NS_PER_SECOND, SampleRef, SampleSpan, Segment, SourceCatalog, load_source, utc_ns


ROLES: tuple[Role, Role] = ("original", "calibration")
MINISEED_FILTER = "MiniSEED (*.mseed *.miniseed *.seed *.ms *.MSD);;All files (*)"
SESSION_FILTER = "SeisSlate review (*.json)"
NOT_AVAILABLE = "—"
MIN_WAVEFORM_VIEW_S = 0.05


def _utc(nanoseconds: int) -> str:
    seconds, remainder = divmod(nanoseconds, NS_PER_SECOND)
    stamp = datetime(1970, 1, 1) + timedelta(seconds=seconds)
    return f"{stamp.isoformat()}.{remainder:09d}Z"


def _capture_versions() -> tuple[tuple[str, str], ...]:
    versions = [("Python", python_version())]
    for name in ("SeisSlate", "numpy", "matplotlib", "PyQt5", "pyrocko"):
        try:
            release = version(name)
        except PackageNotFoundError:
            release = "unknown (unpackaged runtime)"
        versions.append((name, release))
    return tuple(versions)


def _marker_ns(seconds) -> int:
    whole = int(seconds)
    return whole * NS_PER_SECOND + round(float(seconds - whole) * NS_PER_SECOND)


def _nslc(channel: str) -> tuple[str, ...]:
    return tuple(channel.split("."))


def _marker(channel, start_ns, stop_ns=None, *, kind):
    start = start_ns / NS_PER_SECOND
    stop = start if stop_ns is None else nextafter(stop_ns / NS_PER_SECOND, start)
    return Marker([_nslc(channel)], start, stop, kind=kind)


class Workbench(qc.QObject):
    def __init__(self, window, initial=None, *, fixture=None):
        super().__init__(window)
        self.window = window
        self.fixture = fixture
        self.viewer = window.get_view()
        self.viewer.setAcceptDrops(False)
        file_menu = next(action.menu() for action in window.menuBar().actions() if action.text() == "&File")
        for action in file_menu.actions():
            if action.text() in ("Open waveform files...", "Open waveform directory..."):
                action.setEnabled(False)
            elif action.text() == "Save markers...":
                action.setShortcut(qg.QKeySequence())
        self.waveforms = Snuffling()
        self.waveforms.set_name("SeisSlate recordings")
        self.viewer.add_snuffling(self.waveforms)
        self.overlay = Snuffling()
        self.overlay.set_name("SeisSlate review markers")
        self.viewer.add_snuffling(self.overlay)
        self.managed_markers: dict[Marker, str | None] = {}
        self._active: tuple[SourceCatalog, ReviewHistory] | None = None
        self.result: Analysis
        self.receivers: dict[Role, Receiver] = {}
        self.segments: tuple[Segment, ...] = ()
        self.selected_blow_id: str | None = None
        self.selected_group_id: str | None = None
        self.session_path: Path | None = None
        self.saved_revision = None
        self.buttons: dict[str, qw.QPushButton] = {}
        self.plot_blow_ids: tuple[str, ...] = ()
        self._evidence_target: tuple[str, Role] | None = None
        self._raw_draft: Revision | None = None
        self._raw_confirmation: RawConfirmation | None = None
        self._raw_dirty = True
        self._raw_gap = False

        window.setWindowTitle("SeisSlate")
        window.resize(1440, 920)
        window.installEventFilter(self)
        self.viewer.installEventFilter(self)
        self.viewer.marker_selection_changed.connect(self.link_marker_selection)
        toolbar = window.addToolBar("Review files")
        for name, label, method in (
            ("open_pair_action", "Open pair…", self.open_pair),
            ("open_session_action", "Open review…", self.open_review),
            ("example_action", "Synthetic example", self.open_example),
            ("save_action", "Save review…", self.save_review),
            ("export_action", "Export CSV…", self.export_review),
        ):
            action = toolbar.addAction(label)
            action.triggered.connect(self.wrap(method))
            setattr(self, name, action)
        file_menu.addSeparator()
        self.export_evidence_action = file_menu.addAction("Export evidence CSV…")
        self.export_evidence_action.triggered.connect(self.wrap(self.export_evidence_review))
        self.open_pair_action.setShortcut(qg.QKeySequence.Open)
        self.save_action.setShortcut(qg.QKeySequence.Save)

        panel = qw.QWidget()
        panel_layout = qw.QVBoxLayout(panel)
        self.source_label = qw.QLabel("Open a recording pair, saved review, or synthetic example.")
        self.source_label.setTextFormat(qc.Qt.PlainText)
        self.source_label.setWordWrap(True)
        panel_layout.addWidget(self.source_label)

        navigation = qw.QHBoxLayout()
        self.unresolved_filter = qw.QComboBox()
        self.unresolved_filter.setAccessibleName("Unresolved review filter")
        self.unresolved_filter.addItem("Hammer groups", "groups")
        self.unresolved_filter.addItem("Blows", "blows")
        navigation.addWidget(self.unresolved_filter)
        self.button(navigation, "previous_unresolved", "Previous unresolved", lambda: self.unresolved(-1))
        self.button(navigation, "next_unresolved", "Next unresolved", lambda: self.unresolved(1))
        panel_layout.addLayout(navigation)
        for name, keys in (("previous_unresolved", "Ctrl+Up"), ("next_unresolved", "Ctrl+Down")):
            self.buttons[name].setShortcut(qg.QKeySequence(keys))
            self.buttons[name].setToolTip(keys)

        self.review_tabs = qw.QTabWidget()
        panel_layout.addWidget(self.review_tabs, 1)
        paired = qw.QWidget()
        layout = qw.QVBoxLayout(paired)
        self.review_tabs.addTab(paired, "Paired review")
        self.review_tabs.addTab(self.create_group_panel(), "Hammer groups")

        self.table = qw.QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(("Blow", "Fit status", "Lag (ms)", "Orig. UTC", "Cal. UTC",
                                               "Δx (m)", "Resid. (m)"))
        self.table.setSelectionBehavior(qw.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(qw.QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(qw.QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().setSectionResizeMode(qw.QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemSelectionChanged.connect(self.table_selected)
        self.table.cellClicked.connect(self.zoom_to_selected)
        font = self.table.font()
        font.setPointSize(10)
        self.table.setFont(font)
        self.table.setMinimumWidth(555)
        self.table.setMinimumHeight(150)
        layout.addWidget(self.table, 1)

        self.fig = Figure(figsize=(5.2, 2.9), layout="constrained")
        self.axes = self.fig.add_subplot()
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setMinimumHeight(200)
        self.canvas.mpl_connect("pick_event", self.plot_selected)
        layout.addWidget(self.canvas, 1)
        self.fit_label = qw.QLabel()
        self.fit_label.setWordWrap(True)
        layout.addWidget(self.fit_label)

        self.details = qw.QLabel()
        self.details.setTextFormat(qc.Qt.PlainText)
        self.details.setWordWrap(True)
        self.details.setTextInteractionFlags(qc.Qt.TextSelectableByMouse | qc.Qt.TextSelectableByKeyboard)
        layout.addWidget(self.details)
        self.role = qw.QComboBox()
        for role in (*ROLES, "both"):
            self.role.addItem(role.capitalize(), role)
        row = qw.QHBoxLayout()
        row.addWidget(self.role)
        self.button(row, "earlier", "−1 sample", lambda: self.nudge(-1))
        self.button(row, "later", "+1 sample", lambda: self.nudge(1))
        self.button(row, "point", "Use point", self.use_point)
        self.button(row, "overview", "Full waveform", self.overview)
        layout.addLayout(row)
        row = qw.QHBoxLayout()
        self.button(row, "mask", "Mask interval", lambda: self.edit_mask(restore=False))
        self.button(row, "restore", "Restore interval", lambda: self.edit_mask(restore=True))
        self.button(row, "inclusion", "Toggle fit inclusion", self.toggle_inclusion)
        layout.addLayout(row)
        row = qw.QHBoxLayout()
        self.button(row, "add", "Add blow…", self.new_blow)
        self.button(row, "geometry", "Distances…", self.change_geometry)
        self.button(row, "pairing", "Pair status…", self.change_pairing)
        self.button(row, "raw_review", "Review &now", self.open_raw_review)
        self.button(row, "mark_unresolved", "Mark &unresolved", self.mark_unresolved)
        layout.addLayout(row)

        self.reason = qw.QLineEdit()
        self.reason.setPlaceholderText("Reason for edit or raw review")
        self.reason.setAccessibleName("Reason for edit")
        panel_layout.addWidget(self.reason)
        row = qw.QHBoxLayout()
        self.button(row, "apply", "Apply", self.apply)
        self.button(row, "cancel", "Cancel preview", self.cancel)
        self.button(row, "undo", "Undo", lambda: self.navigate(self.history.undo))
        self.button(row, "redo", "Redo", lambda: self.navigate(self.history.redo))
        panel_layout.addLayout(row)
        self.buttons["undo"].setShortcut("Ctrl+Z")
        self.buttons["redo"].setShortcut("Ctrl+Shift+Z")

        help_text = qw.QLabel("Double-click a trace, then click for a point or drag for an interval.")
        help_text.setWordWrap(True)
        layout.addWidget(help_text)
        self.status = qw.QLabel()
        self.status.setTextFormat(qc.Qt.PlainText)
        self.status.setWordWrap(True)
        panel_layout.addWidget(self.status)
        window.add_panel("Paired review", panel, visible=True, where=qc.Qt.RightDockWidgetArea)
        self.create_evidence_panel()
        self.role.currentIndexChanged.connect(self.update_controls)

        if initial is not None:
            self.set_session(*initial)
        else:
            self.draw_results()
            self.update_controls()

    def create_evidence_panel(self):
        panel = qw.QWidget()
        layout = qw.QVBoxLayout(panel)
        for name in ("origin", "summary", "notice"):
            label = qw.QLabel()
            label.setTextFormat(qc.Qt.PlainText)
            label.setWordWrap(True)
            setattr(self, f"evidence_{name}", label)
        layout.addWidget(self.evidence_origin)
        layout.addWidget(self.evidence_summary)
        self.raw_view = RawView()
        layout.addWidget(self.raw_view)

        grid = qw.QGridLayout()
        fields = (
            ("start", "First &sample", 0, 0), ("stop", "&Exclusive stop", 0, 2),
            ("lower", "&Lower inclusive", 1, 0), ("upper", "&Upper inclusive", 1, 2),
            ("reviewer", "&Reviewer label", 2, 0), ("intended", "&Intended UTC (ISO)", 3, 0),
        )
        for name, title, row, column in fields:
            edit = qw.QLineEdit()
            label = qw.QLabel(title)
            label.setBuddy(edit)
            grid.addWidget(label, row, column)
            grid.addWidget(edit, row, column + 1, 1, 3 if name == "intended" else 1)
            setattr(self, f"evidence_{name}", edit)
            if name != "reviewer":
                edit.textChanged.connect(self._raw_inputs_changed)
            if name == "intended":
                self.evidence_intended_label = label
        self.evidence_status = qw.QComboBox()
        status_label = qw.QLabel("&Decision")
        status_label.setBuddy(self.evidence_status)
        grid.addWidget(status_label, 2, 2)
        grid.addWidget(self.evidence_status, 2, 3)
        layout.addLayout(grid)
        self.evidence_provenance = qw.QLabel()
        self.evidence_provenance.setTextFormat(qc.Qt.PlainText)
        self.evidence_provenance.setWordWrap(True)
        self.evidence_provenance.setSizePolicy(qw.QSizePolicy.Ignored, qw.QSizePolicy.Preferred)
        self.evidence_provenance.setTextInteractionFlags(qc.Qt.TextSelectableByMouse | qc.Qt.TextSelectableByKeyboard)
        layout.addWidget(self.evidence_provenance)
        scroll = qw.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(qw.QFrame.NoFrame)
        scroll.setMinimumHeight(170)
        scroll.setWidget(panel)
        dock_panel = qw.QWidget()
        dock_layout = qw.QVBoxLayout(dock_panel)
        dock_layout.setContentsMargins(0, 0, 0, 0)
        dock_layout.addWidget(scroll)
        row = qw.QHBoxLayout()
        self.button(row, "show_raw_span", "&Show raw span", self.show_raw_span)
        self.button(row, "confirm_raw_review", "&Confirm review", self.confirm_raw_review)
        row.addWidget(self.evidence_notice, 1)
        dock_layout.addLayout(row)
        self.window.setCorner(qc.Qt.BottomRightCorner, qc.Qt.RightDockWidgetArea)
        self.window.add_panel("Raw pick evidence", dock_panel, where=qc.Qt.BottomDockWidgetArea)
        self.evidence_dock = dock_panel.parentWidget()
        self.close_raw_review()

    def close_raw_review(self):
        self._evidence_target = None
        self._raw_dirty = True
        self._raw_gap = False
        self.raw_view.clear()
        for field in (self.evidence_start, self.evidence_stop, self.evidence_lower,
                      self.evidence_upper, self.evidence_intended):
            field.clear()
        self.evidence_status.clear()
        self.evidence_notice.clear()
        self.evidence_dock.hide()
        self.describe_evidence()

    def _raw_inputs_changed(self, *_):
        self._raw_dirty = True
        self.evidence_notice.setText("Show the edited span before confirming your review.")

    def open_raw_review(self):
        if self._raw_draft is not None:
            self.evidence_dock.show()
            self.evidence_dock.raise_()
            return
        if self.history.draft is not None:
            raise ValueError("Apply or cancel the active preview first.")
        blow, role = self.require_blow(), self.role_for_pick()
        self._evidence_target = blow.blow_id, role
        point = getattr(blow, role).reviewed_ref
        decision = self._pick_decision(self._evidence_target)
        span = decision.inspection_span if decision is not None else None
        intended = decision.intended_utc_ns if decision is not None else getattr(blow, role).imported_ns
        self.evidence_intended.setVisible(point is None)
        self.evidence_intended_label.setVisible(point is None)
        self.evidence_lower.setEnabled(point is not None)
        self.evidence_upper.setEnabled(point is not None)
        self.evidence_start.setText(str(span.start) if span else "")
        self.evidence_stop.setText(str(span.stop) if span else "")
        self.evidence_lower.setText(str(decision.lower_ref.sample_index) if decision and decision.lower_ref else "")
        self.evidence_upper.setText(str(decision.upper_ref.sample_index) if decision and decision.upper_ref else "")
        self.evidence_intended.setText(_utc(intended) if point is None and intended is not None else "")
        self.evidence_status.clear()
        if point is not None:
            self.evidence_status.addItem("Candidate", "candidate")
            self.evidence_status.addItem("Accepted", "accepted")
        self.evidence_status.addItem("Unresolved", "unresolved")
        if decision is not None:
            self.evidence_status.setCurrentIndex(self.evidence_status.findData(decision.status))
        self.raw_view.clear()
        self._raw_dirty = True
        self._raw_gap = False
        self.evidence_dock.show()
        self.evidence_dock.raise_()
        self.window.resizeDocks([self.evidence_dock], [500], qc.Qt.Vertical)
        self.describe_evidence()
        self.update_controls()
        if point is not None or intended is not None:
            self.show_raw_span()
            self.raw_view.sample_control.setFocus()
        else:
            self.evidence_notice.setText("Enter the intended UTC for this unresolved arrival.")
            self.evidence_intended.setFocus()

    def mark_unresolved(self):
        if self.history.draft is not None:
            raise ValueError("Apply or cancel the active preview first.")
        self.open_raw_review()
        self.evidence_status.setCurrentIndex(self.evidence_status.findData("unresolved"))

    def _pick_decision(self, target: tuple[str, Role] | None) -> PickDecision | None:
        return next((item for item in self.state().pick_decisions if item.target == target), None)

    def show_raw_span(self):
        if self._evidence_target is None:
            raise ValueError("Open a raw review for a blow and receiver first.")
        blow_id, role = self._evidence_target
        point = getattr(self.state().blow(blow_id), role).reviewed_ref
        self._raw_gap = False
        if point is not None:
            segment = self.catalog.segment(point)
            anchor = point.sample_index
        else:
            intended = utc_ns(self.evidence_intended.text().strip())
            receiver = self.receivers[role]
            segment = next((segment for segment in self.segments
                            if segment.channel == receiver.channel and segment.start_ns <= intended
                            < segment.start_ns + segment.npts * segment.period_ns), None)
            if segment is None:
                self.raw_view.clear()
                self._raw_gap = True
                self._raw_dirty = False
                self.evidence_notice.setText("Raw evidence unavailable at the intended UTC. You can record an unresolved decision.")
                self.describe_evidence()
                return
            anchor = segment.lower_bound(intended)

        start_text, stop_text = self.evidence_start.text().strip(), self.evidence_stop.text().strip()
        if not (start_text or stop_text):
            start_text, stop_text = str(max(0, anchor - 25)), str(min(segment.npts, anchor + 26))
            self.evidence_start.setText(start_text)
            self.evidence_stop.setText(stop_text)
        try:
            span = SampleSpan(segment.segment_id, int(start_text), int(stop_text))
        except ValueError as error:
            raise ValueError("Enter integer first and exclusive stop samples for a nonempty raw span.") from error
        lower, upper = self.evidence_lower.text().strip(), self.evidence_upper.text().strip()
        bounds = None
        if lower or upper:
            try:
                bounds = segment.ref(int(lower)), segment.ref(int(upper))
            except ValueError as error:
                raise ValueError("Enter both inclusive bounds as source sample indices.") from error
        self.raw_view.show_span(segment, span, point=point, bounds=bounds)
        self._raw_dirty = False
        if self._raw_confirmation is not None and self.displayed_raw_evidence() != self._raw_confirmation.view:
            self._raw_confirmation = None
        if self._raw_draft is None:
            self.evidence_notice.setText("Inspect the raw samples and confirm your review. Enter a reason in the review panel.")
        elif self._raw_confirmation is None:
            self.evidence_notice.setText("Confirm the current raw review before Apply.")
        else:
            self.evidence_notice.setText(f"{self._raw_draft.reason} preview ready. Apply or Cancel.")
        self.describe_evidence()

    def displayed_raw_evidence(self) -> RawEvidenceView | None:
        view = self.raw_view.view
        if view is None:
            return None
        return RawEvidenceView(view.source_hash, view.channel, view.span, view.display_start_ns,
                               view.display_stop_ns, view.point, view.bounds, RAW_ORDINATE_SCALE)

    def _raw_review_decision(self, view: RawEvidenceView | None) -> PickDecision:
        if self._evidence_target is None:
            raise ValueError("Open a raw review for a blow and receiver first.")
        if self._evidence_target != (self.selected_blow_id, self.role.currentData()):
            raise ValueError("Open the raw review for the selected blow and receiver before confirming.")
        if self._raw_dirty:
            raise ValueError("Show the edited raw span, then confirm the current review.")
        blow_id, role = self._evidence_target
        point = getattr(self.state().blow(blow_id), role).reviewed_ref
        if view is not None and view.point != point:
            raise ValueError("The reviewed point changed. Show its raw span again before confirming.")
        if view is None and not (point is None and self._raw_gap):
            raise ValueError("A displayed raw span is required to confirm this review.")
        if view is not None and not self.evidence_reviewer.text().strip():
            raise ValueError("Enter a local reviewer label before confirming raw evidence.")
        lower, upper = view.bounds if view is not None and view.bounds is not None else (None, None)
        return PickDecision(
            blow_id, role, point, self.evidence_status.currentData(),
            lower_ref=lower, upper_ref=upper, reason=self.reason.text().strip(),
            reviewer=self.evidence_reviewer.text().strip(),
            intended_utc_ns=utc_ns(self.evidence_intended.text().strip()) if point is None else None,
            inspection_span=view.span if view is not None else None,
        )

    def confirm_raw_review(self):
        if self.history.draft is not self._raw_draft:
            raise ValueError("Apply or cancel the other active preview first.")
        view = self.displayed_raw_evidence()
        decision = self._raw_review_decision(view)
        state = set_pick_decision(self.state(), self.catalog, decision)
        origin = f"{decision.blow_id} · {decision.role.capitalize()}"
        self._raw_draft = self.history.preview(state, f"{origin} raw review")
        self._raw_confirmation = None
        self.describe_group()
        self.update_controls()
        if view is not None:
            self._raw_confirmation = self.history.confirm_current_pick_evidence(self.catalog, decision, view)
        self.evidence_notice.setText(f"{origin} preview ready. Apply or Cancel.")

    def describe_evidence(self):
        if self._evidence_target is None:
            self.evidence_origin.setText("Choose a blow and receiver, then select Review now.")
            self.evidence_summary.clear()
            self.evidence_provenance.clear()
            return
        blow_id, role = self._evidence_target
        blow = self.state().blow(blow_id)
        pick = getattr(blow, role)
        decision = self._pick_decision(self._evidence_target)
        imported = _utc(pick.imported_ns) if pick.imported_ns is not None else NOT_AVAILABLE
        lines = [f"{blow_id} · {role.capitalize()} · {self.receivers[role].channel}", f"Imported UTC {imported}"]
        if pick.original_ref is not None and pick.original_ref == pick.reviewed_ref:
            ref = pick.original_ref
            timestamp = self.catalog.time_ns(ref)
            samples = f"Original sample {ref.sample_index} · Reviewed sample {ref.sample_index}"
            if timestamp == pick.imported_ns:
                lines.append(f"{samples} · At imported UTC")
            else:
                lines.extend((samples, f"Sample UTC {_utc(timestamp)}"))
        else:
            lines += [f"Original {self._sample_text(pick.original_ref)}",
                      f"Reviewed {self._sample_text(pick.reviewed_ref)}"]
        if decision is not None:
            if decision.intended_utc_ns is not None:
                lines.append(f"Intended UTC {_utc(decision.intended_utc_ns)}")
            if decision.lower_ref is not None:
                lower, upper = decision.lower_ref, decision.upper_ref
                bounds = self._sample_text(lower)
                bounds += " (zero width)" if lower == upper else f" to {self._sample_text(upper)}"
                lines.append(f"Plausible onset bounds, inclusive · {bounds}")
        self.evidence_origin.setText("\n".join(lines))
        self._describe_pick_provenance(blow, pick, decision)
        if decision is None:
            self.evidence_summary.setText("Decision legacy-unassessed. Raw evidence not recorded.")
            return
        history_status = f"Historical decision {decision.status}"
        if decision.reviewer:
            history_status += f" by {decision.reviewer}"
        if decision.reason:
            history_status += f" · {decision.reason}"
        if decision.evidence_id is None:
            self.evidence_summary.setText(f"{history_status}. Raw evidence unavailable.")
            return
        replay = self.history.replay_pick_evidence(self.catalog, blow_id, role, view=self.displayed_raw_evidence())
        details = [history_status, "Attestation current" if replay.attestation_current else "Attestation stale or unavailable",
                   replay.source_identity.replace("_", " "), replay.raw_samples.replace("_", " ")]
        if replay.grid_matches is not None:
            details.append("Sampling grid matches" if replay.grid_matches else "Sampling grid differs")
        if replay.method_supported is False:
            details.append("Recorded method unsupported")
        if replay.recorded_view_valid is False:
            details.append("Recorded view is invalid")
        if replay.view_restored is not None:
            details.append("View restored" if replay.view_restored else "Displayed view differs")
        if replay.masks_changed:
            details.append("Mask changed within recorded support. Confirm a new raw review.")
        self.evidence_summary.setText(" · ".join(details))

    def _sample_text(self, ref: SampleRef | None) -> str:
        if ref is None:
            return "sample unavailable"
        return f"sample {ref.sample_index} · {_utc(self.catalog.time_ns(ref))}"

    def _describe_pick_provenance(self, blow: Blow, pick: Pick, decision: PickDecision | None) -> None:
        lines = []
        if blow.reason:
            lines.append(f"Fit inclusion reason · {blow.reason}")
        if blow.pair_reason:
            lines.append(f"Pairing reason · {blow.pair_reason}")
        identities: dict[tuple[str, str, str], list[str]] = {}
        for label, ref in (("Original", pick.original_ref), ("Reviewed", pick.reviewed_ref)):
            if ref is not None:
                identities.setdefault((ref.source_hash, ref.channel, ref.segment_id), []).append(label)
        for (source, channel, segment), labels in identities.items():
            label = " and ".join(labels)
            lines.append(f"{label} · Source SHA-256 {source} · Channel {channel} · Segment {segment}")
        if decision is not None and decision.evidence_id is not None:
            record = next(item for item in self.state().raw_evidence if item.evidence_id == decision.evidence_id)
            view = record.confirmation.view
            if (view.source_hash, view.channel, view.span.segment_id) not in identities:
                lines.append(f"Capture source SHA-256 {view.source_hash} · Channel {view.channel} · "
                             f"Segment {view.span.segment_id}")
            releases = " · ".join(f"{name} {release}" for name, release in record.software_versions)
            lines.append(f"{record.method_id} · Revision {record.applied_revision_id} · "
                         f"Captured {_utc(record.captured_utc_ns)}\n{releases}")
        self.evidence_provenance.setText("\n".join(lines))
        self.evidence_provenance.setVisible(bool(lines))

    def create_group_panel(self):
        panel = qw.QWidget()
        layout = qw.QVBoxLayout(panel)
        actions = qw.QHBoxLayout()
        self.button(actions, "import_groups", "Import group CSV…", self.import_groups)
        self.button(actions, "review_group", "Mark reviewed", self.review_group)
        self.buttons["review_group"].setToolTip("Change the group's queue status")
        self.button(actions, "add_group_blow", "Add blow to group…", self.add_group_blow)
        layout.addLayout(actions)

        self.group_table = qw.QTableWidget(0, 3)
        self.group_table.setAccessibleName("Hammer group windows")
        self.group_table.setHorizontalHeaderLabels(("Group ID", "Queue status", "Start / End (UTC)"))
        self.group_table.setSelectionBehavior(qw.QAbstractItemView.SelectRows)
        self.group_table.setSelectionMode(qw.QAbstractItemView.SingleSelection)
        self.group_table.setEditTriggers(qw.QAbstractItemView.NoEditTriggers)
        self.group_table.verticalHeader().hide()
        self.group_table.verticalHeader().setSectionResizeMode(qw.QHeaderView.ResizeToContents)
        header = self.group_table.horizontalHeader()
        header.resizeSection(0, 112)
        header.setSectionResizeMode(1, qw.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, qw.QHeaderView.Stretch)
        self.group_table.itemSelectionChanged.connect(self.group_selected)
        self.group_table.cellClicked.connect(self.zoom_to_group)
        layout.addWidget(self.group_table, 1)

        self.group_details = qw.QPlainTextEdit()
        self.group_details.setAccessibleName("Selected hammer group evidence")
        self.group_details.setReadOnly(True)
        self.group_details.setPlaceholderText("Select a hammer group to inspect its evidence.")
        layout.addWidget(self.group_details, 2)
        return panel

    @property
    def catalog(self) -> SourceCatalog:
        if self._active is None:
            raise ValueError("Open a recording pair first.")
        return self._active[0]

    @property
    def history(self) -> ReviewHistory:
        if self._active is None:
            raise ValueError("Open a recording pair first.")
        return self._active[1]

    def wrap(self, method):
        def invoke():
            try:
                method()
            except Exception as error:
                self.status.setText(f"Could not complete the action. {error}")
        return invoke

    def button(self, row, name, label, method):
        button = qw.QPushButton(label)
        button.clicked.connect(self.wrap(method))
        self.buttons[name] = button
        row.addWidget(button)

    def eventFilter(self, watched, event):
        event_type = event.type()
        if (event_type == qc.QEvent.Close and watched is getattr(self, "window", None)
                and not self.confirm_switch()):
            event.ignore()
            return True
        if event_type == qc.QEvent.KeyPress and watched is getattr(self, "viewer", None):
            protected = (qc.Qt.Key_Left, qc.Qt.Key_Right, qc.Qt.Key_Backspace, qc.Qt.Key_Delete)
            if event.key() in protected and any(
                marker in self.managed_markers for marker in self.overlay.get_selected_markers()
            ):
                self.status.setText("Use the review controls to edit a saved or preview pick.")
                return True
        return False

    def confirm_switch(self):
        if self._active is None or (self.history.draft is None and self.history_stamp() == self.saved_revision):
            return True
        result = qw.QMessageBox.question(
            self.window, "Unsaved review", "Discard unsaved changes?",
            qw.QMessageBox.Discard | qw.QMessageBox.Cancel, qw.QMessageBox.Cancel,
        )
        return result == qw.QMessageBox.Discard

    def history_stamp(self):
        return tuple(self.history.revisions), self.history.cursor

    def set_session(self, catalog, history, path=None):
        receivers = {receiver.role: receiver for receiver in history.current.state.receivers}
        if set(receivers) != set(ROLES):
            raise ValueError("Choose one channel for each of the two receiver roles.")
        if receivers["original"].channel == receivers["calibration"].channel:
            raise ValueError("Snuffler needs distinct trace IDs for the two receiver tracks.")
        segments = tuple(
            segment for role in ROLES for segment in catalog.segments
            if segment.channel == receivers[role].channel
            and receivers[role].source_hash in ("", segment.source_hash)
        )
        if len({segment.channel for segment in segments}) != 2:
            raise ValueError("Both receiver channels need recorded coverage.")
        segment_ids = {segment.segment_id for segment in segments}
        if any(a in segment_ids and b in segment_ids for a, b in catalog.overlaps):
            raise ValueError("The selected channels have overlapping source segments. Choose unambiguous recordings.")
        traces = [Trace(*_nslc(seg.channel), tmin=seg.start_ns / NS_PER_SECOND,
                        deltat=1 / seg.sampling_rate_hz, ydata=seg.data) for seg in segments]
        for marker in list(self.overlay.get_markers()):
            if marker not in self.managed_markers:
                self.viewer.remove_marker(marker)
        self.waveforms.cleanup()
        self.waveforms.add_traces(traces)
        self._active = catalog, history
        self._raw_draft = None
        self._raw_confirmation = None
        self.close_raw_review()
        self.segments, self.receivers = segments, receivers
        self.selected_blow_id = history.current.state.blows[0].blow_id if history.current.state.blows else None
        self.selected_group_id = None
        self.session_path = path
        self.saved_revision = self.history_stamp()
        self.source_label.setText(
            f"Original  {receivers['original'].channel}   ·   Calibration  {receivers['calibration'].channel}\n"
            "Waveform navigation view; processing not recorded."
        )
        self.source_label.setToolTip(history.current.state.timing_evidence or "Receiver timing evidence not recorded.")
        self.reason.clear()
        self.refresh()
        self.overview()

    def open_example(self):
        if not self.confirm_switch():
            return
        fixture = self.fixture
        default = Path(__file__).resolve().parents[1] / "fixtures" / "synthetic"
        if fixture is None and not (default / "expected.json").is_file():
            fixture = qw.QFileDialog.getExistingDirectory(self.window, "Open synthetic fixture directory")
            if not fixture:
                return
        self.set_session(*load_example(fixture))

    def open_pair(self):
        if not self.confirm_switch():
            return
        sources, receivers = [], []
        for role in ROLES:
            name, _ = qw.QFileDialog.getOpenFileName(self.window, f"Open {role} MiniSEED", "", MINISEED_FILTER)
            if not name:
                return
            source = load_source(name)
            ids = sorted({segment.channel for segment in source.segments})
            channel = ids[0]
            if len(ids) > 1:
                channel, accepted = qw.QInputDialog.getItem(
                    self.window, "Receiver channel", f"Select {role} channel", ids, editable=False,
                )
                if not accepted:
                    return
            sources.append(source)
            receivers.append(Receiver(role, channel, source.sha256))
        evidence, accepted = qw.QInputDialog.getMultiLineText(
            self.window, "Receiver timing", "Clock alignment evidence (optional)",
        )
        if not accepted:
            return
        state = ReviewState(blows=(), receivers=tuple(receivers), timing_evidence=evidence.strip())
        self.set_session(SourceCatalog(tuple(sources)), ReviewHistory(state))

    def open_review(self):
        from .session import SessionSourceError, load_session

        if not self.confirm_switch():
            return
        name, _ = qw.QFileDialog.getOpenFileName(self.window, "Open review", "", SESSION_FILTER)
        if not name:
            return
        path = Path(name)
        relink: dict[str, Path] = {}
        while True:
            try:
                self.set_session(*load_session(path, relink=relink), path=path)
                return
            except SessionSourceError as error:
                if error.expected_hash in relink:
                    raise
                selected, _ = qw.QFileDialog.getOpenFileName(
                    self.window, f"Find matching recording for {Path(error.path).name}", "", MINISEED_FILTER,
                )
                if not selected:
                    return
                relink[error.expected_hash] = Path(selected)

    def save_review(self):
        from .session import save_session

        default = str(self.session_path) if self.session_path else "review.json"
        name, _ = qw.QFileDialog.getSaveFileName(self.window, "Save review", default, SESSION_FILTER)
        if name:
            path = Path(name)
            save_session(path, self.catalog, self.history)
            self.session_path = path
            self.saved_revision = self.history_stamp()
            self.update_controls()
            self.status.setText(f"Saved {path.name}")

    def export_review(self):
        from .session import export_csv

        self._export_csv(export_csv, "Export applied review", "review.csv")

    def export_evidence_review(self):
        from .session import export_evidence_csv

        self._export_csv(export_evidence_csv, "Export applied pick evidence", "review-evidence.csv")

    def _export_csv(self, writer, title: str, default_name: str) -> None:
        if self.history.draft is not None:
            raise ValueError("Apply or cancel the preview before exporting.")
        name, _ = qw.QFileDialog.getSaveFileName(self.window, title, default_name, "CSV (*.csv)")
        if name:
            path = Path(name)
            writer(path, self.catalog, self.history, session_path=self.session_path)
            self.status.setText(f"Exported {path.name}")

    def overview(self):
        start = min(segment.start_ns for segment in self.segments) / NS_PER_SECOND
        stop = max(segment.end_ns for segment in self.segments) / NS_PER_SECOND
        pad = max((stop - start) * 0.02, 0.001)
        self.viewer.set_time_range(start - pad, stop + pad)
        self.viewer.update()

    def state(self):
        return self.history.current.state

    def selected_blow(self):
        return next((blow for blow in self.state().blows if blow.blow_id == self.selected_blow_id), None)

    def pick_time(self, pick):
        if pick.reviewed_ref is not None:
            return self.catalog.time_ns(pick.reviewed_ref)
        return pick.imported_ns

    def render_markers(self):
        self.overlay.cleanup()
        self.managed_markers = {}
        if self._active is None:
            return
        applied = self.state()
        draft = self.history.draft
        state = draft.state if draft else applied
        previous = {blow.blow_id: blow for blow in applied.blows}
        for blow in state.blows:
            for role in ROLES:
                receiver = self.receivers[role]
                pick = getattr(blow, role)
                timestamp = self.pick_time(pick)
                if timestamp is None:
                    continue
                if pick.imported_ns is not None and pick.imported_ns != timestamp:
                    self.managed_markers[_marker(receiver.channel, pick.imported_ns, kind=0)] = blow.blow_id
                prior = previous.get(blow.blow_id)
                changed = draft is not None and (prior is None or getattr(prior, role) != pick)
                kind = 4 if changed else 2 if blow.blow_id == self.selected_blow_id else 1
                self.managed_markers[_marker(receiver.channel, timestamp, kind=kind)] = blow.blow_id
        for span in state.masks:
            segment = self.catalog.segment(span.segment_id)
            start_ns = segment.time_ns(span.start)
            if span.stop == segment.npts:
                stop_ns = segment.end_ns + round(NS_PER_SECOND / segment.sampling_rate_hz)
            else:
                stop_ns = segment.time_ns(span.stop)
            kind = 4 if draft is not None and span not in applied.masks else 3
            self.managed_markers[_marker(segment.channel, start_ns, stop_ns, kind=kind)] = None
        if self.managed_markers:
            self.overlay.add_markers(list(self.managed_markers))
        self.viewer.update()

    def draw_results(self):
        axes = self.axes
        axes.clear()
        axes.set_xlabel("Arrival time difference (ms) · Original − Calibration")
        axes.set_ylabel("Distance difference (m)")
        axes.spines[["top", "right"]].set_visible(False)
        axes.grid(axis="y", color="#e2e8f0", linewidth=0.8)
        self.plot_points = None
        if self._active is not None:
            rows = [(row, row.lag_ns / 1e6, row.dx_m) for row in self.result.reviewed.observations
                    if row.lag_ns is not None and row.dx_m is not None]
            self.plot_blow_ids = tuple(row.blow_id for row, _, _ in rows)
            positions = [x for _, x, _ in rows]
            heights = [y for _, _, y in rows]
            for fit, label, color, linestyle in (
                (self.result.baseline_all, "Imported all", "#5f6873", ":"),
                (self.result.baseline_strict, "Imported strict", "#49525f", "--"),
                (self.result.reviewed, "Reviewed", "#08735e", "-"),
            ):
                included = [row.lag_ns / 1e6 for row in fit.observations if row.effective_included and row.lag_ns is not None]
                if fit.slope_m_per_s is not None and fit.intercept_m is not None and included:
                    domain = [min(included), max(included)]
                    axes.plot(domain, [fit.intercept_m + x / 1000 * fit.slope_m_per_s for x in domain],
                              label=label, color=color, linestyle=linestyle)
            if rows:
                self.plot_points = axes.scatter(
                    positions, heights,
                    c=["#08735e" if row.effective_included else "white" for row, _, _ in rows],
                    edgecolors="#49525f", s=28, picker=6, zorder=5,
                )
                if self.selected_blow_id in self.plot_blow_ids:
                    index = self.plot_blow_ids.index(self.selected_blow_id)
                    axes.scatter(positions[index], heights[index], facecolors="none", edgecolors="#603db0",
                                 linewidths=2, s=110, zorder=6)
            if axes.lines:
                axes.legend(loc="best", frameon=False, fontsize=9)
        self.canvas.draw_idle()

    def fill_rows(self):
        signals = self.table.blockSignals(True)
        self.table.setRowCount(len(self.state().blows))
        observations = {row.blow_id: row for row in self.result.reviewed.observations}
        for index, blow in enumerate(self.state().blows):
            row = observations[blow.blow_id]
            picks = tuple(self.pick_time(getattr(blow, role)) for role in ROLES)
            texts = (
                blow.blow_id,
                ("Included" if row.effective_included else "Geometry" if row.status == "invalid_geometry"
                 else row.status.capitalize()),
                f"{row.lag_ns / 1e6:.3f}" if row.lag_ns is not None else NOT_AVAILABLE,
                *(_utc(ns)[11:23] if ns is not None else NOT_AVAILABLE for ns in picks),
                f"{row.dx_m:.2f}" if row.dx_m is not None else NOT_AVAILABLE,
                f"{row.residual_m:.2f}" if row.residual_m is not None else NOT_AVAILABLE,
            )
            for column, text in enumerate(texts):
                item = qw.QTableWidgetItem(text)
                item.setToolTip(row.reason)
                if column in (3, 4):
                    timestamp = picks[column - 3]
                    if timestamp is not None:
                        item.setToolTip(_utc(timestamp))
                if column in (2, 5, 6):
                    item.setTextAlignment(qc.Qt.AlignRight | qc.Qt.AlignVCenter)
                self.table.setItem(index, column, item)
            if blow.blow_id == self.selected_blow_id:
                self.table.selectRow(index)
        self.table.blockSignals(signals)

    def group_state(self) -> ReviewState:
        draft = self.history.draft
        return draft.state if draft else self.state()

    def fill_group_rows(self):
        groups = sorted(self.group_state().groups, key=lambda group: (group.start_ns, group.group_id))
        applied = {group.group_id: group for group in self.state().groups}
        signals = self.group_table.blockSignals(True)
        self.group_table.clearContents()
        self.group_table.setRowCount(len(groups))
        selected = None
        for row, group in enumerate(groups):
            status = "Reviewed" if group.reviewed else "Unresolved"
            if group != applied.get(group.group_id):
                status = f"Preview\n{status}"
            window = f"{_utc(group.start_ns)}\n{_utc(group.stop_ns)}"
            for column, label in enumerate((group.group_id, status, window)):
                self.group_table.setItem(row, column, qw.QTableWidgetItem(label))
            if group.group_id == self.selected_group_id:
                selected = row
        if selected is None:
            self.selected_group_id = None
        else:
            self.group_table.setCurrentCell(selected, 0)
        self.group_table.blockSignals(signals)
        self.describe_group()

    def selected_group(self) -> GroupWindow | None:
        return next((group for group in self.group_state().groups if group.group_id == self.selected_group_id), None)

    def describe_group(self):
        group = self.selected_group()
        can_edit = group is not None and self.history.draft is None
        self.buttons["review_group"].setEnabled(can_edit)
        self.buttons["add_group_blow"].setEnabled(can_edit)
        self.buttons["review_group"].setText("Reopen group" if group and group.reviewed else "Mark reviewed")
        if group is None:
            self.group_details.clear()
            return
        state = self.group_state()
        report = inspect_group(self.catalog, state, group)
        source_stop = group.end_utc if group.end_utc is not None else f"{group.duration_s} s"
        lines = [
            f"{group.group_id} · Source position {group.source_position_id or 'Missing'}",
            f"CSV time  {group.start_utc} → {source_stop} · {group.time_basis}",
        ]
        for role in ROLES:
            receiver = getattr(report, role)
            lines.append(
                f"{role.capitalize()} · Header {receiver.channel_id or 'Missing'} · "
                f"Declared {receiver.declared_channel or 'Missing'} · {receiver.channel_status.capitalize()}"
            )
            lines.append(f"  Recording coverage  {receiver.coverage.capitalize()}")
            if receiver.sha256:
                lines.append(f"  SHA-256  {receiver.sha256}")
        timing_note = f" · {group.timing_note}" if group.timing_note else ""
        lines.append(f"Timing evidence  {report.timing_status.capitalize()}{timing_note}")
        if state.timing_evidence:
            lines.append(f"  Session clock note  {state.timing_evidence}")
        geometry_note = f" · {group.geometry_note}" if group.geometry_note else ""
        lines.extend((
            f"Geometry evidence  {report.geometry_status.capitalize()}{geometry_note}",
            f"CSV  {group.source_file} · row {group.source_row}",
            f"SHA-256  {group.source_sha256}",
        ))
        self.group_details.setPlainText("\n".join(lines))

    def group_selected(self):
        rows = self.group_table.selectionModel().selectedRows()
        self.selected_group_id = self.group_table.item(rows[0].row(), 0).text() if rows else None
        self.describe_group()
        self.zoom_to_group()

    def zoom_to_group(self, *_):
        group = self.selected_group()
        if group is not None:
            self.show_group_window(group)

    def show_group_window(self, group: GroupWindow):
        start, stop = group.start_ns / NS_PER_SECOND, group.stop_ns / NS_PER_SECOND
        lower, upper, _, _ = util.get_working_system_time_range()
        if not (lower <= start < upper and lower < stop <= upper):
            self.status.setText(f"{group.group_id} is outside the waveform viewer's UTC range.")
            return
        minimum_span = max(MIN_WAVEFORM_VIEW_S, self.viewer.content_deltat_range()[0])
        padded = stop - start < minimum_span
        if padded:
            # Snuffler's sub-millisecond ticks require high-precision floats on some platforms.
            start = min(max((start + stop - minimum_span) / 2, lower), upper - minimum_span)
            stop = start + minimum_span
        self.viewer.set_time_range(start, stop)
        self.viewer.update()
        self.update_controls()
        if padded:
            self.status.setText(f"{self.status.text()} · Waveform padded for display.")

    def import_groups(self):
        name, _ = qw.QFileDialog.getOpenFileName(self.window, "Import hammer group windows", "", "CSV (*.csv)")
        if not name:
            return
        groups = load_group_csv(Path(name))
        state = add_groups(self.state(), groups)
        self.selected_group_id = min(groups, key=lambda group: (group.start_ns, group.group_id)).group_id
        count = len(groups)
        self.preview(state, f"Import {count} hammer {'group' if count == 1 else 'groups'}")
        self.zoom_to_group()

    def review_group(self):
        group = self.selected_group()
        if group is None:
            return
        state = set_group_reviewed(self.state(), group.group_id, not group.reviewed)
        reason = f"Reopen {group.group_id}" if group.reviewed else f"Mark {group.group_id} reviewed"
        self.preview(state, reason)

    def add_group_blow(self):
        group = self.selected_group()
        if group is not None:
            self.new_blow(group_id=group.group_id)

    def unresolved(self, step):
        if self.unresolved_filter.currentData() == "blows":
            if self._raw_draft is not None:
                self.status.setText(f"Preview • {self._raw_draft.reason}. Apply or Cancel before changing picks.")
                return
            self.review_tabs.setCurrentIndex(0)
            self.unresolved_blow(step)
            return

        self.review_tabs.setCurrentIndex(1)
        ordered = [self.group_table.item(row, 0).text() for row in range(self.group_table.rowCount())]
        unresolved = {group.group_id for group in self.group_state().groups if not group.reviewed}
        if not unresolved:
            self.status.setText("No unresolved hammer groups.")
            return
        selected = self.selected_group_id
        order = ordered if step > 0 else ordered[::-1]
        start = order.index(selected) + 1 if selected in order else 0
        target = next(item for item in order[start:] + order[:start] if item in unresolved)
        self.group_table.setCurrentCell(ordered.index(target), 0)
        self.zoom_to_group()
        if target == selected:
            self.status.setText(f"{self.status.text()} · {target} is the only unresolved entry.")

    def unresolved_blow(self, step):
        observations = self.result.reviewed.observations
        order = [(row.blow_id, role) for row in observations for role in ROLES]
        if step < 0:
            order.reverse()
        selected = self.selected_blow_id, self.role.currentData()
        if selected in order:
            start = order.index(selected) + 1
        else:
            start = next((index for index, (blow_id, _) in enumerate(order) if blow_id == selected[0]), 0)
        numeric = {
            row.blow_id: self._fit_issue_roles(self.state().blow(row.blow_id))
            for row in observations if row.status not in ("valid", "excluded")
        }
        for blow_id, role in order[start:] + order[:start]:
            decision = self._pick_decision((blow_id, role))
            if (role in numeric.get(blow_id, ()) or decision is None or decision.status != "accepted"
                    or not self.history.replay_pick_evidence(self.catalog, blow_id, role).attestation_current):
                self.select_blow(blow_id, role=role)
                if (blow_id, role) == selected:
                    self.status.setText(f"{self.status.text()} · {blow_id} {role.capitalize()} is the only unresolved pick.")
                return
        self.status.setText("No unresolved blows.")

    def _fit_issue_roles(self, blow: Blow) -> tuple[Role, ...]:
        if blow.pair_status != "confirmed":
            return ("original",)
        issues = []
        for role in ROLES:
            receiver = self.receivers[role]
            ref = getattr(blow, role).reviewed_ref
            if (ref is None or ref.channel != receiver.channel
                    or (receiver.source_hash and ref.source_hash != receiver.source_hash)
                    or self.state().is_masked(ref)):
                issues.append(role)
                continue
            try:
                self.catalog.segment(ref)
            except ValueError:
                issues.append(role)
        return tuple(issues) or ("original",)

    def update_controls(self):
        if self._active is None:
            for button in self.buttons.values():
                button.setEnabled(False)
            self.save_action.setEnabled(False)
            self.export_action.setEnabled(False)
            self.export_evidence_action.setEnabled(False)
            return
        draft = self.history.draft
        pending = draft is not None
        selected = self.selected_blow_id is not None
        evidence_pending = self._raw_draft is not None
        if self._evidence_target is not None:
            if evidence_pending:
                with qc.QSignalBlocker(self.role):
                    self.role.setCurrentIndex(self.role.findData(self._evidence_target[1]))
            elif self._evidence_target != (self.selected_blow_id, self.role.currentData()):
                self.close_raw_review()
        self.role.setEnabled(not evidence_pending)
        self.table.setEnabled(not evidence_pending)
        self.buttons["raw_review"].setEnabled(evidence_pending or (
            selected and not pending and self.role.currentData() in ROLES
        ))
        self.buttons["mark_unresolved"].setEnabled(selected and not pending and self.role.currentData() in ROLES)
        can_edit_evidence = self._evidence_target is not None and (not pending or evidence_pending)
        for key in ("show_raw_span", "confirm_raw_review"):
            self.buttons[key].setEnabled(can_edit_evidence)
        for key in ("earlier", "later", "point", "inclusion", "geometry", "pairing"):
            self.buttons[key].setEnabled(selected and not pending)
        for key in ("mask", "restore", "add"):
            self.buttons[key].setEnabled(not pending)
        for key in ("previous_unresolved", "next_unresolved"):
            self.buttons[key].setEnabled(True)
        self.buttons["import_groups"].setEnabled(not pending)
        self.buttons["overview"].setEnabled(True)
        self.buttons["apply"].setEnabled(pending)
        self.buttons["cancel"].setEnabled(pending)
        self.buttons["undo"].setEnabled(not pending and self.history.can_undo)
        self.buttons["redo"].setEnabled(not pending and self.history.can_redo)
        self.save_action.setEnabled(not pending)
        self.export_action.setEnabled(not pending)
        self.export_evidence_action.setEnabled(not pending)
        if draft is not None:
            suffix = "Apply or Cancel." if evidence_pending else "Blow table and fit show the applied revision."
            self.status.setText(f"Preview • {draft.reason}. {suffix}")
        else:
            saved = " · unsaved changes" if self.history_stamp() != self.saved_revision else ""
            self.status.setText(f"Applied revision {self.history.current.revision_id}{saved}")

    def refresh(self):
        if self.selected_blow() is None:
            blows = self.state().blows
            self.selected_blow_id = blows[0].blow_id if blows else None
        self.result = analyze(self.catalog, self.history.revisions[0].state, self.state())
        fit = self.result.reviewed
        fit_text = fit.diagnostic
        if fit.slope_m_per_s is not None and fit.intercept_m is not None:
            fit_text = f"Slope {fit.slope_m_per_s:.2f} m/s  ·  Intercept {fit.intercept_m:.2f} m  ·  n = {len(fit.blow_ids)}"
        self.fit_label.setText(fit_text)
        self.fit_label.setToolTip(self.result.field_interpretation + " " + fit.diagnostic)
        self.fill_rows()
        self.fill_group_rows()
        self.draw_results()
        self.render_markers()
        self.describe_selection()
        self.update_controls()
        self.describe_evidence()

    def describe_selection(self):
        blow = self.selected_blow()
        if blow is None:
            self.details.setText("Select a blow in the table or on the fit plot.")
            self.details.setToolTip("")
            return
        row = next(row for row in self.result.reviewed.observations if row.blow_id == blow.blow_id)
        if row.lag_ns is None:
            lag = NOT_AVAILABLE
        else:
            whole_ms, fraction_ns = divmod(abs(row.lag_ns), 1_000_000)
            lag = f"{'-' if row.lag_ns < 0 else ''}{whole_ms}.{fraction_ns:06d} ms"
        residual = f"{row.residual_m:.3f} m" if row.residual_m is not None else NOT_AVAILABLE
        requested = "yes" if row.requested_included else "no"
        effective = "yes" if row.effective_included else "no"
        details = [f"{blow.blow_id} · Observed lag {lag} · Residual {residual}",
                   f"Fit requested {requested} · effective {effective} · {row.status.replace('_', ' ')}"]
        dispositions = []
        for role in ROLES:
            decision = self._pick_decision((blow.blow_id, role))
            if decision is None:
                disposition = "legacy-unassessed"
            elif decision.evidence_id is None:
                disposition = f"{decision.status} · Evidence unavailable"
            else:
                replay = self.history.replay_pick_evidence(self.catalog, blow.blow_id, role)
                status = "Attestation current" if replay.attestation_current else "Attestation stale or unavailable"
                disposition = f"{decision.status} · {status}"
            dispositions.append(f"{role.capitalize()} {disposition}")
        details.append(" · ".join(dispositions))
        if blow.group_id is not None:
            group = next(group for group in self.state().groups if group.group_id == blow.group_id)
            details.append(f"Group {group.group_id}")
            if any((time := self.pick_time(getattr(blow, role))) is not None
                   and not (group.start_ns <= time < group.stop_ns) for role in ROLES):
                details.append("An arrival is outside the group window.")
        self.details.setText("\n".join(details))
        self.details.setToolTip(row.reason)

    def select_blow(self, blow_id, *, zoom=True, role: Role | None = None):
        if self._raw_draft is not None and blow_id != self.selected_blow_id:
            return
        if not any(blow.blow_id == blow_id for blow in self.state().blows):
            return
        self.selected_blow_id = blow_id
        if role is not None:
            with qc.QSignalBlocker(self.role):
                self.role.setCurrentIndex(self.role.findData(role))
        self.fill_rows()
        self.draw_results()
        self.render_markers()
        self.describe_selection()
        self.update_controls()
        if zoom:
            self.zoom_to_selected()

    def zoom_to_selected(self):
        blow = self.selected_blow()
        if blow is None:
            return
        times = [self.pick_time(getattr(blow, role)) for role in ROLES]
        positions = [ns / NS_PER_SECOND for ns in times if ns is not None]
        if positions:
            pad = max(MIN_WAVEFORM_VIEW_S / 2, 4 / min(segment.sampling_rate_hz for segment in self.segments))
            self.viewer.set_time_range(min(positions) - pad, max(positions) + pad)
            self.viewer.update()
        elif blow.group_id is not None:
            group = next(group for group in self.state().groups if group.group_id == blow.group_id)
            self.show_group_window(group)
        else:
            self.status.setText(f"{blow.blow_id} has no arrival UTC for waveform navigation.")

    def table_selected(self):
        items = self.table.selectedItems()
        if items:
            blow_id = self.table.item(items[0].row(), 0).text()
            if blow_id != self.selected_blow_id:
                if self._raw_draft is None:
                    self.select_blow(blow_id)
                else:
                    self.fill_rows()

    def plot_selected(self, event):
        if event.artist is self.plot_points and len(event.ind) == 1:
            self.select_blow(self.plot_blow_ids[event.ind[0]])

    def link_marker_selection(self, _selection):
        selected = self.overlay.get_selected_markers()
        if len(selected) == 1:
            blow_id = self.managed_markers.get(selected[0])
            if blow_id and blow_id != self.selected_blow_id:
                # Snuffler is still notifying its own marker table about this selection.
                qc.QTimer.singleShot(0, lambda: self.select_blow(blow_id, zoom=False))

    def selected_marker(self, *, interval):
        selected = self.overlay.get_selected_markers()
        if len(selected) != 1 or selected[0] in self.managed_markers:
            raise ValueError("Select one new waveform marker.")
        marker = selected[0]
        if (marker.get_tmin() < marker.get_tmax()) != interval:
            raise ValueError("Select an interval marker." if interval else "Select a point marker.")
        return marker

    def role_for_pick(self):
        role = self.role.currentData()
        if role not in ROLES:
            raise ValueError("Select Original or Calibration for a pick edit.")
        return role

    def preview(self, state, reason, *, use_edit_reason=True):
        if self._raw_draft is not None:
            raise ValueError("Apply or cancel the raw review preview first.")
        entered = self.reason.text().strip() if use_edit_reason else ""
        self.history.preview(state, entered or reason)
        self.render_markers()
        if state.groups != self.state().groups:
            self.fill_group_rows()
        else:
            self.describe_group()
        self.update_controls()

    def apply(self):
        if self._raw_draft is None:
            self.history.apply()
        else:
            view = self.displayed_raw_evidence()
            decision = self._raw_review_decision(view)
            if (self.history.draft is not self._raw_draft
                    or decision not in self._raw_draft.state.pick_decisions):
                raise ValueError("Confirm the edited raw review before Apply.")
            if view is not None:
                if self._raw_confirmation is None:
                    raise ValueError("Confirm the current raw view before Apply.")
                self.history.apply_with_evidence(self.catalog, self._raw_confirmation, view,
                                                 captured_utc_ns=time_ns(), software_versions=_capture_versions())
            else:
                self.history.apply()
            self._raw_draft = None
            self._raw_confirmation = None
            self.evidence_notice.setText("Review applied.")
        self.reason.clear()
        self.refresh()

    def cancel(self):
        self.history.cancel()
        if self._raw_draft is not None:
            self._raw_draft = None
            self._raw_confirmation = None
            self.evidence_notice.setText("Review preview cancelled.")
        self.refresh()

    def navigate(self, move):
        move()
        self.refresh()

    def nudge(self, step):
        role = self.role_for_pick()
        blow_id = self.require_blow().blow_id
        self.preview(nudge_pick(self.state(), self.catalog, blow_id, role, step),
                     f"{blow_id} {role} pick nudged {step:+} sample")

    def require_blow(self) -> Blow:
        blow = self.selected_blow()
        if blow is None:
            raise ValueError("Select a blow first.")
        return blow

    def use_point(self):
        role = self.role_for_pick()
        blow_id = self.require_blow().blow_id
        receiver = self.receivers[role]
        marker = self.selected_marker(interval=False)
        if set(marker.get_nslc_ids()) != {_nslc(receiver.channel)}:
            raise ValueError(f"Mark a point on the {role} receiver's track.")
        seen_ns = _marker_ns(marker.get_tmin())
        lookup = self.catalog.lookup(receiver.channel, seen_ns, source_hash=receiver.source_hash or None)
        if lookup.status not in ("exact", "off_grid"):
            raise ValueError(f"The selected UTC time is {lookup.status} on the {role} receiver.")
        options = sorted((abs(self.catalog.time_ns(ref) - seen_ns), ref.sample_index, ref) for ref in lookup.candidates)
        if len(options) == 2 and options[0][0] == options[1][0]:
            raise ValueError("The selected UTC time is equally close to two samples. Zoom in and mark again.")
        _, index, ref = options[0]
        if lookup.status == "off_grid":
            accepted = qw.QMessageBox.question(
                self.window, "Snap to source sample",
                f"Map {_utc(seen_ns)} to {role} sample {index} at {_utc(self.catalog.time_ns(ref))}?",
                qw.QMessageBox.Ok | qw.QMessageBox.Cancel, qw.QMessageBox.Cancel,
            )
            if accepted != qw.QMessageBox.Ok:
                return
        self.preview(move_pick(self.state(), self.catalog, blow_id, role, ref),
                     f"{blow_id} {role} pick set to source sample {index}")
        self.viewer.remove_marker(marker)

    def edit_mask(self, *, restore):
        marker = self.selected_marker(interval=True)
        scope = self.role.currentData()
        if scope != "both" and set(marker.get_nslc_ids()) != {_nslc(self.receivers[scope].channel)}:
            raise ValueError(f"Mark an interval on the {scope.capitalize()} receiver's track or choose Both.")
        roles = ROLES if scope == "both" else (scope,)
        start, stop = (_marker_ns(t) for t in (marker.get_tmin(), marker.get_tmax()))
        spans = tuple(
            span for role in roles
            for span in self.catalog.spans(
                [self.receivers[role].channel], start, stop,
                source_hash=self.receivers[role].source_hash or None,
            )
        )
        if not spans:
            raise ValueError("The selection contains no samples on the chosen receiver scope.")
        edit = restore_spans if restore else mask_spans
        entered = self.reason.text().strip()
        require_reason(entered, "A restore" if restore else "A mask")
        state = edit(self.state(), spans, catalog=self.catalog, reason=entered)
        self.preview(state, f"{'Restored' if restore else 'Masked'} interval on {scope}", use_edit_reason=False)
        self.viewer.remove_marker(marker)

    def toggle_inclusion(self):
        blow = self.require_blow()
        included = not blow.included
        reason = self.reason.text().strip()
        if not included:
            require_reason(reason, "A fit exclusion")
        state = set_inclusion(self.state(), blow.blow_id, included, reason or "Included by reviewer")
        summary = f"Included {blow.blow_id} in fit" if included else f"Excluded {blow.blow_id} from fit"
        self.preview(state, summary, use_edit_reason=False)

    def new_blow(self, *, group_id=None):
        title = "Add blow" if group_id is None else f"Add blow to group {group_id}"
        label, accepted = qw.QInputDialog.getText(self.window, title, "Blow ID")
        if accepted and label.strip():
            blow = Blow(label.strip(), Pick(None), Pick(None), None, None,
                        included=False, pair_status="unmatched", group_id=group_id)
            self.preview(add_blow(self.state(), blow), f"Added {blow.blow_id}")

    def change_geometry(self):
        blow = self.require_blow()
        distances = []
        for role in ROLES:
            previous = getattr(blow, f"distance_{role}_m")
            value, accepted = qw.QInputDialog.getText(
                self.window, "Source-to-receiver distance", f"{role.capitalize()} distance (m)",
                text="" if previous is None else str(previous),
            )
            if not accepted:
                return
            distances.append(float(value) if value.strip() else None)
        self.preview(set_geometry(self.state(), blow.blow_id, *distances), f"{blow.blow_id} distances updated")

    def change_pairing(self):
        blow = self.require_blow()
        choices = ("confirmed", "ambiguous", "unmatched")
        status, accepted = qw.QInputDialog.getItem(
            self.window, "Blow association", "Pair status", choices,
            current=choices.index(blow.pair_status), editable=False,
        )
        if accepted:
            reason = self.reason.text().strip() or f"Marked {status} by reviewer"
            self.preview(set_pair_status(self.state(), blow.blow_id, status, reason), f"{blow.blow_id} pairing updated")


def launch_workbench(initial=None, *, fixture=None, check=None):
    """Start the stock Snuffler window with a peer Qt review panel."""
    errors = []

    def attach(window):
        workbench = Workbench(window, initial, fixture=fixture)
        if check is not None:
            def exercise():
                try:
                    check(workbench)
                except Exception as error:
                    errors.append(error)
                finally:
                    window.removeEventFilter(workbench)
                    window.close()
            qc.QTimer.singleShot(0, exercise)

    # Disable the upstream welcome/telemetry wizard in memory for this launcher.
    settings = config.raw_config("snuffler")
    settings.first_start = False
    settings.demean = False
    snuffle(pile.Pile(), ntracks=2, controls=False, opengl=False, instant_close=True, launch_hook=attach)
    if errors:
        raise errors[0]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    start = parser.add_mutually_exclusive_group()
    start.add_argument("--example", nargs="?", const="", metavar="FIXTURE_DIRECTORY")
    start.add_argument("--session", type=Path, metavar="REVIEW_JSON")
    args = parser.parse_args(argv)
    fixture = Path(args.example) if args.example else None
    if args.example is not None:
        initial = load_example(fixture)
    elif args.session is not None:
        from .session import load_session

        initial = load_session(args.session)
    else:
        initial = None
    launch_workbench(initial, fixture=fixture)


if __name__ == "__main__":
    main()
