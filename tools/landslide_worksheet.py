"""Stand-alone Qt worksheet for the bounded landslide comparison exercise."""

from __future__ import annotations

from pathlib import Path

from matplotlib.axes import Axes
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import numpy as np
from obspy import UTCDateTime
from PyQt5 import QtCore, QtWidgets

from seisslate.landslide import Worksheet
from seisslate.sources import NS_PER_SECOND
from tools.landslide_pilot import STATUSES, Pack, digest_file, open_pack, reopen, save_review, snap_sample


STATUS_TEXT = {
    "unreviewed": "Unreviewed", "identifiable": "Identifiable", "ambiguous": "Ambiguous",
    "no_identifiable_signal": "No identifiable signal", "insufficient_support": "Insufficient support",
    "unavailable": "Unavailable",
}
BOUNDARY_COLORS = {"onset": "#006d77", "end": "#8b4b11"}
TABLE_STATUS_TEXT = {"no_identifiable_signal": "No signal seen", "insufficient_support": "Low support"}


class LandslideWorksheet(QtWidgets.QWidget):
    def __init__(self, pack: Pack, output: Path, review: Path | None = None, reviewer: str = ""):
        super().__init__()
        self.pack = pack
        self.worksheet = reopen(pack, review) if review else Worksheet(pack.rows)
        self.previous_review_sha256 = digest_file(review) if review else None
        self.output = output
        self.reviewer = reviewer
        self.pending_sample = None
        self._displayed_channel = None
        self._on_pick_artists: list[Line2D] = []

        self.setWindowTitle(f'{pack.manifest["case_id"]} · Observed signal duration')
        self.resize(1320, 730)
        self.figure = Figure(figsize=(8, 6), layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.canvas.mpl_connect("button_release_event", self._pick)
        self.axes: dict[str, Axes] = {}
        waveforms = QtWidgets.QWidget()
        waveform_layout = QtWidgets.QVBoxLayout(waveforms)
        waveform_layout.addWidget(self.toolbar)
        waveform_layout.addWidget(self.canvas, 1)

        self.table = QtWidgets.QTableWidget(len(pack.rows), 5)
        self.table.setHorizontalHeaderLabels(["Channel", "Band", "Onset", "End", "Duration (s)"])
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.table.currentCellChanged.connect(self._select)

        self.boundary = QtWidgets.QComboBox()
        self.boundary.addItems(["Onset", "End"])
        self.boundary.currentIndexChanged.connect(self._reset_editor)
        self.status = QtWidgets.QComboBox()
        self.status.addItems(STATUSES)
        self.status.currentTextChanged.connect(self._change_status)
        self.reason = QtWidgets.QLineEdit()
        self.sample_label = QtWidgets.QLabel()
        self.earlier_button = QtWidgets.QPushButton("−1 sample")
        self.earlier_button.setShortcut("Alt+Left")
        self.earlier_button.clicked.connect(lambda: self._nudge(-1))
        self.later_button = QtWidgets.QPushButton("+1 sample")
        self.later_button.setShortcut("Alt+Right")
        self.later_button.clicked.connect(lambda: self._nudge(1))
        nudge = QtWidgets.QHBoxLayout()
        nudge.addWidget(self.earlier_button)
        nudge.addWidget(self.later_button)
        self.support = QtWidgets.QLabel()
        self.support.setWordWrap(True)

        editor = QtWidgets.QFormLayout()
        editor.addRow("Boundary", self.boundary)
        editor.addRow("Status", self.status)
        editor.addRow("Reason", self.reason)
        editor.addRow("Sample", self.sample_label)
        editor.addRow(nudge)
        editor.addRow("Support", self.support)
        self.apply_button = QtWidgets.QPushButton("Apply")
        self.apply_button.clicked.connect(self._apply)
        self.undo_button = QtWidgets.QPushButton("Undo")
        self.undo_button.clicked.connect(self._undo)
        self.undo_button.setShortcut("Ctrl+Z")
        reset_button = QtWidgets.QPushButton("Reset preview")
        reset_button.clicked.connect(self._reset_editor)
        self.save_button = QtWidgets.QPushButton("Save review")
        self.save_button.clicked.connect(self._save)
        actions = QtWidgets.QHBoxLayout()
        for button in (self.apply_button, reset_button, self.undo_button, self.save_button):
            actions.addWidget(button)
        self.message = QtWidgets.QLabel("Select a band, then click its trace to preview a sample.")
        self.message.setWordWrap(True)
        details = QtWidgets.QWidget()
        detail_layout = QtWidgets.QVBoxLayout(details)
        detail_layout.addWidget(self.table, 1)
        detail_layout.addLayout(editor)
        detail_layout.addLayout(actions)
        detail_layout.addWidget(self.message)
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        split.addWidget(waveforms)
        split.addWidget(details)
        split.setSizes([735, 570])
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(split)

        self._refresh_table()
        self.table.setCurrentCell(0, 0)

    @property
    def row(self):
        return self.pack.rows[self.table.currentRow()]

    def _refresh_table(self):
        for index, row in enumerate(self.pack.rows):
            result = self.worksheet.result(row.channel, row.band)
            duration = "—" if result.duration_seconds is None else f"{result.duration_seconds:.9g}"
            statuses = [TABLE_STATUS_TEXT.get(part.status, STATUS_TEXT[part.status]) for part in (result.onset, result.end)]
            values = (row.channel, row.band, *statuses, duration)
            for column, value in enumerate(values):
                item = self.table.item(index, column)
                if item is None:
                    item = QtWidgets.QTableWidgetItem()
                    self.table.setItem(index, column, item)
                item.setText(value)
                if column in (2, 3):
                    boundary = result.onset if column == 2 else result.end
                    item.setToolTip(f"{STATUS_TEXT[boundary.status]}\n{boundary.reason}".rstrip())
                elif column == 4:
                    item.setToolTip(result.duration_status.replace("_", " "))
        self.table.resizeColumnsToContents()
        headers = self.table.horizontalHeader().length() + self.table.verticalHeader().width()
        self.table.setMinimumWidth(headers + 22)

    def _select(self, *_):
        self._reset_editor()
        self._plot()

    def _reset_editor(self, *_):
        row = self.row
        current = getattr(self.worksheet.result(row.channel, row.band), self.boundary.currentText().lower())
        blocked = self.status.blockSignals(True)
        self.status.setCurrentText(current.status)
        self.status.blockSignals(blocked)
        self.reason.setText(current.reason)
        self.pending_sample = current.sample
        spans = sum(waveform.support is not None for waveform in row.waveforms)
        if spans:
            detail = ("Full review window" if row.full_review_support else
                      f"Partial coverage · {spans} eligible span{'s' if spans != 1 else ''}")
        else:
            causes = {waveform.support_reason for waveform in row.waveforms if waveform.support_reason}
            detail = ", ".join(sorted(causes)).replace("_", " ") or STATUS_TEXT[row.initial_status]
        self.support.setText(detail)
        self._refresh_indicators()

    def _change_status(self, new_status):
        if new_status != "identifiable":
            self.pending_sample = None
            self._refresh_indicators()

    def _plot(self):
        channel = self.row.channel
        if channel == self._displayed_channel:
            self._refresh_indicators()
            return
        self._displayed_channel = channel
        self._on_pick_artists = []
        self.figure.clear()
        raw, hf, lp = self.figure.subplots(3, 1, sharex=True)
        self.axes = {"RW": raw, "HF": hf, "LP": lp}
        self.figure.suptitle(channel)
        rows = [row for row in self.pack.rows if row.channel == channel]

        def draw(axis, segment, first, values):
            # Subtract the local origin before converting offsets to floats.
            start = (segment.time_ns(first) - self.pack.start_ns) / NS_PER_SECOND
            offsets = start + np.arange(len(values)) * float(segment.period_ns / NS_PER_SECOND)
            axis.plot(offsets, values, color="#2a3c48", linewidth=0.85)

        for segment in rows[0].segments:
            first = segment.lower_bound(self.pack.start_ns)
            stop = segment.lower_bound(self.pack.stop_ns)
            if first < stop:
                draw(raw, segment, first, segment.data[first:stop])
        raw.set_ylabel("Raw\nRecorded units")
        processing = self.pack.manifest["processing"]
        for row in rows:
            axis = self.axes[row.band]
            for waveform in row.waveforms:
                support = waveform.support
                if support is not None:
                    start = support.start - waveform.first_index
                    stop = support.stop - waveform.first_index
                    draw(axis, waveform.segment, support.start, waveform.values[start:stop])
            band = processing["bands"][row.band]
            axis.set_ylabel(f'{row.band}\n{band["low_hz"]:g}–{band["high_hz"]:g} Hz')
            if not axis.lines:
                axis.text(0.5, 0.5, STATUS_TEXT[row.initial_status], transform=axis.transAxes,
                          horizontalalignment="center", color="#59616a")
        epoch = UTCDateTime(ns=self.pack.start_ns, precision=9)
        lp.set_xlabel(f"Seconds since {epoch}")
        lp.set_xlim(0, (self.pack.stop_ns - self.pack.start_ns) / NS_PER_SECOND)
        for axis in self.axes.values():
            axis.grid(axis="x", linewidth=0.5, alpha=0.35)
        self._refresh_indicators()
        self.canvas.draw_idle()

    def _refresh_indicators(self):
        for artist in self._on_pick_artists:
            artist.remove()
        self._on_pick_artists = []
        if self._displayed_channel != self.row.channel:
            return
        boundary_name = self.boundary.currentText().lower()
        for row in (row for row in self.pack.rows if row.channel == self.row.channel):
            axis = self.axes[row.band]
            result = self.worksheet.result(row.channel, row.band)
            for name, color in BOUNDARY_COLORS.items():
                boundary = getattr(result, name)
                if boundary.utc_ns is not None:
                    x = (boundary.utc_ns - self.pack.start_ns) / NS_PER_SECOND
                    self._on_pick_artists.append(axis.axvline(x, color=color, linewidth=1.5))
        self.sample_label.setText("—")
        self.earlier_button.setEnabled(self.pending_sample is not None)
        self.later_button.setEnabled(self.pending_sample is not None)
        if self.pending_sample is not None:
            segment = self.row.supporting_run(self.pending_sample).segment
            index = self.pending_sample.sample_index
            stamp = segment.time_ns(index)
            x = (stamp - self.pack.start_ns) / NS_PER_SECOND
            applied = getattr(self.worksheet.result(self.row.channel, self.row.band), boundary_name)
            if self.pending_sample != applied.sample:
                self._on_pick_artists.append(self.axes[self.row.band].axvline(
                    x, color=BOUNDARY_COLORS[boundary_name], linestyle="--", linewidth=1.5,
                ))
            self.sample_label.setText(f"{UTCDateTime(ns=stamp, precision=9)} · #{index}")
        self.canvas.draw_idle()

    def _pick(self, event):
        if event.button != 1 or self.toolbar.mode or event.inaxes != self.axes.get(self.row.band):
            return
        if event.xdata is None:
            return
        try:
            self.pending_sample = snap_sample(self.row, self.pack.start_ns + round(event.xdata * NS_PER_SECOND))
        except ValueError as error:
            self.pending_sample = None
            self._refresh_indicators()
            self.message.setText(str(error))
            return
        self.status.setCurrentText("identifiable")
        self.message.setText(f"{self.boundary.currentText()} preview. Apply to record the change.")
        self._refresh_indicators()

    def _nudge(self, direction):
        if self.pending_sample is None:
            return
        run = self.row.supporting_run(self.pending_sample)
        index = self.pending_sample.sample_index + direction
        if not run.support.start <= index < run.support.stop:
            self.message.setText("The next sample is outside this eligible span.")
            return
        self.pending_sample = run.segment.ref(index)
        self.message.setText(f"{self.boundary.currentText()} preview. Apply to record the change.")
        self._refresh_indicators()

    def _apply(self):
        row = self.row
        try:
            edit = self.worksheet.preview(
                row.channel, row.band, self.boundary.currentText().lower(),
                self.status.currentText(), self.reason.text(), self.pending_sample,
            )
            self.worksheet.apply(edit)
        except ValueError as error:
            self.message.setText(str(error))
            return
        self.message.setText(f"{self.boundary.currentText()} applied · {edit.result.duration_status.replace('_', ' ')}")
        self._refresh_table()
        self._reset_editor()

    def _undo(self):
        if self.worksheet.undo():
            self.message.setText("Undid the last applied edit.")
            self._refresh_table()
            self._reset_editor()

    def _save(self):
        destination = self.output
        if destination.exists():
            name, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save review as a new file", str(destination.parent), "Review JSON (*.json)"
            )
            if not name:
                return
            destination = Path(name)
        try:
            save_review(self.pack, self.worksheet, destination, method="Qt worksheet", reviewer=self.reviewer,
                        evidence={"previous_review_sha256": self.previous_review_sha256})
            self.previous_review_sha256 = digest_file(destination)
        except (OSError, ValueError) as error:
            self.message.setText(str(error))
            return
        self.message.setText(f"Saved {destination}")


def launch(pack_path: Path, output: Path, review: Path | None = None, reviewer: str = "") -> None:
    pack = open_pack(pack_path)
    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = LandslideWorksheet(pack, output, review, reviewer)
    window.show()
    application.exec_()
