"""App-owned, raw-only sample inspection for evidence review."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
from PyQt5.QtCore import QSignalBlocker, Qt
from PyQt5.QtGui import QResizeEvent
from PyQt5.QtWidgets import QHBoxLayout, QLabel, QSpinBox, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from .sources import NS_PER_SECOND, SampleRef, SampleSpan, Segment


MAX_RAW_SAMPLES = 20_000
RAW_ORDINATE_SCALE = "linear-autoscale"


@dataclass(frozen=True)
class RawViewIdentity:
    """The selected samples, indicators and integer UTC display limits.

    The display starts at the first sample's rounded UTC time. Its exclusive
    end is the next sample's rounded time, or Segment.stop_ns at segment end.
    The sample span and the segment's sampling grid define the exact support.
    """

    source_hash: str
    channel: str
    span: SampleSpan
    display_start_ns: int
    display_stop_ns: int
    point: SampleRef | None
    bounds: tuple[SampleRef, SampleRef] | None


def _utc(nanoseconds: int) -> str:
    seconds, remainder = divmod(nanoseconds, NS_PER_SECOND)
    try:
        stamp = datetime(1970, 1, 1) + timedelta(seconds=seconds)
    except OverflowError:
        return f"UTC {nanoseconds} ns since Unix epoch"
    return f"{stamp.isoformat()}.{remainder:09d}Z"


class RawView(QWidget):
    """Render one original span and inspect its samples with a native spin box."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAccessibleName("Raw evidence inspector")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.span_label = QLabel()
        self.span_label.setTextFormat(Qt.PlainText)
        self.span_label.setWordWrap(True)
        layout.addWidget(self.span_label)

        figure = Figure(figsize=(5.5, 1.35), layout="constrained")
        self.canvas = FigureCanvasQTAgg(figure)
        self.canvas.setMinimumHeight(120)
        self.canvas.setAccessibleName("Raw sample ordinates by elapsed time")
        self.axes = figure.add_subplot()
        layout.addWidget(self.canvas, stretch=1)

        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        control_row = QWidget()
        controls = QHBoxLayout(control_row)
        controls.setContentsMargins(0, 0, 0, 0)
        sample_label = QLabel("&Sample in view")
        controls.addWidget(sample_label)
        self.sample_control = QSpinBox()
        self.sample_control.setAccessibleName("Sample in view")
        self.sample_control.setKeyboardTracking(False)
        sample_label.setBuddy(self.sample_control)
        controls.addWidget(self.sample_control)
        controls.addStretch()
        layout.addWidget(control_row)

        self.readout = QLabel()
        self.readout.setTextFormat(Qt.PlainText)
        self.readout.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)
        self.readout.setWordWrap(True)
        layout.addWidget(self.readout)

        self._segment: Segment | None = None
        self._selection: RawViewIdentity | None = None
        self._cursor: Line2D | None = None
        self._offsets: tuple[float, ...] = ()
        self.sample_control.valueChanged.connect(self._inspect)
        self.clear()

    @property
    def view(self) -> RawViewIdentity | None:
        """Return the displayed identity when capture is available."""
        return self._selection if self._cursor is not None else None

    def clear(self) -> None:
        self._segment = None
        self._selection = None
        self._cursor = None
        self._offsets = ()
        self.axes.clear()
        self.canvas.hide()
        self.span_label.setText("Select a raw sample span.")
        self.status.hide()
        self.sample_control.setEnabled(False)
        self.readout.clear()
        self._fit_labels()

    def resizeEvent(self, event: QResizeEvent | None) -> None:
        super().resizeEvent(event)
        self._fit_labels()

    def _fit_labels(self) -> None:
        # Qt's one-line minimum height can overlap the rows below a wrapped label.
        for label in (self.span_label, self.status, self.readout):
            label.setMinimumHeight(max(0, label.heightForWidth(self.width())))

    def show_span(
        self,
        segment: Segment,
        span: SampleSpan,
        *,
        point: SampleRef | None = None,
        bounds: tuple[SampleRef, SampleRef] | None = None,
    ) -> None:
        """Display the exact half-open span, or retain the old view on invalid input."""
        if span.segment_id != segment.segment_id or span.stop > segment.npts:
            raise ValueError("Raw inspection span is outside its source segment")
        count = span.stop - span.start
        if count > MAX_RAW_SAMPLES:
            raise ValueError(f"Narrow the raw inspection span to at most {MAX_RAW_SAMPLES:,} samples.")

        refs: tuple[SampleRef, ...] = (point,) if point is not None else ()
        if bounds is not None:
            lower, upper = bounds
            if point is None or not lower.sample_index <= point.sample_index <= upper.sample_index:
                raise ValueError("Inclusive raw bounds must contain the point")
            bounds = (lower, upper)
            refs = (lower, point, upper)
        for ref in refs:
            if not span.start <= ref.sample_index < span.stop or segment.ref(ref.sample_index) != ref:
                raise ValueError("The point and bounds must be samples from the raw inspection span")

        start_ns = segment.time_ns(span.start)
        stop_ns = segment.time_ns(span.stop) if span.stop < segment.npts else segment.stop_ns
        selection = RawViewIdentity(segment.source_hash, segment.channel, span, start_ns, stop_ns, point, bounds)
        if self._segment is segment and self.view == selection:
            return

        self._selection = selection
        self._segment = segment
        self._cursor = None
        self.axes.clear()
        label = f"{segment.channel} · Samples [{span.start}, {span.stop})"
        if point is not None:
            label += f" · Point {point.sample_index}"
        if bounds is not None:
            label += f" · Inclusive bounds {lower.sample_index}–{upper.sample_index}"
            if lower == upper:
                label += " (zero width)"
        self.span_label.setText(label)

        values = segment.data[span.start:span.stop]
        issue = self._draw(selection, segment, values)
        if issue is not None:
            self.axes.clear()
        self.canvas.setVisible(issue is None)
        self.status.setText(f"Plot and capture unavailable. {issue}" if issue else "")
        self.status.setVisible(issue is not None)

        first_shown = point.sample_index - span.start + 1 if point is not None else 1
        with QSignalBlocker(self.sample_control):
            self.sample_control.setRange(1, count)
            self.sample_control.setValue(first_shown)
        self.sample_control.setEnabled(True)
        self._inspect(first_shown)

    def _draw(self, selection: RawViewIdentity, segment: Segment, values: np.ndarray) -> str | None:
        if not np.isfinite(values).all():
            return "The span contains nonfinite raw values. Narrow the span to plot finite samples."

        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                span = selection.span
                self._offsets = tuple(
                    (segment.time_ns(index) - selection.display_start_ns) / NS_PER_SECOND
                    for index in range(span.start, span.stop)
                )
                duration = (selection.display_stop_ns - selection.display_start_ns) / NS_PER_SECOND
                self.axes.plot(self._offsets, values, linestyle="none", marker="o", markersize=3,
                               color="#16647c", clip_on=False)
                self.axes.set_xlim(0, duration)
                bottom, top = self.axes.get_ylim()
                if not bottom <= float(values.min()) <= float(values.max()) <= top:
                    return "The native linear axis cannot display this raw value range."

                if selection.bounds is not None:
                    for bound in dict.fromkeys(selection.bounds):
                        self.axes.axvline(self._offsets[bound.sample_index - span.start],
                                          color="#5f5f5f", linestyle=":", clip_on=False)
                if selection.point is not None:
                    self.axes.axvline(self._offsets[selection.point.sample_index - span.start],
                                      color="#824413", linestyle="--", clip_on=False)
                cursor, = self.axes.plot([], [], linestyle="none", marker="s", markersize=7,
                                         markerfacecolor="none", color="#242424", clip_on=False, zorder=4)
                self.axes.set_navigate(False)
                self.axes.set_xlabel("Time from span start (s)", fontsize=10)
                self.axes.set_ylabel("Raw value", fontsize=10)
                self.axes.tick_params(labelsize=10)
                self.axes.grid(axis="y", color="#e5e5e5")
                self.canvas.draw()
        except (FloatingPointError, OverflowError, ValueError):
            return "The native linear axis cannot display this raw value or time range."
        self._cursor = cursor
        return None

    def _inspect(self, position: int) -> None:
        if self._segment is None or self._selection is None:
            return
        segment = self._segment
        offset = position - 1
        index = self._selection.span.start + offset
        sample = segment.data[index:index + 1]
        value = sample[0].item()
        dtype = sample.dtype
        self.readout.setText(f"Source sample {index} · {_utc(segment.time_ns(index))}\n"
                             f"Raw {value} · {dtype.name} ({dtype.str}) · units unknown")
        self._fit_labels()
        if self._cursor is not None:
            self._cursor.set_data([self._offsets[offset]], sample)
            self.canvas.draw_idle()
