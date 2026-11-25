"""The native Qt editor and raw-only view for standalone RMS review."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from fractions import Fraction
from math import ceil, floor

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from pyrocko.gui.qt_compat import qc, qw

from .rms import RmsResult, RmsReview, RmsWindow
from .sources import NS_PER_SECOND, Segment, SourceCatalog


BANDS = {
    "Event E": ("#177087", "/"),
    "Background B": ("#864d17", "\\"),
    "Mask": ("#595959", "xx"),
}
REASONS = {
    "missing_window": "Missing window",
    "wrong_source_or_channel": "Wrong source or channel",
    "cross_segment": "Windows need coverage in one segment",
    "out_of_bounds": "Requested window outside recording coverage",
    "unequal_length": "Unequal window lengths",
    "too_short": "Fewer than two samples",
    "too_large": "Window exceeds the calculation limit",
    "overlapping_windows": "Event and background overlap",
    "ambiguous_coverage": "Overlapping recording segments",
    "masked_samples": "Masked samples",
    "nonfinite_samples": "Nonfinite raw samples",
    "epoch_conflict": "Conflicting instrument epochs",
    "zero_background": "Background RMS is zero",
    "numeric_failure": "Numeric range unavailable",
}
EPOCHS = {
    "unknown": "unknown",
    "unverified": "unverified",
    "independently_verified": "independently verified",
    "conflicting": "conflicting",
}


def boundary_ns(segment: Segment, index: int) -> Fraction:
    return segment.start_ns + index * segment.period_ns


def _utc(timestamp: int | Fraction) -> str:
    whole = floor(timestamp)
    seconds, ns = divmod(whole, NS_PER_SECOND)
    try:
        stamp = datetime(1970, 1, 1) + timedelta(seconds=seconds)
    except OverflowError:
        return f"{timestamp} ns since Unix epoch"
    suffix = f" + {timestamp - whole} ns" if timestamp != whole else ""
    return f"{stamp.isoformat()}.{ns:09d}Z{suffix}"


def _label() -> qw.QLabel:
    label = qw.QLabel()
    label.setTextFormat(qc.Qt.PlainText)
    label.setWordWrap(True)
    label.setTextInteractionFlags(qc.Qt.TextSelectableByMouse | qc.Qt.TextSelectableByKeyboard)
    return label


class EndpointEditor(qw.QGroupBox):
    def __init__(self, title: str):
        super().__init__(title)
        self.segment = qw.QComboBox()
        self.segment.setAccessibleName(f"{title} segment")
        self.start = qw.QLineEdit()
        self.start.setAccessibleName(f"{title} start index")
        self.stop = qw.QLineEdit()
        self.stop.setAccessibleName(f"{title} exclusive stop index")
        self.use_marker = qw.QPushButton(f"Use waveform interval for {title[-1]}")
        self.clear_button = qw.QPushButton(f"Clear {title[-1]}")
        self.request: RmsWindow | None = None
        self._request_stamp = None
        layout = qw.QGridLayout(self)
        layout.addWidget(self.segment, 0, 0, 1, 4)
        layout.addWidget(qw.QLabel("Start index"), 1, 0)
        layout.addWidget(self.start, 1, 1)
        layout.addWidget(qw.QLabel("Stop index (exclusive)"), 1, 2)
        layout.addWidget(self.stop, 1, 3)
        layout.addWidget(self.use_marker, 2, 0, 1, 3)
        layout.addWidget(self.clear_button, 2, 3)
        self.segment.setSizeAdjustPolicy(qw.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.segment.setMinimumContentsLength(12)
        for field in (self.start, self.stop):
            field.setMinimumWidth(58)
            field.setMaximumWidth(100)

    def stamp(self):
        return self.segment.currentData(), self.start.text(), self.stop.text()

    def form_stamp(self):
        fields = self.stamp()
        return fields, self.request if fields == self._request_stamp else None

    def load_segments(self, segments: tuple[Segment, ...]):
        self.segment.clear()
        for index, segment in enumerate(segments, start=1):
            self.segment.addItem(f"Segment {index} · {_utc(segment.start_ns)} · {segment.npts:,} samples",
                                 segment.segment_id)

    def set_request(self, window: RmsWindow | None, segments: tuple[Segment, ...]):
        self.start.clear()
        self.stop.clear()
        if window is not None:
            for segment in segments:
                start = (window.start_ns - segment.start_ns) / segment.period_ns
                stop = (window.stop_ns - segment.start_ns) / segment.period_ns
                if start.denominator == stop.denominator == 1 and 0 <= start < stop <= segment.npts:
                    self.segment.setCurrentIndex(self.segment.findData(segment.segment_id))
                    self.start.setText(str(int(start)))
                    self.stop.setText(str(int(stop)))
                    break
        self.request = window
        self._request_stamp = self.stamp()

    def window(self, segments: tuple[Segment, ...]) -> RmsWindow | None:
        if self.stamp() == self._request_stamp:
            return self.request
        _, first, last = self.stamp()
        if not first.strip() and not last.strip():
            return None
        segment = next(segment for segment in segments if segment.segment_id == self.segment.currentData())
        try:
            start, stop = int(first), int(last)
            if not 0 <= start < stop <= segment.npts:
                raise ValueError()
        except ValueError:
            raise ValueError(f"{self.title()} needs integer endpoints with 0 ≤ start < stop ≤ {segment.npts}.") from None
        return RmsWindow(segment.source_hash, segment.channel, boundary_ns(segment, start), boundary_ns(segment, stop))


@dataclass(frozen=True)
class RawBand:
    label: str
    start_ns: int | Fraction
    stop_ns: int | Fraction


class RmsPanel(qw.QScrollArea):
    def __init__(self):
        super().__init__()
        self.setWidgetResizable(True)
        self.setFrameShape(qw.QFrame.NoFrame)
        content = qw.QWidget()
        self.setWidget(content)
        layout = qw.QVBoxLayout(content)

        self.applied = _label()
        self.applied.setAccessibleName("Applied RMS review")
        self.preview = _label()
        self.preview.setAccessibleName("Preview RMS review")
        figure = Figure(figsize=(5, 1.6), layout="constrained")
        self.canvas = FigureCanvasQTAgg(figure)
        self.canvas.setMinimumHeight(145)
        self.canvas.setAccessibleName("Raw waveform with labeled event, background and mask bands")
        self.axes = figure.add_subplot()
        layout.addWidget(self.canvas)
        hint = _label()
        hint.setText("Double-click the navigation trace, then drag an interval. Use it for E or B, "
                     "or enter exact sample indices. Waveform picks snap to the nearest sample boundary.")
        layout.addWidget(hint)
        self.event = EndpointEditor("Event E")
        self.background = EndpointEditor("Background B")
        layout.addWidget(self.event)
        layout.addWidget(self.background)
        disposition_row = qw.QHBoxLayout()
        self.disposition = qw.QComboBox()
        self.disposition.setAccessibleName("Window disposition")
        for value in ("candidate", "accepted", "unresolved", "rejected"):
            self.disposition.addItem(value.capitalize(), value)
        self.reviewer = qw.QLineEdit()
        self.reviewer.setPlaceholderText("Reviewer")
        self.reviewer.setAccessibleName("Window reviewer")
        disposition_row.addWidget(qw.QLabel("Window disposition"))
        disposition_row.addWidget(self.disposition)
        disposition_row.addWidget(self.reviewer, 1)
        layout.addLayout(disposition_row)

        actions = qw.QGridLayout()
        self.preview_button = qw.QPushButton("Preview &windows")
        self.mask_button = qw.QPushButton("Mask interval")
        self.restore_button = qw.QPushButton("Restore interval")
        self.overview_button = qw.QPushButton("Full waveform")
        actions.addWidget(self.preview_button, 0, 0)
        actions.addWidget(self.overview_button, 0, 1)
        actions.addWidget(self.mask_button, 1, 0)
        actions.addWidget(self.restore_button, 1, 1)
        layout.addLayout(actions)
        layout.addWidget(self.preview)
        self.provenance = qw.QPlainTextEdit()
        self.provenance.setReadOnly(True)
        self.provenance.setMaximumHeight(160)
        self.provenance.setAccessibleName("RMS source and replay details")
        detail_panel = qw.QGroupBox("Source and replay details")
        detail_layout = qw.QVBoxLayout(detail_panel)
        detail_layout.addWidget(self.provenance)
        layout.addWidget(detail_panel)
        layout.addStretch()
        self._plot_stamp = None

    def stamp(self):
        return (
            self.event.form_stamp(), self.background.form_stamp(),
            self.disposition.currentData(), self.reviewer.text(),
        )

    def load_segments(self, segments: tuple[Segment, ...]):
        for editor in (self.event, self.background):
            editor.load_segments(segments)
        self._plot_stamp = None

    def load_review(self, review: RmsReview | None, segments: tuple[Segment, ...]):
        self.event.set_request(review.windows.event if review else None, segments)
        self.background.set_request(review.windows.background if review else None, segments)
        index = self.disposition.findData(review.disposition if review else "candidate")
        self.disposition.setCurrentIndex(index)
        self.reviewer.setText(review.reviewer if review else "")

    @staticmethod
    def summary(title: str, review: RmsReview | None, result: RmsResult | None,
                catalog: SourceCatalog, *, replay_status: str = "", ready: bool = False, detail: bool = True) -> str:
        who = f" by {review.reviewer}" if review and review.reviewer else ""
        disposition = review.disposition if review else "candidate"
        reason = f" · {review.reason}" if review and review.reason else ""
        lines = [f"{title} · Window disposition {disposition}{who}{reason}"]
        if result is None:
            issue = ("unsupported recorded method" if replay_status == "unsupported_method"
                     else "no recorded computation")
            return "\n".join(lines + [f"Unavailable · {issue}", "Epoch unknown · raw units unknown",
                                      "Eligible for evidence use no"])
        epoch = EPOCHS[result.epoch_status]
        qualification = " · within-record contrast only" if result.qualified else ""
        lines.append(f"Epoch {epoch} · {result.units}{qualification}")
        if replay_status in ("numeric_mismatch", "unsupported_method"):
            issue = "replay mismatch" if replay_status == "numeric_mismatch" else "unsupported recorded method"
            lines.append(f"Unavailable · {issue}")
        elif result.available and (title != "Applied" or replay_status == "numeric_match"):
            lines.append(f"Contrast {result.contrast:.8g} (dimensionless, demeaned RMS E/B)")
        elif result.available:
            lines.append("Unavailable · no recorded computation")
        else:
            reasons = " · ".join(REASONS[reason] for reason in result.reasons)
            lines.append(f"Unavailable · {reasons}")
        if title == "Applied":
            lines.append(f"Eligible for evidence use {'yes' if ready else 'no'}")
        if detail and review is not None:
            lines.extend(RmsPanel.window_details(review, result, catalog))
        return "\n".join(lines)

    @staticmethod
    def window_details(review: RmsReview, result: RmsResult | None, catalog: SourceCatalog) -> list[str]:
        lines = []
        for label, window, measurement in (
            ("E", review.windows.event, result.event if result else None),
            ("B", review.windows.background, result.background if result else None),
        ):
            if window is None:
                continue
            span = measurement.span if measurement else None
            if span is None:
                lines.append(f"{label} · no unambiguous sample span")
            else:
                count = span.stop - span.start
                duration = count * catalog.segment(span.segment_id).period_ns / NS_PER_SECOND
                values = (f" · mean {measurement.mean:.8g} · RMS {measurement.rms:.8g}"
                          if measurement is not None and measurement.rms is not None else "")
                lines.append(f"{label} [{span.start}, {span.stop}) · N {count:,} · {float(duration):.8g} s{values}")
            lines.append(f"    UTC [{_utc(window.start_ns)}, {_utc(window.stop_ns)})")
        return lines

    def draw_raw(self, segments: tuple[Segment, ...], bands: tuple[RawBand, ...], *, draft: bool):
        stamp = bands, draft
        if self._plot_stamp == stamp:
            return
        self._plot_stamp = stamp
        axes = self.axes
        axes.clear()
        axes.set_title("Preview raw waveform" if draft else "Applied raw waveform", loc="left", fontsize=10)
        axes.set_ylabel("Raw value", fontsize=9)
        axes.tick_params(labelsize=8)
        axes.set_navigate(False)
        if not segments:
            self.canvas.draw_idle()
            return

        windows = [band for band in bands if band.label != "Mask"]
        first = min(segment.start_ns for segment in segments)
        last = max(boundary_ns(segment, segment.npts) for segment in segments)
        origin = Fraction(first)
        stop = min(last, boundary_ns(segments[0], min(segments[0].npts, 2000)))
        if windows:
            left, right = min(band.start_ns for band in windows), max(band.stop_ns for band in windows)
            margin = max(Fraction(right - left, 20), segments[0].period_ns * 2)
            if left < last and right > first:
                origin, stop = max(Fraction(first), left - margin), min(last, right + margin)

        handles = {}
        for band in bands:
            left, right = max(origin, band.start_ns), min(stop, band.stop_ns)
            if right <= left:
                continue
            color, hatch = BANDS[band.label]
            patch = axes.axvspan(float((left - origin) / NS_PER_SECOND), float((right - origin) / NS_PER_SECOND),
                                 facecolor="none", edgecolor=color, hatch=hatch, alpha=0.65, linewidth=1)
            handles[band.label] = patch

        bucketed = False
        can_draw = True
        for segment in segments:
            start = max(0, ceil((origin - segment.start_ns) / segment.period_ns))
            end = min(segment.npts, ceil((stop - segment.start_ns) / segment.period_ns))
            if end <= start:
                continue
            ordinates = segment.data[start:end]
            base = float((boundary_ns(segment, start) - origin) / NS_PER_SECOND)
            step = float(segment.period_ns / NS_PER_SECOND)
            if len(ordinates) <= 2000:
                finite = np.ma.masked_invalid(ordinates)
                if _plot_range_supported(finite.compressed()):
                    axes.plot(base + np.arange(len(ordinates)) * step, finite, color="#253039", linewidth=0.8,
                              marker=".", markersize=2)
                else:
                    can_draw = False
                continue

            # Preserve per-bin extrema. Bins with nonfinite input leave a gap in the display.
            bucketed = True
            stride = ceil(len(ordinates) / 2000)
            full = len(ordinates) // stride * stride
            blocks = ordinates[:full].reshape(-1, stride)
            lower, upper = blocks.min(axis=1), blocks.max(axis=1)
            if full < len(ordinates):
                tail = ordinates[full:]
                lower, upper = np.append(lower, tail.min()), np.append(upper, tail.max())
            coords = base + np.arange(len(lower)) * step * stride
            valid = np.isfinite(lower) & np.isfinite(upper)
            coords, lower, upper = coords[valid], lower[valid], upper[valid]
            if not _plot_range_supported(np.concatenate((lower, upper))):
                can_draw = False
                continue
            axes.vlines(coords, lower, upper, color="#253039", linewidth=0.7)
            flat = lower == upper
            axes.plot(coords[flat], lower[flat], ".", color="#253039", markersize=2)
        if handles:
            axes.legend(handles.values(), handles.keys(), loc="lower right",
                        bbox_to_anchor=(1, 1), ncols=len(handles), fontsize=7, frameon=False)
        axes.set_xlabel(f"Seconds from {_utc(origin)}", fontsize=9)
        if not can_draw:
            axes.set_title("Some raw ordinates exceed the plot range", loc="left", fontsize=10)
        elif bucketed:
            axes.set_title(axes.get_title(loc="left") + " · min/max display", loc="left", fontsize=10)
        axes.set_xlim(0, float((stop - origin) / NS_PER_SECOND))
        self.canvas.draw_idle()


def _plot_range_supported(values: np.ndarray) -> bool:
    # Leave headroom for Matplotlib's automatic limits and linear tick calculations.
    return not values.size or float(values.min()) >= -1e300 and float(values.max()) <= 1e300
