"""Standalone analyst worksheet for the bounded landslide review pilot."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import ceil
from typing import Iterable, Literal, Mapping

import numpy as np
import obspy
import scipy
from obspy.signal.filter import bandpass

from .evidence import raw_digest
from .sources import NS_PER_SECOND, SampleRef, SampleSpan, Segment, SourceCatalog


Band = Literal["HF", "LP"]
BoundaryName = Literal["onset", "end"]
Status = Literal[
    "unreviewed", "no_identifiable_signal", "ambiguous", "identifiable", "insufficient_support", "unavailable",
]
DurationStatus = Literal["eligible", "missing_boundary", "separate_support", "unordered"]
_BANDS: dict[Band, tuple[float, float, int]] = {"HF": (1.0, 5.0, 2), "LP": (0.01, 0.05, 200)}
_CORNERS = 4
_NYQUIST_RATIO_LIMIT = 1 - 1e-6
_SCHEMA = "seisslate-landslide-pilot-v1"
_STATUS_PRIORITY: tuple[Status, ...] = (
    "identifiable", "ambiguous", "unavailable", "insufficient_support", "unreviewed", "no_identifiable_signal",
)


def processing_definition() -> dict:
    """Return the filter and support policy needed to reproduce the pilot."""
    return {
        "id": "obspy-butterworth-pilot-v1",
        "obspy": obspy.__version__,
        "scipy": scipy.__version__,
        "numpy": np.__version__,
        "bands": {
            band: {"low_hz": low, "high_hz": high, "edge_guard_seconds": guard}
            for band, (low, high, guard) in _BANDS.items()
        },
        "corners": _CORNERS,
        "phase": "forward and backward",
        "initialization": "zero SOS state on each pass; no pad, demean, taper or response removal",
        "sample_rate_rule": f"upper corner / Nyquist < {_NYQUIST_RATIO_LIMIT}",
        "filter_input": "entire contiguous finite run within a source segment, in original recording units",
        "support": "drop ceil(edge_guard_seconds * sample_rate) samples from each input edge, then intersect "
        "with the half-open UTC review interval; reject overlapping source segments anywhere in their coverage",
        "guard_scope": "eligibility rule; the IIR output still depends on the full input run",
        "negative_review": "require one eligible run covering the entire explicit UTC review interval on its grid",
    }


@dataclass(frozen=True)
class Waveform:
    segment: Segment
    first_index: int
    stop_index: int
    values: np.ndarray = field(repr=False, compare=False)
    support: SampleSpan | None
    support_reason: str


@dataclass(frozen=True)
class WorksheetRow:
    channel: str
    band: Band
    segments: tuple[Segment, ...]
    waveforms: tuple[Waveform, ...]
    initial_status: Status
    start_ns: int | None
    stop_ns: int | None

    @property
    def full_review_support(self) -> bool:
        if self.start_ns is None or self.stop_ns is None:
            return False
        for waveform in self.waveforms:
            support, segment = waveform.support, waveform.segment
            if (support is not None and support.start == segment.lower_bound(self.start_ns)
                    and support.stop == segment.lower_bound(self.stop_ns)):
                return True
        return False

    def supporting_run(self, ref: SampleRef) -> Waveform:
        for waveform in self.waveforms:
            support = waveform.support
            if (
                support is not None
                and support.start <= ref.sample_index < support.stop
                and waveform.segment.ref(ref.sample_index) == ref
            ):
                return waveform
        raise ValueError("Sample reference has no eligible processing support in this row")


def _finite_runs(values: np.ndarray) -> Iterable[tuple[int, int]]:
    edges = np.flatnonzero(np.diff(np.isfinite(values), prepend=False, append=False)).tolist()
    return zip(edges[::2], edges[1::2])


def _process(
    segment: Segment, band: Band, review_start: int, review_stop: int, overlap: bool,
) -> Iterable[Waveform]:
    low, high, guard_seconds = _BANDS[band]
    guard = ceil(guard_seconds * NS_PER_SECOND / segment.period_ns)
    empty = np.frombuffer(b"", dtype=np.float64)
    for first, stop in _finite_runs(segment.data):
        start_eligible = max(first + guard, review_start)
        stop_eligible = min(stop - guard, review_stop)
        if overlap:
            reason = "overlap"
        elif high / (segment.sampling_rate_hz / 2) >= _NYQUIST_RATIO_LIMIT:
            reason = "nyquist"
        elif stop - first <= 2 * guard:
            reason = "edge_guard"
        elif start_eligible >= stop_eligible:
            reason = "review_window"
        else:
            filtered = bandpass(
                segment.data[first:stop], low, high, segment.sampling_rate_hz,
                corners=_CORNERS, zerophase=True, ftype="butter",
            )
            if np.isfinite(filtered).all():
                values = np.frombuffer(filtered.tobytes(), dtype=filtered.dtype)
                support = SampleSpan(segment.segment_id, start_eligible, stop_eligible)
                yield Waveform(segment, first, stop, values, support, "eligible")
                continue
            reason = "nonfinite_output"
        yield Waveform(segment, first, stop, empty, None, reason)


def build_rows(
    catalog: SourceCatalog,
    channels: Iterable[str],
    *,
    start_ns: int | None = None,
    stop_ns: int | None = None,
) -> tuple[WorksheetRow, ...]:
    """Join explicit channel identifiers to both bands, including missing channels."""
    if any(bound is not None and type(bound) is not int for bound in (start_ns, stop_ns)):
        raise ValueError("Review bounds require integer UTC nanoseconds")
    if start_ns is not None and stop_ns is not None and start_ns >= stop_ns:
        raise ValueError("Review bounds must be strictly ordered")
    overlap_ids = {segment_id for pair in catalog.overlaps for segment_id in pair}

    rows = []
    for channel in dict.fromkeys(channels):
        segments = tuple(sorted((
            segment for segment in catalog.segments
            if segment.channel == channel
            and (start_ns is None or segment.end_ns >= start_ns)
            and (stop_ns is None or segment.start_ns < stop_ns)
        ), key=lambda segment: (segment.start_ns, segment.segment_id)))
        for band in _BANDS:
            waveforms = tuple(
                waveform for segment in segments for waveform in _process(
                    segment, band,
                    0 if start_ns is None else segment.lower_bound(start_ns),
                    segment.npts if stop_ns is None else segment.lower_bound(stop_ns),
                    segment.segment_id in overlap_ids,
                )
            )
            status: Status = "insufficient_support" if segments else "unavailable"
            if any(waveform.support is not None for waveform in waveforms):
                status = "unreviewed"
            rows.append(WorksheetRow(channel, band, segments, waveforms, status, start_ns, stop_ns))
    return tuple(rows)


@dataclass(frozen=True)
class Boundary:
    status: Status
    sample: SampleRef | None = None
    utc_ns: int | None = None
    reason: str = ""


@dataclass(frozen=True)
class Result:
    onset: Boundary
    end: Boundary
    status: Status
    duration_status: DurationStatus = "missing_boundary"
    duration_seconds: float | None = None


@dataclass(frozen=True)
class Edit:
    channel: str
    band: Band
    boundary: BoundaryName
    previous: Result
    result: Result

    @property
    def selection(self) -> Boundary:
        return self.result.onset if self.boundary == "onset" else self.result.end


def _evaluate(row: WorksheetRow, onset: Boundary, end: Boundary) -> Result:
    status = next(status for status in _STATUS_PRIORITY if status in (onset.status, end.status))
    if onset.sample is None or end.sample is None:
        return Result(onset, end, status)
    left, right = row.supporting_run(onset.sample), row.supporting_run(end.sample)
    if left is not right:
        return Result(onset, end, status, "separate_support")
    distance = end.sample.sample_index - onset.sample.sample_index
    if distance <= 0:
        return Result(onset, end, status, "unordered")
    duration = float(distance * left.segment.period_ns / NS_PER_SECOND)
    return Result(onset, end, status, "eligible", duration)


def _row_manifest(row: WorksheetRow) -> dict:
    return {
        "channel": row.channel,
        "band": row.band,
        "start_ns": row.start_ns,
        "stop_ns": row.stop_ns,
        "initial_status": row.initial_status,
        "full_review_support": row.full_review_support,
        "segments": [{
            "source_hash": segment.source_hash,
            "segment_id": segment.segment_id,
            "start_ns": segment.start_ns,
            "npts": segment.npts,
            "period_ns": [segment.period_ns.numerator, segment.period_ns.denominator],
            "raw_sha256": raw_digest(segment.data),
        } for segment in row.segments],
        "runs": [{
            "segment_id": waveform.segment.segment_id,
            "first_index": waveform.first_index,
            "stop_index": waveform.stop_index,
            "support": asdict(waveform.support) if waveform.support is not None else None,
            "support_reason": waveform.support_reason,
            "derived_sha256": raw_digest(waveform.values) if waveform.values.size else None,
        } for waveform in row.waveforms],
    }


class Worksheet:
    """Preview, apply, undo and validate replay of independent boundary edits."""

    def __init__(self, rows: Iterable[WorksheetRow]) -> None:
        self.rows = tuple(rows)
        self._rows = {(row.channel, row.band): row for row in self.rows}
        if len(self.rows) != len(self._rows):
            raise ValueError("Each channel and band can appear only once")
        self._results = {
            key: Result(Boundary(row.initial_status), Boundary(row.initial_status), row.initial_status)
            for key, row in self._rows.items()
        }
        self._edits: list[Edit] = []
        self._cursor = 0

    def result(self, channel: str, band: Band) -> Result:
        return self._results[channel, band]

    def preview(
        self, channel: str, band: Band, boundary: BoundaryName, status: Status, reason: str,
        sample: SampleRef | None = None,
    ) -> Edit:
        if boundary not in ("onset", "end") or status not in _STATUS_PRIORITY:
            raise ValueError("Unknown boundary or status")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("A reason is required to edit a boundary")
        if (status == "identifiable") != (sample is not None):
            raise ValueError("Only an identifiable boundary requires a sample")
        row = self._rows[channel, band]
        if ((row.initial_status != "unreviewed" and status != row.initial_status)
                or (status == "unavailable" and row.initial_status != "unavailable")):
            raise ValueError("Boundary status contradicts the row's waveform availability or support")
        if status == "no_identifiable_signal" and not row.full_review_support:
            raise ValueError("A no-signal review requires processing support over the full review window")
        utc_ns = None
        if sample is not None:
            segment = row.supporting_run(sample).segment
            utc_ns = segment.time_ns(sample.sample_index)
        selection = Boundary(status, sample, utc_ns, reason)
        previous = self.result(channel, band)
        onset, end = (selection, previous.end) if boundary == "onset" else (previous.onset, selection)
        return Edit(channel, band, boundary, previous, _evaluate(row, onset, end))

    def apply(self, edit: Edit) -> None:
        selection = edit.selection
        if edit != self.preview(edit.channel, edit.band, edit.boundary, selection.status,
                                selection.reason, selection.sample):
            raise ValueError("Boundary preview is stale")
        self._edits[self._cursor:] = [edit]
        self._cursor += 1
        self._results[edit.channel, edit.band] = edit.result

    def undo(self) -> bool:
        if self._cursor == 0:
            return False
        self._cursor -= 1
        edit = self._edits[self._cursor]
        self._results[edit.channel, edit.band] = edit.previous
        return True

    def to_dict(self) -> dict:
        """Encode revisions with exact sample identity and processing provenance."""
        return {
            "schema": _SCHEMA,
            "processing": processing_definition(),
            "rows": [_row_manifest(row) for row in self.rows],
            "edits": [{
                "channel": edit.channel, "band": edit.band, "boundary": edit.boundary,
                "selection": asdict(edit.selection),
            } for edit in self._edits],
            "cursor": self._cursor,
            "results": [
                {"channel": row.channel, "band": row.band, **asdict(self.result(row.channel, row.band))}
                for row in self.rows
            ],
        }

    @classmethod
    def from_dict(cls, rows: Iterable[WorksheetRow], payload: Mapping[str, object]) -> Worksheet:
        """Replay against freshly reconstructed rows and reject stale evidence."""
        if payload.get("schema") != _SCHEMA or payload.get("processing") != processing_definition():
            raise ValueError("Worksheet schema or processing definition differs")
        worksheet = cls(rows)
        if payload.get("rows") != [_row_manifest(row) for row in worksheet.rows]:
            raise ValueError("Worksheet waveform identity or processing support differs")
        cursor, edits = payload.get("cursor"), payload.get("edits")
        if not isinstance(edits, list) or type(cursor) is not int or not 0 <= cursor <= len(edits):
            raise ValueError("Invalid worksheet edit history")
        try:
            for item in edits:
                selection = item["selection"]
                ref = None if selection["sample"] is None else SampleRef(**selection["sample"])
                edit = worksheet.preview(item["channel"], item["band"], item["boundary"],
                                         selection["status"], selection["reason"], ref)
                worksheet.apply(edit)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Invalid worksheet replay action") from error
        while worksheet._cursor > cursor:
            worksheet.undo()
        if worksheet.to_dict() != payload:
            raise ValueError("Worksheet replay disagrees with saved evidence or results")
        return worksheet
