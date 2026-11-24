"""Approximate hammer-group windows and preflight against saved receiver roles."""

from __future__ import annotations

import csv
import hashlib
import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from .sources import NS_PER_SECOND, SourceCatalog, utc_ns

if TYPE_CHECKING:
    from .review import Receiver, ReviewState


_REQUIRED = {"group_id", "start_utc"}
_WINDOW_END = {"end_utc", "duration_s"}
_OPTIONAL = {"time_basis", "source_position_id", "original_channel", "calibration_channel", "timing_note", "geometry_note"}
_DURATION = re.compile(r"([0-9]+)(?:\.([0-9]{1,9}))?")
_LAST_UTC_NS = utc_ns("9999-12-31T23:59:59.999999999Z")


def _explicit_utc(text: str) -> int:
    text = text.strip()
    if not text.endswith(("Z", "+00:00")):
        raise ValueError("Expected an explicit UTC timestamp ending in Z or +00:00")
    return utc_ns(text)


def _window_bounds(start_utc: str, end_utc: str | None, duration_s: str | None) -> tuple[int, int]:
    start_ns = _explicit_utc(start_utc)
    if end_utc is not None and duration_s is None:
        stop_ns = _explicit_utc(end_utc)
    elif duration_s is not None and end_utc is None:
        match = _DURATION.fullmatch(duration_s.strip())
        if match is None:
            raise ValueError("Duration needs decimal seconds with at most nine fractional digits")
        whole_seconds = match[1].lstrip("0") or "0"
        if len(whole_seconds) > 12:
            raise ValueError("Duration exceeds the range representable in UTC")
        seconds = int(whole_seconds)
        stop_ns = start_ns + seconds * NS_PER_SECOND + int((match[2] or "").ljust(9, "0"))
    else:
        raise ValueError("Specify exactly one of end_utc and duration_s")
    if not start_ns < stop_ns <= _LAST_UTC_NS:
        raise ValueError("Group end must follow its start and be representable in UTC")
    return start_ns, stop_ns


@dataclass(frozen=True)
class GroupWindow:
    """A half-open orientation window with the raw time text and CSV provenance."""

    group_id: str
    start_ns: int
    stop_ns: int
    start_utc: str
    end_utc: str | None
    duration_s: str | None
    source_file: str
    source_sha256: str
    source_row: int
    time_basis: str = "UTC"
    source_position_id: str | None = None
    original_channel: str | None = None
    calibration_channel: str | None = None
    timing_note: str | None = None
    geometry_note: str | None = None
    reviewed: bool = False

    def __post_init__(self) -> None:
        for name in ("group_id", "start_utc", "time_basis", "source_file", "source_sha256"):
            value = getattr(self, name)
            if type(value) is not str or not value.strip() or "\0" in value:
                raise ValueError(f"Group {name} needs nonempty text")
        for name in ("end_utc", "duration_s", "source_position_id", "original_channel", "calibration_channel",
                     "timing_note", "geometry_note"):
            value = getattr(self, name)
            if value is not None and (type(value) is not str or not value.strip() or "\0" in value):
                raise ValueError(f"Group {name} needs nonempty text or null")
        if self.source_file in (".", "..") or "/" in self.source_file or "\\" in self.source_file:
            raise ValueError("Group source file must be a basename")
        if not re.fullmatch(r"[0-9a-f]{64}", self.source_sha256):
            raise ValueError("Invalid group source fingerprint")
        if type(self.source_row) is not int or self.source_row < 2:
            raise ValueError("Invalid group source row")
        if type(self.reviewed) is not bool:
            raise ValueError("Group reviewed status must be boolean")
        if (type(self.start_ns) is not int or type(self.stop_ns) is not int
                or (self.start_ns, self.stop_ns) != _window_bounds(self.start_utc, self.end_utc, self.duration_s)):
            raise ValueError("Group bounds must match the original timestamp text")


def load_group_csv(path: Path | str) -> tuple[GroupWindow, ...]:
    """Read one batch without changing a review, reporting errors at physical CSV lines."""
    path = Path(path)
    contents = path.read_bytes()
    try:
        text = contents.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        line = len(re.findall(rb"\r\n|\r|\n", error.object[:error.start])) + 1
        raise ValueError(f"Line {line}: expected a UTF-8 group CSV") from error
    fingerprint = hashlib.sha256(contents).hexdigest()
    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    groups = []
    seen = set()
    line = 1
    try:
        header = next(reader, [])
        columns = set(header)
        if len(header) != len(columns):
            raise ValueError("Duplicate column names")
        if not _REQUIRED <= columns or not _WINDOW_END & columns:
            raise ValueError("Expected group_id, start_utc, and end_utc or duration_s")
        if unknown := columns - (_REQUIRED | _WINDOW_END | _OPTIONAL):
            raise ValueError(f"Unknown columns: {', '.join(sorted(unknown))}")

        while True:
            line = reader.line_num + 1
            row = next(reader, None)
            if row is None:
                break
            if not row:
                continue
            if len(row) != len(header):
                raise ValueError(f"Expected {len(header)} fields, found {len(row)}")
            cells = dict(zip(header, row, strict=True))
            group_id = cells["group_id"].strip()
            if group_id in seen:
                raise ValueError(f"Duplicate group ID {group_id!r}")
            start_utc = cells["start_utc"]
            raw_end = cells.get("end_utc", "")
            raw_duration = cells.get("duration_s", "")
            end_utc = raw_end if raw_end.strip() else None
            duration_s = raw_duration if raw_duration.strip() else None
            start_ns, stop_ns = _window_bounds(start_utc, end_utc, duration_s)
            groups.append(GroupWindow(
                group_id, start_ns, stop_ns, start_utc, end_utc, duration_s, path.name, fingerprint, line,
                time_basis=cells.get("time_basis", "").strip() or "UTC",
                source_position_id=cells.get("source_position_id", "").strip() or None,
                original_channel=cells.get("original_channel", "").strip() or None,
                calibration_channel=cells.get("calibration_channel", "").strip() or None,
                timing_note=cells.get("timing_note", "").strip() or None,
                geometry_note=cells.get("geometry_note", "").strip() or None,
            ))
            seen.add(group_id)
        if not groups:
            line = 1
            raise ValueError("Expected at least one group row")
    except (ValueError, csv.Error) as error:
        raise ValueError(f"Line {line}: {error}") from error
    return tuple(groups)


@dataclass(frozen=True)
class RoleInspection:
    channel_id: str | None
    sha256: str | None
    declared_channel: str | None
    coverage: Literal["none", "partial", "full", "conflict"]
    channel_status: Literal["missing", "unverified", "match", "conflict"]
    intervals: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True)
class GroupInspection:
    original: RoleInspection
    calibration: RoleInspection
    timing_status: Literal["missing", "unverified"]
    geometry_status: Literal["missing", "unverified"]


def _inspect_role(
    catalog: SourceCatalog, receiver: Receiver | None, declared: str | None, start_ns: int, stop_ns: int,
) -> RoleInspection:
    if receiver is None:
        return RoleInspection(None, None, declared, "none", "missing")
    segments = [
        segment for segment in catalog.segments
        if segment.channel == receiver.channel and (not receiver.source_hash or segment.source_hash == receiver.source_hash)
    ]
    fingerprints = {segment.source_hash for segment in segments}
    if not fingerprints:
        return RoleInspection(receiver.channel, receiver.source_hash or None, declared, "none", "missing")
    if len(fingerprints) != 1:
        return RoleInspection(receiver.channel, None, declared, "conflict", "conflict")

    intervals: list[tuple[int, int]] = []
    overlap = False
    for start, stop in sorted(
        (max(start_ns, segment.start_ns), min(stop_ns, segment.stop_ns)) for segment in segments
    ):
        if start >= stop:
            continue
        if intervals and start <= intervals[-1][1]:
            previous_start, previous_stop = intervals[-1]
            overlap |= start < previous_stop
            intervals[-1] = previous_start, max(stop, previous_stop)
        else:
            intervals.append((start, stop))

    coverage: Literal["none", "partial", "full", "conflict"]
    if overlap:
        coverage = "conflict"
    elif intervals == [(start_ns, stop_ns)]:
        coverage = "full"
    else:
        coverage = "partial" if intervals else "none"
    channel_status: Literal["unverified", "match", "conflict"]
    if declared is None:
        channel_status = "unverified"
    else:
        channel_status = "match" if declared == receiver.channel else "conflict"
    return RoleInspection(receiver.channel, fingerprints.pop(), declared, coverage, channel_status, tuple(intervals))


def inspect_group(catalog: SourceCatalog, state: ReviewState, group: GroupWindow) -> GroupInspection:
    """Check coverage on explicitly assigned recordings without inferring a channel alias."""
    return GroupInspection(
        _inspect_role(catalog, state.receiver("original"), group.original_channel, group.start_ns, group.stop_ns),
        _inspect_role(catalog, state.receiver("calibration"), group.calibration_channel, group.start_ns, group.stop_ns),
        "unverified" if group.timing_note or state.timing_evidence else "missing",
        "unverified" if group.geometry_note else "missing",
    )
