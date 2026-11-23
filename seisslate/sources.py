"""Read-only MiniSEED sample identities and UTC lookups."""

from __future__ import annotations

import hashlib
import io
import re
from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from types import MappingProxyType
from typing import Iterable, Literal, Mapping

import numpy as np
from obspy import Stream, read
from obspy.io.mseed.util import get_record_information


NS_PER_SECOND = 1_000_000_000
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
_ISO_TIME = re.compile(
    r"(?P<whole>\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2})"
    r"(?:\.(?P<fraction>\d{1,9}))?(?P<zone>Z|[+-]\d{2}:\d{2})"
)


def utc_ns(text: str) -> int:
    """Parse an ISO timestamp with an explicit UTC offset and up to nine digits."""
    match = _ISO_TIME.fullmatch(text)
    if match is None:
        raise ValueError("Expected an ISO timestamp with an explicit UTC offset")
    zone = match["zone"] if match["zone"] != "Z" else "+00:00"
    difference = datetime.fromisoformat(match["whole"] + zone) - _EPOCH
    fraction_ns = int((match["fraction"] or "").ljust(9, "0"))
    return (difference.days * 86_400 + difference.seconds) * NS_PER_SECOND + fraction_ns


@dataclass(frozen=True)
class SampleRef:
    source_hash: str
    channel: str
    segment_id: str
    sample_index: int

    def __post_init__(self) -> None:
        if type(self.sample_index) is not int or self.sample_index < 0:
            raise ValueError("Sample index must be a nonnegative integer")


@dataclass(frozen=True, order=True)
class SampleSpan:
    """A nonempty half-open range of original samples in one segment."""

    segment_id: str
    start: int
    stop: int

    def __post_init__(self) -> None:
        if type(self.start) is not int or type(self.stop) is not int or not 0 <= self.start < self.stop:
            raise ValueError("A sample span must have integer bounds with 0 <= start < stop")


@dataclass(frozen=True, init=False)
class Segment:
    source_hash: str
    channel: str
    segment_id: str
    start_ns: int
    sampling_rate_hz: float
    npts: int
    _period_ns: Fraction = field(repr=False)
    _data: bytes = field(repr=False, compare=False)
    _dtype: np.dtype = field(repr=False, compare=False)

    def __init__(
        self,
        source_hash: str,
        channel: str,
        segment_id: str,
        start_ns: int,
        sampling_rate_hz: float,
        data: np.ndarray,
    ) -> None:
        if type(start_ns) is not int:
            raise ValueError("Segment start must be integer UTC nanoseconds")
        if not 0 < sampling_rate_hz <= NS_PER_SECOND:
            raise ValueError("Sample rate must be positive and representable in nanoseconds")
        if data.ndim != 1 or not data.size or data.dtype.kind not in "iuf":
            raise ValueError("A segment requires a nonempty numeric sample array")
        object.__setattr__(self, "source_hash", source_hash)
        object.__setattr__(self, "channel", channel)
        object.__setattr__(self, "segment_id", segment_id)
        object.__setattr__(self, "start_ns", start_ns)
        object.__setattr__(self, "sampling_rate_hz", sampling_rate_hz)
        object.__setattr__(self, "npts", data.size)
        object.__setattr__(self, "_period_ns", NS_PER_SECOND / Fraction(str(sampling_rate_hz)))
        object.__setattr__(self, "_data", data.tobytes())
        object.__setattr__(self, "_dtype", data.dtype)

    @property
    def data(self) -> np.ndarray:
        return np.frombuffer(self._data, dtype=self._dtype)

    @property
    def end_ns(self) -> int:
        """UTC time of this segment's last sample."""
        return self.time_ns(self.npts - 1)

    def _validate_index(self, index: int) -> None:
        if type(index) is not int or not 0 <= index < self.npts:
            raise ValueError(f"Sample index is outside segment {self.segment_id}")

    def ref(self, index: int) -> SampleRef:
        self._validate_index(index)
        return SampleRef(self.source_hash, self.channel, self.segment_id, index)

    def time_ns(self, index: int) -> int:
        self._validate_index(index)
        return round(self.start_ns + index * self._period_ns)

    def lower_bound(self, timestamp_ns: int) -> int:
        """Index of the first sample at or after a UTC timestamp, or npts."""
        return bisect_left(range(self.npts), timestamp_ns, key=self.time_ns)


@dataclass(frozen=True)
class Source:
    path: Path
    sha256: str
    size_bytes: int
    segments: tuple[Segment, ...]


class SourceIdentityError(ValueError):
    """The file bytes do not match the requested source identity."""


@dataclass
class _RecordRun:
    start_ns: int
    period_ns: Fraction
    npts: int = 0
    records: list[bytes] = field(default_factory=list)


def _read_traces(contents: bytes) -> Stream:
    # ObsPy's default reader can coalesce records separated by up to half a
    # sample period. Group exactly contiguous original grids before decoding.
    file = io.BytesIO(contents)
    runs: list[_RecordRun] = []
    latest: dict[tuple[str, ...], _RecordRun] = {}
    offset = 0
    while offset < len(contents):
        if contents[offset + 6:offset + 7] not in (b"D", b"R", b"Q", b"M"):
            raise ValueError(f"Expected a MiniSEED 2 data record at byte {offset}")
        info = get_record_information(file, offset=offset)
        length = info["record_length"]
        if length < 128 or offset + length > len(contents):
            raise ValueError(f"Truncated or invalid MiniSEED 2 record at byte {offset}")
        record = contents[offset:offset + length]
        offset += length
        if not info["npts"]:
            continue
        rate = info["samp_rate"]
        if not 0 < rate <= NS_PER_SECOND:
            raise ValueError("Sample rate must be positive and representable in nanoseconds")
        start_ns = info["starttime"].ns
        period_ns = NS_PER_SECOND / Fraction(str(rate))
        key = tuple(info[name] for name in ("network", "station", "location", "channel"))
        run = latest.get(key)
        if run is None or run.period_ns != period_ns or run.start_ns + run.npts * run.period_ns != start_ns:
            run = latest[key] = _RecordRun(start_ns, period_ns)
            runs.append(run)
        run.records.append(record)
        run.npts += info["npts"]

    traces = Stream()
    for run in runs:
        decoded = read(io.BytesIO(b"".join(run.records)), format="MSEED")
        if sum(trace.stats.npts for trace in decoded) != run.npts:
            raise ValueError("MiniSEED decoding did not preserve the original sample count")
        traces.extend(decoded)
    return traces


def load_source(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    expected_size_bytes: int | None = None,
) -> Source:
    """Hash and decode MiniSEED 2 waveform data records through ObsPy."""
    path = Path(path).resolve(strict=True)
    contents = path.read_bytes()
    fingerprint = hashlib.sha256(contents).hexdigest()
    if expected_size_bytes is not None and len(contents) != expected_size_bytes:
        raise SourceIdentityError(f"Source size does not match for {path}")
    if expected_sha256 is not None and fingerprint != expected_sha256:
        raise SourceIdentityError(f"Source SHA-256 does not match for {path}")
    traces = _read_traces(contents)
    traces.sort(keys=["network", "station", "location", "channel", "starttime", "sampling_rate", "npts"])
    segments = tuple(
        Segment(
            fingerprint,
            trace.id,
            f"{fingerprint}:{index}",
            trace.stats.starttime.ns,
            trace.stats.sampling_rate,
            trace.data,
        )
        for index, trace in enumerate(traces)
    )
    if not segments:
        raise ValueError(f"No samples found in {path}")
    return Source(path, fingerprint, len(contents), segments)


@dataclass(frozen=True)
class SampleLookup:
    status: Literal["exact", "off_grid", "gap", "ambiguous"]
    candidates: tuple[SampleRef, ...]
    utc_ns: int


@dataclass(frozen=True)
class SourceCatalog:
    sources: tuple[Source, ...]
    _by_id: Mapping[str, Segment] = field(init=False, repr=False, compare=False)
    _by_channel: Mapping[str, tuple[Segment, ...]] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        sources = tuple(self.sources)
        if len({source.sha256 for source in sources}) != len(sources):
            raise ValueError("The same recording bytes were supplied more than once")
        object.__setattr__(self, "sources", sources)
        by_id = {}
        by_channel = defaultdict(list)
        for source in sources:
            for segment in source.segments:
                if segment.source_hash != source.sha256 or segment.segment_id in by_id:
                    raise ValueError("Inconsistent source segment identity")
                by_id[segment.segment_id] = segment
                by_channel[segment.channel].append(segment)
        object.__setattr__(self, "_by_id", MappingProxyType(by_id))
        object.__setattr__(self, "_by_channel", MappingProxyType({
            channel: tuple(sorted(segments, key=lambda segment: segment.start_ns))
            for channel, segments in by_channel.items()
        }))

    @property
    def segments(self) -> tuple[Segment, ...]:
        return tuple(self._by_id.values())

    @property
    def overlaps(self) -> tuple[tuple[str, str], ...]:
        """Return channel segments with overlapping sampled time coverage."""
        overlaps = []
        for segments in self._by_channel.values():
            for index, left in enumerate(segments):
                end_ns = left.end_ns
                for right_index in range(index + 1, len(segments)):
                    right = segments[right_index]
                    if right.start_ns > end_ns:
                        break
                    overlaps.append((left.segment_id, right.segment_id))
        return tuple(overlaps)

    def segment(self, ref: SampleRef | str) -> Segment:
        segment_id = ref.segment_id if isinstance(ref, SampleRef) else ref
        try:
            segment = self._by_id[segment_id]
        except KeyError as error:
            raise ValueError(f"Unknown segment {segment_id}") from error
        if isinstance(ref, SampleRef) and segment.ref(ref.sample_index) != ref:
            raise ValueError("Sample reference does not match the recording and channel")
        return segment

    def time_ns(self, ref: SampleRef) -> int:
        return self.segment(ref).time_ns(ref.sample_index)

    def lookup(self, channel: str, utc_ns: int, *, source_hash: str | None = None) -> SampleLookup:
        if type(utc_ns) is not int:
            raise ValueError("Sample lookup requires integer UTC nanoseconds")
        segments = [
            segment for segment in self._by_channel.get(channel, ())
            if (source_hash is None or segment.source_hash == source_hash)
            and segment.start_ns <= utc_ns <= segment.end_ns
        ]
        candidates = []
        exact = False
        for segment in segments:
            index = segment.lower_bound(utc_ns)
            exact = segment.time_ns(index) == utc_ns
            if not exact:
                candidates.append(segment.ref(index - 1))
            candidates.append(segment.ref(index))
        if len(segments) > 1:
            return SampleLookup("ambiguous", tuple(candidates), utc_ns)
        if not candidates:
            return SampleLookup("gap", (), utc_ns)
        return SampleLookup("exact" if exact else "off_grid", tuple(candidates), utc_ns)

    def spans(
        self,
        channels: Iterable[str],
        start_ns: int,
        stop_ns: int,
        *,
        source_hash: str | None = None,
    ) -> tuple[SampleSpan, ...]:
        """Map a shared half-open UTC selection to each intersected original grid."""
        if type(start_ns) is not int or type(stop_ns) is not int or start_ns >= stop_ns:
            raise ValueError("A UTC selection needs ordered integer nanosecond endpoints")
        channels = set(channels)
        spans = []
        for segment in self._by_id.values():
            if segment.channel not in channels or (source_hash is not None and segment.source_hash != source_hash):
                continue
            start = segment.lower_bound(start_ns)
            stop = segment.lower_bound(stop_ns)
            if start < stop:
                spans.append(SampleSpan(segment.segment_id, start, stop))
        return tuple(spans)
