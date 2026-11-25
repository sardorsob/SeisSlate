"""Exact-window descriptive RMS contrast on one original waveform grid."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, replace
from fractions import Fraction
from math import ceil, frexp, fsum, isfinite, ldexp, sqrt
from sys import float_info
from typing import TYPE_CHECKING, Literal

import numpy as np

from .evidence import require_reason
from .sources import SampleSpan, Segment, SourceCatalog

if TYPE_CHECKING:
    from .review import ReviewState


RMS_METHOD = "demeaned-rms-v1"
RMS_REPLAY_RELATIVE_TOLERANCE = 1e-12
MAX_WINDOW_SAMPLES = 1_000_000
UtcNs = int | Fraction
EpochStatus = Literal["unknown", "unverified", "independently_verified", "conflicting"]
RmsUnits = Literal["raw counts", "raw units unknown"]
RmsDisposition = Literal["candidate", "accepted", "unresolved", "rejected"]
RmsReason = Literal[
    "missing_window", "wrong_source_or_channel", "cross_segment", "out_of_bounds", "unequal_length",
    "too_short", "too_large", "overlapping_windows", "ambiguous_coverage", "masked_samples",
    "nonfinite_samples", "epoch_conflict", "zero_background", "numeric_failure",
]
RMS_REASON_ORDER: tuple[RmsReason, ...] = (
    "missing_window", "wrong_source_or_channel", "cross_segment", "out_of_bounds", "unequal_length",
    "too_short", "too_large", "overlapping_windows", "ambiguous_coverage", "masked_samples",
    "nonfinite_samples", "epoch_conflict", "zero_background", "numeric_failure",
)


def _validate_interval(start: UtcNs, stop: UtcNs) -> None:
    if type(start) not in (int, Fraction) or type(stop) not in (int, Fraction) or start >= stop:
        raise ValueError("RMS intervals need ordered exact UTC nanosecond endpoints")


@dataclass(frozen=True)
class RmsWindow:
    """Original half-open selection intent, before any resolution to sample indices."""

    source_hash: str
    channel: str
    start_ns: UtcNs
    stop_ns: UtcNs

    def __post_init__(self) -> None:
        _validate_interval(self.start_ns, self.stop_ns)


@dataclass(frozen=True)
class RmsEvidence:
    """A retained, trace-linked metadata reference with exact validity bounds."""

    source_hash: str
    channel: str
    start_ns: UtcNs
    stop_ns: UtcNs
    locator: str
    sha256: str
    provenance: str

    def __post_init__(self) -> None:
        _validate_interval(self.start_ns, self.stop_ns)
        for label in (self.source_hash, self.channel, self.locator, self.provenance):
            if not isinstance(label, str) or not label.strip():
                raise ValueError("RMS metadata evidence needs trace identity, a locator and provenance")
        if not isinstance(self.sha256, str) or re.fullmatch(r"[0-9a-f]{64}", self.sha256) is None:
            raise ValueError("RMS metadata evidence needs a SHA-256 fingerprint")


@dataclass(frozen=True)
class EpochEvidence:
    """Source-tagged response evidence and separate independent verification provenance.

    The caller supplies verification for a retained, checked record. Clear it if
    that record is unavailable during replay or handoff.
    """

    reference: RmsEvidence
    verification: str = ""
    conflicting: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.verification, str) or type(self.conflicting) is not bool:
            raise ValueError("Invalid epoch evidence verification or conflict flag")


@dataclass(frozen=True)
class RmsWindows:
    source_hash: str
    channel: str
    event: RmsWindow | None = None
    background: RmsWindow | None = None
    epoch_evidence: EpochEvidence | None = None
    counts_evidence: RmsEvidence | None = None


@dataclass(frozen=True)
class RmsWindowResult:
    span: SampleSpan | None = None
    mean: float | None = None
    rms: float | None = None
    reasons: tuple[RmsReason, ...] = ()


@dataclass(frozen=True)
class _WindowMeasurement:
    result: RmsWindowResult
    scaled_rms: tuple[float, int] | None = None


@dataclass(frozen=True)
class RmsResult:
    event: RmsWindowResult
    background: RmsWindowResult
    reasons: tuple[RmsReason, ...]
    contrast: float | None
    samples_per_window: int | None
    duration_ns: Fraction | None
    epoch_status: EpochStatus
    units: RmsUnits

    @property
    def available(self) -> bool:
        return self.contrast is not None and not self.reasons

    @property
    def qualified(self) -> bool:
        """Whether the available ratio calls for an instrument-epoch qualification."""
        return self.available and self.epoch_status in ("unknown", "unverified")


@dataclass(frozen=True)
class RmsReview:
    """Human disposition and the separately captured measurement baseline."""

    windows: RmsWindows
    disposition: RmsDisposition = "candidate"
    reviewer: str = ""
    reason: str = ""
    baseline: RmsResult | None = None
    method_version: str = RMS_METHOD

    def __post_init__(self) -> None:
        if type(self.method_version) is not str or not self.method_version.strip():
            raise ValueError("RMS method version is required")
        if self.disposition not in ("candidate", "accepted", "unresolved", "rejected"):
            raise ValueError("Unknown RMS review disposition")
        if not isinstance(self.reviewer, str) or not isinstance(self.reason, str):
            raise ValueError("RMS reviewer and reason must be text")
        if self.disposition != "candidate":
            require_reason(self.reason, "An accepted, unresolved or rejected RMS review")


def _stable_mean(values: np.ndarray, scaled: np.ndarray, exponent: int) -> float:
    count = len(values)
    try:
        total = fsum(values)
    except OverflowError:
        scaled_total = fsum(scaled)
        # Treat tiny scaled operands exactly if cancellation makes them matter.
        if abs(scaled_total) <= float_info.min * count and np.any(
            (values != 0) & (np.abs(scaled) < float_info.min)
        ):
            exact = sum(Fraction.from_float(float(value)) for value in values) / count
            mean = float(exact)
            if exact and mean == 0:
                raise ArithmeticError("Mean underflow")
            return mean
        if abs(scaled_total) < 1:
            total = ldexp(scaled_total, exponent)
        else:
            return ldexp(scaled_total / count, exponent)
    mean = total / count
    if total and mean == 0:
        raise ArithmeticError("Mean underflow")
    return mean


def _stable_statistics(samples: np.ndarray) -> tuple[float, float, tuple[float, int]]:
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        values = samples.astype(np.float64, copy=False)
        if samples.dtype.itemsize >= 8 and samples.dtype != np.dtype("float64"):
            if samples.dtype.kind in "iu":
                limit = 2.0 ** (8 * samples.dtype.itemsize - (samples.dtype.kind == "i"))
                if np.any(values >= limit):
                    raise ArithmeticError("Integer conversion overflow")
            if np.any(values.astype(samples.dtype) != samples):
                raise ArithmeticError("Sample conversion loses precision")

        low, high = float(np.min(values)), float(np.max(values))
        if low == high:
            return low, 0.0, (0.0, 0)
        exponent = frexp(max(abs(low), abs(high)))[1]
        scaled = np.ldexp(values, -exponent)
        mean = _stable_mean(values, scaled, exponent)
        # A nearby origin preserves small variations on top of a large offset.
        centered = scaled - scaled[0]
        centered -= fsum(centered) / len(centered)
        mantissa, shift = frexp(sqrt(float(np.mean(np.square(centered)))))
        rms_exponent = exponent + shift
        rms = ldexp(mantissa, rms_exponent)
        if not isfinite(mean) or not isfinite(rms) or rms == 0:
            raise ArithmeticError("Window statistics are outside float64 range")
        return mean, rms, (mantissa, rms_exponent)


def _measure_window(
    window: RmsWindow | None, identity: tuple[str, str], segments: tuple[Segment, ...], masks: tuple[SampleSpan, ...],
) -> _WindowMeasurement:
    if window is None:
        return _WindowMeasurement(RmsWindowResult(reasons=("missing_window",)))
    if (window.source_hash, window.channel) != identity or not segments:
        return _WindowMeasurement(RmsWindowResult(reasons=("wrong_source_or_channel",)))

    # Test coverage inside the recording separately from either requested edge.
    segment_stops = tuple(segment.start_ns + segment.npts * segment.period_ns for segment in segments)
    inside_start = max(window.start_ns, segments[0].start_ns)
    inside_stop = min(window.stop_ns, max(segment_stops))
    out_of_bounds = inside_start != window.start_ns or inside_stop != window.stop_ns

    covering = None
    coverage_stop = inside_start
    ambiguous = False
    for segment, segment_stop in zip(segments, segment_stops, strict=True):
        start = max(inside_start, segment.start_ns)
        stop = min(inside_stop, segment_stop)
        if start >= stop:
            continue
        ambiguous |= start < coverage_stop
        coverage_stop = max(coverage_stop, stop)
        if segment.start_ns <= inside_start and inside_stop <= segment_stop:
            covering = segment
    reasons: list[RmsReason] = []
    if covering is None and inside_start < inside_stop:
        reasons.append("cross_segment")
    if out_of_bounds:
        reasons.append("out_of_bounds")
    if ambiguous:
        reasons.append("ambiguous_coverage")
    if reasons:
        return _WindowMeasurement(RmsWindowResult(reasons=tuple(reasons)))

    assert covering is not None
    segment = covering
    start_index = ceil((window.start_ns - segment.start_ns) / segment.period_ns)
    stop_index = ceil((window.stop_ns - segment.start_ns) / segment.period_ns)
    count = stop_index - start_index
    if count == 0:
        return _WindowMeasurement(RmsWindowResult(reasons=("too_short",)))
    span = SampleSpan(segment.segment_id, start_index, stop_index)
    masked = any(mask.segment_id == span.segment_id and mask.start < span.stop and span.start < mask.stop for mask in masks)
    if count > MAX_WINDOW_SAMPLES:
        oversized: tuple[RmsReason, ...] = ("too_large", "masked_samples") if masked else ("too_large",)
        return _WindowMeasurement(RmsWindowResult(span, reasons=oversized))

    if count < 2:
        reasons.append("too_short")
    if masked:
        reasons.append("masked_samples")
    samples = segment.data[span.start:span.stop]
    if not np.all(np.isfinite(samples)):
        reasons.append("nonfinite_samples")
    if reasons:
        return _WindowMeasurement(RmsWindowResult(span, reasons=tuple(reasons)))
    try:
        mean, rms, scaled_rms = _stable_statistics(samples)
    except ArithmeticError:
        return _WindowMeasurement(RmsWindowResult(span, reasons=("numeric_failure",)))
    return _WindowMeasurement(RmsWindowResult(span, mean, rms), scaled_rms)


def _provenance(windows: RmsWindows, identity: tuple[str, str]) -> tuple[EpochStatus, RmsUnits]:
    event, background = windows.event, windows.background
    if event is None or background is None or any(
        (window.source_hash, window.channel) != identity for window in (event, background)
    ):
        return "unknown", "raw units unknown"
    start = min(event.start_ns, background.start_ns)
    stop = max(event.stop_ns, background.stop_ns)

    def matches(reference: RmsEvidence) -> bool:
        return (reference.source_hash, reference.channel) == identity

    def covers(reference: RmsEvidence) -> bool:
        return matches(reference) and reference.start_ns <= start and stop <= reference.stop_ns

    units: RmsUnits = "raw counts" if windows.counts_evidence and covers(windows.counts_evidence) else "raw units unknown"
    epoch = windows.epoch_evidence
    if epoch is None or not matches(epoch.reference):
        return "unknown", units
    if epoch.conflicting and epoch.reference.start_ns < stop and start < epoch.reference.stop_ns:
        return "conflicting", units
    return ("independently_verified" if epoch.verification.strip() and covers(epoch.reference) else "unverified"), units


def validate_baseline_provenance(windows: RmsWindows, baseline: RmsResult) -> None:
    """Check that the historical labels are possible with the retained references."""
    epoch, units = _provenance(windows, (windows.source_hash, windows.channel))
    allowed_epochs = (epoch, "unverified") if epoch == "independently_verified" else (epoch,)
    if baseline.epoch_status not in allowed_epochs or baseline.units not in (units, "raw units unknown"):
        raise ValueError("RMS baseline provenance does not match the retained evidence")
    if ("epoch_conflict" in baseline.reasons) != (baseline.epoch_status == "conflicting"):
        raise ValueError("RMS baseline conflict reason does not match its epoch status")
    if ("zero_background" in baseline.reasons) != (baseline.background.rms == 0):
        raise ValueError("RMS baseline zero-background reason does not match its background RMS")


def compute_rms_contrast(catalog: SourceCatalog, state: ReviewState, windows: RmsWindows) -> RmsResult:
    """Measure equal original-sample windows and report reasons for unavailable contrast."""
    identity = windows.source_hash, windows.channel
    segments = tuple(sorted(
        (segment for segment in catalog.segments if (segment.source_hash, segment.channel) == identity),
        key=lambda segment: segment.start_ns,
    ))
    event_measurement = _measure_window(windows.event, identity, segments, state.masks)
    background_measurement = _measure_window(windows.background, identity, segments, state.masks)
    event, background = event_measurement.result, background_measurement.result
    reasons = set(event.reasons + background.reasons)
    count = None
    duration = None
    if event.span is not None and background.span is not None:
        same_segment = event.span.segment_id == background.span.segment_id
        if not same_segment:
            reasons.add("cross_segment")
        selected_count = event.span.stop - event.span.start
        if selected_count != background.span.stop - background.span.start:
            reasons.add("unequal_length")
        elif same_segment:
            count = selected_count
            duration = count * catalog.segment(event.span.segment_id).period_ns
    first, second = windows.event, windows.background
    if first is not None and second is not None and (
        (first.source_hash, first.channel) == (second.source_hash, second.channel) == identity
        and first.start_ns < second.stop_ns and second.start_ns < first.stop_ns
    ):
        reasons.add("overlapping_windows")

    epoch, units = _provenance(windows, identity)
    if epoch == "conflicting":
        reasons.add("epoch_conflict")
    if background.rms == 0:
        reasons.add("zero_background")
    contrast = None
    if not reasons and event_measurement.scaled_rms is not None and background_measurement.scaled_rms is not None:
        event_mantissa, event_exponent = event_measurement.scaled_rms
        background_mantissa, background_exponent = background_measurement.scaled_rms
        try:
            # Divide before restoring the scales so rounded subnormal RMS values do not distort the ratio.
            ratio = ldexp(event_mantissa / background_mantissa, event_exponent - background_exponent)
            if ratio == 0 and event_mantissa != 0:
                raise ArithmeticError("Contrast underflow")
            contrast = ratio
        except ArithmeticError:
            reasons.add("numeric_failure")

    return RmsResult(event, background, tuple(reason for reason in RMS_REASON_ORDER if reason in reasons), contrast,
                     count, duration, epoch, units)


@dataclass(frozen=True)
class RmsComparison:
    quantity: str
    baseline: float | None
    current: float | None
    delta: float | None
    tolerance: float | None
    matches: bool


@dataclass(frozen=True)
class RmsReplay:
    disposition: RmsDisposition
    baseline: RmsResult | None
    current: RmsResult | None = None
    status: Literal["not_recorded", "unsupported_method", "numeric_match", "numeric_mismatch"] = "not_recorded"
    comparisons: tuple[RmsComparison, ...] = ()
    structure_matches: bool | None = None

    @property
    def ready(self) -> bool:
        return (
            self.disposition == "accepted" and self.status == "numeric_match"
            and self.current is not None and self.current.available
        )


def measure_review(
    catalog: SourceCatalog, state: ReviewState, windows: RmsWindows, *, checked_evidence: Iterable[RmsEvidence] = (),
) -> RmsResult:
    """Use only supporting records the caller independently checked for this run."""
    checked = tuple(checked_evidence)
    epoch = windows.epoch_evidence
    if epoch is not None and epoch.reference not in checked:
        epoch = replace(epoch, verification="")
    counts = windows.counts_evidence if windows.counts_evidence in checked else None
    return compute_rms_contrast(catalog, state, replace(windows, epoch_evidence=epoch, counts_evidence=counts))


def _compare_number(quantity: str, baseline: float | None, current: float | None) -> RmsComparison:
    if baseline is None or current is None:
        return RmsComparison(quantity, baseline, current, None, None, baseline is current)
    delta = current - baseline
    tolerance = abs(baseline) * RMS_REPLAY_RELATIVE_TOLERANCE
    # Exact zero has a distinct interpretation, including at the subnormal boundary.
    matches = (baseline == 0) == (current == 0) and abs(delta) <= tolerance
    return RmsComparison(quantity, baseline, current, delta if isfinite(delta) else None, tolerance, matches)


def inspect_rms_review(
    catalog: SourceCatalog, state: ReviewState, review: RmsReview, *, checked_evidence: Iterable[RmsEvidence] = (),
) -> RmsReplay:
    baseline = review.baseline
    if review.method_version != RMS_METHOD:
        return RmsReplay(review.disposition, baseline, status="unsupported_method")
    current = measure_review(catalog, state, review.windows, checked_evidence=checked_evidence)
    if baseline is None:
        return RmsReplay(review.disposition, baseline, current)

    def structure(result: RmsResult) -> tuple:
        return (
            result.event.span, result.event.reasons, result.background.span, result.background.reasons,
            result.reasons, result.samples_per_window, result.duration_ns,
        )

    structure_matches = structure(baseline) == structure(current)
    comparisons = (
        _compare_number("event_mean", baseline.event.mean, current.event.mean),
        _compare_number("event_rms", baseline.event.rms, current.event.rms),
        _compare_number("background_mean", baseline.background.mean, current.background.mean),
        _compare_number("background_rms", baseline.background.rms, current.background.rms),
        _compare_number("contrast", baseline.contrast, current.contrast),
    )
    matches = structure_matches and all(check.matches for check in comparisons)
    return RmsReplay(
        review.disposition, baseline, current, "numeric_match" if matches else "numeric_mismatch",
        comparisons, structure_matches,
    )
