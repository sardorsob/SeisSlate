"""Human review records tied to original sample identities."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
from string import hexdigits
from typing import Literal

import numpy as np

from .sources import SampleRef, SampleSpan, Segment, SourceCatalog


Role = Literal["original", "calibration"]
PickStatus = Literal["candidate", "accepted", "unresolved"]
IntervalEdit = Literal["mask", "restore"]
ROLES = ("original", "calibration")
RAW_METHOD_ID = "raw-v1"
RAW_ORDINATE_SCALE = "linear-autoscale"


def require_reason(reason: str, action: str) -> None:
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError(f"{action} requires a reason")


def validate_span(catalog: SourceCatalog, span: SampleSpan) -> None:
    if span.stop > catalog.segment(span.segment_id).npts:
        raise ValueError("Inspection or action span extends beyond its source segment")


@dataclass(frozen=True)
class PickDecision:
    """The human decision for one blow and receiver, with optional stated bounds."""

    blow_id: str
    role: Role
    reviewed_ref: SampleRef | None
    status: PickStatus
    lower_ref: SampleRef | None = None
    upper_ref: SampleRef | None = None
    reason: str = ""
    reviewer: str = ""
    intended_utc_ns: int | None = None
    inspection_span: SampleSpan | None = None
    evidence_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.blow_id, str) or not self.blow_id.strip() or self.role not in ROLES:
            raise ValueError("A pick decision requires a blow ID and receiver role")
        if self.status not in ("candidate", "accepted", "unresolved"):
            raise ValueError("Unknown pick decision status")
        if type(self.reason) is not str or type(self.reviewer) is not str:
            raise ValueError("Decision reason and reviewer label must be text")
        if self.status in ("accepted", "unresolved"):
            require_reason(self.reason, "An accepted or unresolved decision")
        if self.intended_utc_ns is not None and type(self.intended_utc_ns) is not int:
            raise ValueError("Intended time must be integer UTC nanoseconds")
        if self.evidence_id is not None and (type(self.evidence_id) is not str or not self.evidence_id.strip()):
            raise ValueError("An evidence link needs a nonblank ID")

        point, lower, upper = self.reviewed_ref, self.lower_ref, self.upper_ref
        if (lower is None) != (upper is None):
            raise ValueError("Pick bounds require both endpoints")
        if point is None:
            if self.status != "unresolved" or self.intended_utc_ns is None or lower is not None:
                raise ValueError("A decision without a point must be unresolved, with intended UTC and no bounds")
            return
        if lower is not None and upper is not None:
            for bound in (lower, upper):
                if (bound.source_hash, bound.channel, bound.segment_id) != (
                    point.source_hash, point.channel, point.segment_id,
                ):
                    raise ValueError("Pick bounds must use the same source, channel and segment as the reviewed sample")
            if not lower.sample_index <= point.sample_index <= upper.sample_index:
                raise ValueError("Inclusive bounds must contain the reviewed sample")
        if self.inspection_span is not None:
            span = self.inspection_span
            first, last = lower or point, upper or point
            if span.segment_id != point.segment_id or not span.start <= first.sample_index <= last.sample_index < span.stop:
                raise ValueError("The inspection span must contain the reviewed sample and bounds")

    @property
    def target(self) -> tuple[str, Role]:
        return self.blow_id, self.role


@dataclass(frozen=True)
class IntervalAction:
    """A reason and the exact half-open spans changed by a mask or restore."""

    action: IntervalEdit
    spans: tuple[SampleSpan, ...]
    reason: str

    def __post_init__(self) -> None:
        if self.action not in ("mask", "restore"):
            raise ValueError("Unknown interval action")
        require_reason(self.reason, "An interval action")
        spans = tuple(self.spans)
        if not spans:
            raise ValueError("An interval action requires changed samples")
        object.__setattr__(self, "spans", spans)


@dataclass(frozen=True)
class RawEvidenceView:
    """Displayed sample selection, indicators, and view settings."""

    source_hash: str
    channel: str
    span: SampleSpan
    display_start_ns: int
    display_stop_ns: int
    point: SampleRef | None
    bounds: tuple[SampleRef, SampleRef] | None
    ordinate_scale: str = RAW_ORDINATE_SCALE

    def __post_init__(self) -> None:
        if not all(type(text) is str and text.strip() for text in (self.source_hash, self.channel, self.ordinate_scale)):
            raise ValueError("Raw view requires a source fingerprint, channel and ordinate scale")
        if type(self.display_start_ns) is not int or type(self.display_stop_ns) is not int:
            raise ValueError("Raw display limits must be integer UTC nanoseconds")
        if not self.display_start_ns < self.display_stop_ns:
            raise ValueError("Raw display time limits must increase")
        if self.bounds is not None:
            object.__setattr__(self, "bounds", tuple(self.bounds))


@dataclass(frozen=True)
class RawSamples:
    """Sampling grid and digest of an exact half-open sample span."""

    segment_start_ns: int
    period_numerator: int
    period_denominator: int
    dtype: str
    sample_count: int
    digest: str

    def __post_init__(self) -> None:
        if type(self.segment_start_ns) is not int or any(
            type(value) is not int or value <= 0
            for value in (self.period_numerator, self.period_denominator, self.sample_count)
        ):
            raise ValueError("Raw sample grid requires integer UTC, a positive period and a positive count")
        if type(self.dtype) is not str or not self.dtype:
            raise ValueError("Raw samples require a dtype")
        if type(self.digest) is not str or len(self.digest) != 64 or any(c not in hexdigits for c in self.digest):
            raise ValueError("Raw samples require a SHA-256 digest")


@dataclass(frozen=True)
class RawConfirmation:
    """The exact decision, view and samples the reviewer confirmed before Apply."""

    decision: PickDecision
    view: RawEvidenceView
    samples: RawSamples

    def __post_init__(self) -> None:
        decision, view = self.decision, self.view
        if decision.evidence_id is not None or not decision.reviewer.strip():
            raise ValueError("Raw confirmation needs an unlinked decision and a reviewer label")
        bounds = None if decision.lower_ref is None else (decision.lower_ref, decision.upper_ref)
        if (decision.reviewed_ref, bounds, decision.inspection_span) != (view.point, view.bounds, view.span):
            raise ValueError("Confirmed view must match the decision's point, bounds and span")
        if self.samples.sample_count != view.span.stop - view.span.start:
            raise ValueError("Raw sample count does not match the confirmed span")
        if view.point is not None:
            if (view.point.source_hash, view.point.channel) != (view.source_hash, view.channel):
                raise ValueError("Confirmed point must use the raw view's source and channel")
        else:
            grid = self.samples
            period = Fraction(grid.period_numerator, grid.period_denominator)
            start = grid.segment_start_ns + view.span.start * period
            stop = grid.segment_start_ns + view.span.stop * period
            intended = decision.intended_utc_ns
            if intended is None or not start <= intended < stop:
                raise ValueError("Raw view must cover the intended UTC of an unresolved decision")


@dataclass(frozen=True)
class RawEvidence:
    """Raw confirmation sealed in one applied revision."""

    evidence_id: str
    applied_revision_id: int
    confirmation: RawConfirmation
    captured_utc_ns: int
    software_versions: tuple[tuple[str, str], ...]
    method_id: str = RAW_METHOD_ID

    def __post_init__(self) -> None:
        if type(self.method_id) is not str or not self.method_id.strip():
            raise ValueError("Raw evidence requires a method ID")
        if type(self.applied_revision_id) is not int or self.applied_revision_id <= 0:
            raise ValueError("Raw evidence requires an applied revision")
        if self.evidence_id != f"raw-{self.applied_revision_id}":
            raise ValueError("Raw evidence ID must identify its applied revision")
        if type(self.captured_utc_ns) is not int:
            raise ValueError("Capture time must be integer UTC nanoseconds")
        versions = tuple(self.software_versions)
        if not versions or any(
            not isinstance(item, (list, tuple)) or len(item) != 2
            or not all(type(text) is str and text.strip() for text in item) for item in versions
        ) or len({item[0] for item in versions}) != len(versions):
            raise ValueError("Capture requires distinct software names and their version labels")
        object.__setattr__(self, "software_versions", tuple((name, version) for name, version in versions))


@dataclass(frozen=True)
class RawReplay:
    """Independent replay checks and the current status of the raw attestation."""

    source_identity: Literal["not_recorded", "source_match", "source_differ", "source_missing"] = "not_recorded"
    raw_samples: Literal[
        "not_recorded", "raw_samples_match", "raw_samples_differ", "unsupported_method", "not_checked",
    ] = "not_recorded"
    grid_matches: bool | None = None
    method_supported: bool | None = None
    recorded_view_valid: bool | None = None
    view_restored: bool | None = None
    masks_changed: bool = False

    @property
    def attestation_current(self) -> bool:
        return (
            self.source_identity == "source_match" and self.raw_samples == "raw_samples_match"
            and self.grid_matches is True and self.method_supported is True
            and self.recorded_view_valid is True
            and self.view_restored is not False and not self.masks_changed
        )


def _canonical_dtype(dtype: np.dtype) -> np.dtype:
    if (dtype.kind in "iu" and dtype.itemsize in (1, 2, 4, 8)) or (
        dtype.kind == "f" and dtype.itemsize in (2, 4, 8)
    ):
        return dtype.newbyteorder("<")
    raise ValueError(f"Unsupported raw-v1 sample dtype {dtype}")


def raw_digest(values: np.ndarray) -> str:
    """Digest a nonempty one-dimensional span with the raw-v1 typed frame."""
    if values.ndim != 1 or not values.size:
        raise ValueError("A raw-v1 digest requires a nonempty one-dimensional sample array")
    dtype = _canonical_dtype(values.dtype)
    little_endian = values if values.dtype == dtype else values.byteswap().view(dtype)
    dtype_tag = dtype.str.encode("ascii")
    digest = sha256(b"SeisSlate raw-v1\0" + len(dtype_tag).to_bytes(1, "little") + dtype_tag)
    digest.update(values.size.to_bytes(8, "little"))
    digest.update(little_endian.tobytes(order="C"))
    return digest.hexdigest()


def _sample_descriptor(segment: Segment, span: SampleSpan) -> RawSamples:
    data = segment.data[span.start:span.stop]
    period = segment.period_ns
    return RawSamples(
        segment.start_ns, period.numerator, period.denominator, _canonical_dtype(data.dtype).str,
        data.size, raw_digest(data),
    )


def _valid_display_range(segment: Segment, view: RawEvidenceView) -> bool:
    span = view.span
    stop = segment.time_ns(span.stop) if span.stop < segment.npts else segment.stop_ns
    return (view.display_start_ns, view.display_stop_ns) == (segment.time_ns(span.start), stop)


def confirm_pick_evidence(catalog: SourceCatalog, decision: PickDecision, view: RawEvidenceView) -> RawConfirmation:
    """Capture the confirmed raw view so Apply can compare it again before sealing."""
    validate_span(catalog, view.span)
    segment = catalog.segment(view.span.segment_id)
    if (segment.source_hash, segment.channel) != (view.source_hash, view.channel):
        raise ValueError("Raw view does not match its source and channel")
    if not _valid_display_range(segment, view):
        raise ValueError("The raw view time limits do not match the displayed span")
    if view.ordinate_scale != RAW_ORDINATE_SCALE:
        raise ValueError("Unsupported raw view ordinate scale")
    for ref in (decision.reviewed_ref, decision.lower_ref, decision.upper_ref):
        if ref is not None:
            catalog.segment(ref)
    return RawConfirmation(decision, view, _sample_descriptor(segment, view.span))


def inspect_raw_evidence(
    catalog: SourceCatalog, evidence: RawEvidence, *, view: RawEvidenceView | None = None,
) -> RawReplay:
    """Compare recorded claims independently against the currently loaded sources and view."""
    saved = evidence.confirmation
    support = saved.view.span
    method = evidence.method_id == RAW_METHOD_ID
    samples: Literal["raw_samples_match", "raw_samples_differ", "unsupported_method", "not_checked"] = (
        "not_checked" if method else "unsupported_method"
    )
    try:
        segment = catalog.segment(support.segment_id)
    except ValueError:
        source_present = any(source.sha256 == saved.view.source_hash for source in catalog.sources)
        return RawReplay(
            source_identity="source_match" if source_present else "source_missing", raw_samples=samples,
            grid_matches=False if source_present else None, method_supported=method,
        )
    if (segment.source_hash, segment.channel) != (saved.view.source_hash, saved.view.channel):
        return RawReplay(source_identity="source_differ", raw_samples=samples, method_supported=method)

    baseline = saved.samples
    period = segment.period_ns
    has_support = support.stop <= segment.npts
    grid_matches = has_support and (segment.start_ns, period.numerator, period.denominator) == (
        baseline.segment_start_ns, baseline.period_numerator, baseline.period_denominator,
    )
    if has_support and method:
        try:
            actual = _sample_descriptor(segment, support)
        except ValueError:
            samples = "raw_samples_differ"
        else:
            samples = "raw_samples_match" if (actual.dtype, actual.digest) == (baseline.dtype, baseline.digest) else (
                "raw_samples_differ"
            )
    recorded_view_valid = None
    if method:
        recorded_view_valid = (
            has_support and saved.view.ordinate_scale == RAW_ORDINATE_SCALE
            and _valid_display_range(segment, saved.view)
        )
    return RawReplay(
        source_identity="source_match", raw_samples=samples, grid_matches=grid_matches, method_supported=method,
        recorded_view_valid=recorded_view_valid,
        view_restored=None if view is None or recorded_view_valid is None else recorded_view_valid and view == saved.view,
    )
