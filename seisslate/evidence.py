"""Human review records tied to original sample identities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .sources import SampleRef, SampleSpan, SourceCatalog


Role = Literal["original", "calibration"]
PickStatus = Literal["candidate", "accepted", "unresolved"]
IntervalEdit = Literal["mask", "restore"]
ROLES = ("original", "calibration")


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
        if self.evidence_id is not None:
            raise ValueError("Evidence links require raw evidence capture")

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
