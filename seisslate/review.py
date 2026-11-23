"""Immutable review snapshots and the edits that produce new revisions."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from math import isfinite
from typing import Iterable, Literal

from .sources import SampleRef, SampleSpan, SourceCatalog


Role = Literal["original", "calibration"]
PairStatus = Literal["confirmed", "ambiguous", "unmatched"]
_ROLES = ("original", "calibration")
_PAIR_STATUSES = ("confirmed", "ambiguous", "unmatched")


@dataclass(frozen=True)
class Receiver:
    role: Role
    channel: str
    source_hash: str = ""

    def __post_init__(self) -> None:
        _validate_role(self.role)
        if not self.channel:
            raise ValueError("Receiver channel is required")


@dataclass(frozen=True)
class Pick:
    imported_ns: int | None
    original_ref: SampleRef | None = None
    reviewed_ref: SampleRef | None = None

    def __post_init__(self) -> None:
        if self.imported_ns is not None and type(self.imported_ns) is not int:
            raise ValueError("Imported pick time must be integer UTC nanoseconds")


@dataclass(frozen=True)
class Blow:
    blow_id: str
    original: Pick
    calibration: Pick
    distance_original_m: float | None
    distance_calibration_m: float | None
    included: bool = True
    reason: str = ""
    pair_status: PairStatus = "confirmed"
    pair_reason: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.blow_id, str) or not self.blow_id.strip():
            raise ValueError("Blow ID is required")
        if self.pair_status not in _PAIR_STATUSES:
            raise ValueError("Unknown pair status")
        if type(self.included) is not bool:
            raise ValueError("Requested fit inclusion must be boolean")


@dataclass(frozen=True)
class ReviewState:
    blows: tuple[Blow, ...]
    masks: tuple[SampleSpan, ...] = ()
    receivers: tuple[Receiver, ...] = ()
    timing_evidence: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "blows", tuple(self.blows))
        object.__setattr__(self, "masks", _merge_spans(self.masks))
        object.__setattr__(self, "receivers", tuple(self.receivers))
        if len({blow.blow_id for blow in self.blows}) != len(self.blows):
            raise ValueError("Blow IDs must be unique")
        if len({receiver.role for receiver in self.receivers}) != len(self.receivers):
            raise ValueError("Receiver roles must be unique")
        if len(self.receivers) == 2:
            first, second = self.receivers
            if first.channel == second.channel and (
                not first.source_hash or not second.source_hash or first.source_hash == second.source_hash
            ):
                raise ValueError("Receivers with the same channel ID need distinct source fingerprints")

    def blow(self, blow_id: str) -> Blow:
        for blow in self.blows:
            if blow.blow_id == blow_id:
                return blow
        raise ValueError(f"Unknown blow ID {blow_id}")

    def receiver(self, role: Role) -> Receiver | None:
        _validate_role(role)
        return next((receiver for receiver in self.receivers if receiver.role == role), None)

    def is_masked(self, ref: SampleRef | None) -> bool:
        return ref is not None and any(
            span.segment_id == ref.segment_id and span.start <= ref.sample_index < span.stop
            for span in self.masks
        )


def _validate_role(role: Role) -> None:
    if role not in _ROLES:
        raise ValueError(f"Unknown receiver role {role}")


def import_pick(
    catalog: SourceCatalog,
    channel: str,
    time_ns: int | None,
    chosen_ref: SampleRef | None = None,
    *,
    source_hash: str | None = None,
) -> Pick:
    """Retain the imported UTC and map to a sample only when unambiguous or chosen."""
    if time_ns is None:
        if chosen_ref is not None:
            raise ValueError("A missing imported time cannot be mapped to a sample")
        return Pick(None)
    lookup = catalog.lookup(channel, time_ns, source_hash=source_hash or None)
    if chosen_ref is not None and chosen_ref not in lookup.candidates:
        raise ValueError("Selected sample is not a candidate for this imported time")
    if chosen_ref is None and lookup.status == "exact":
        chosen_ref = lookup.candidates[0]
    return Pick(time_ns, chosen_ref, chosen_ref)


def _replace_blow(state: ReviewState, change: Blow) -> ReviewState:
    return replace(state, blows=tuple(change if blow.blow_id == change.blow_id else blow for blow in state.blows))


def move_pick(
    state: ReviewState, catalog: SourceCatalog, blow_id: str, role: Role, ref: SampleRef | None
) -> ReviewState:
    """Move one reviewed pick to a validated sample, or clear its reviewed value."""
    receiver = state.receiver(role)
    blow = state.blow(blow_id)
    pick = getattr(blow, role)
    if ref is not None:
        catalog.segment(ref)
        prior = pick.original_ref or pick.reviewed_ref
        channel = prior.channel if prior else None
        source_hash = prior.source_hash if prior else None
        if receiver is not None:
            channel = receiver.channel
            source_hash = receiver.source_hash or source_hash
        if (channel and ref.channel != channel) or (source_hash and ref.source_hash != source_hash):
            raise ValueError("Sample reference is outside the requested receiver's recording")
    reviewed = replace(pick, reviewed_ref=ref)
    updated = replace(blow, original=reviewed) if role == "original" else replace(blow, calibration=reviewed)
    return _replace_blow(state, updated)


def nudge_pick(state: ReviewState, catalog: SourceCatalog, blow_id: str, role: Role, step: int) -> ReviewState:
    _validate_role(role)
    if type(step) is not int or step not in (-1, 1):
        raise ValueError("A nudge moves by one original sample")
    ref = getattr(state.blow(blow_id), role).reviewed_ref
    if ref is None:
        raise ValueError("There is no reviewed sample to nudge")
    next_ref = catalog.segment(ref).ref(ref.sample_index + step)
    return move_pick(state, catalog, blow_id, role, next_ref)


def _merge_spans(spans: Iterable[SampleSpan]) -> tuple[SampleSpan, ...]:
    merged: list[SampleSpan] = []
    for span in sorted(spans):
        prior = merged[-1] if merged else None
        if prior and prior.segment_id == span.segment_id and span.start <= prior.stop:
            merged[-1] = SampleSpan(span.segment_id, prior.start, max(prior.stop, span.stop))
        else:
            merged.append(span)
    return tuple(merged)


def mask_spans(state: ReviewState, spans: Iterable[SampleSpan]) -> ReviewState:
    return replace(state, masks=state.masks + tuple(spans))


def restore_spans(state: ReviewState, spans: Iterable[SampleSpan]) -> ReviewState:
    by_segment = defaultdict(list)
    for span in _merge_spans(spans):
        by_segment[span.segment_id].append(span)
    remaining = []
    for masked in state.masks:
        start = masked.start
        for restored in by_segment.get(masked.segment_id, ()):
            if restored.stop <= start:
                continue
            if restored.start >= masked.stop:
                break
            if start < restored.start:
                remaining.append(SampleSpan(masked.segment_id, start, restored.start))
            start = restored.stop
        if start < masked.stop:
            remaining.append(SampleSpan(masked.segment_id, start, masked.stop))
    return replace(state, masks=tuple(remaining))


def set_inclusion(state: ReviewState, blow_id: str, included: bool, reason: str) -> ReviewState:
    if not included and not reason.strip():
        raise ValueError("A fit exclusion requires a reason")
    return _replace_blow(state, replace(state.blow(blow_id), included=included, reason=reason))


def set_pair_status(state: ReviewState, blow_id: str, pair_status: PairStatus, reason: str) -> ReviewState:
    if pair_status != "confirmed" and not reason.strip():
        raise ValueError("An unresolved pair requires a reason")
    return _replace_blow(state, replace(state.blow(blow_id), pair_status=pair_status, pair_reason=reason))


def _validate_distances(*distances: float | None) -> None:
    for value in distances:
        if value is not None and (isinstance(value, bool) or not isfinite(value) or value < 0):
            raise ValueError("Receiver distances must be finite and nonnegative in metres")


def add_blow(state: ReviewState, blow: Blow) -> ReviewState:
    _validate_distances(blow.distance_original_m, blow.distance_calibration_m)
    return replace(state, blows=state.blows + (blow,))


def set_geometry(
    state: ReviewState, blow_id: str, distance_original_m: float | None, distance_calibration_m: float | None
) -> ReviewState:
    _validate_distances(distance_original_m, distance_calibration_m)
    return _replace_blow(state, replace(
        state.blow(blow_id), distance_original_m=distance_original_m, distance_calibration_m=distance_calibration_m
    ))


@dataclass(frozen=True)
class Revision:
    revision_id: int
    state: ReviewState
    reason: str


def _validate_transition(before: ReviewState, after: ReviewState) -> None:
    if before.receivers != after.receivers or before.timing_evidence != after.timing_evidence:
        raise ValueError("A review edit cannot rewrite source or timing evidence")
    after_by_id = {blow.blow_id: blow for blow in after.blows}
    for old in before.blows:
        new = after_by_id.get(old.blow_id)
        if new is None:
            raise ValueError(f"A review edit cannot remove blow {old.blow_id}")
        for role in _ROLES:
            before_pick = getattr(old, role)
            after_pick = getattr(new, role)
            if before_pick.imported_ns != after_pick.imported_ns or before_pick.original_ref != after_pick.original_ref:
                raise ValueError(f"A review edit cannot rewrite imported picks for blow {old.blow_id}")


class ReviewHistory:
    """One applied revision cursor and an optional uncommitted preview."""

    def __init__(
        self,
        initial: ReviewState,
        *,
        revisions: Iterable[Revision] | None = None,
        cursor: int | None = None,
        next_revision_id: int | None = None,
    ) -> None:
        timeline = tuple(revisions) if revisions is not None else (Revision(0, initial, "Initial review"),)
        if not timeline or timeline[0].revision_id != 0 or timeline[0].state != initial:
            raise ValueError("Review history needs the original revision as its first snapshot")
        for previous, following in zip(timeline, timeline[1:]):
            if type(following.revision_id) is not int or following.revision_id <= previous.revision_id:
                raise ValueError("Revision IDs must increase")
            _validate_transition(previous.state, following.state)
        cursor = len(timeline) - 1 if cursor is None else cursor
        if type(cursor) is not int or not 0 <= cursor < len(timeline):
            raise ValueError("Review cursor is outside the revision history")
        next_revision_id = timeline[-1].revision_id + 1 if next_revision_id is None else next_revision_id
        if type(next_revision_id) is not int or next_revision_id <= timeline[-1].revision_id:
            raise ValueError("Next revision ID must follow the applied history")
        self._revisions = timeline
        self._cursor = cursor
        self._next_revision_id = next_revision_id
        self._draft: Revision | None = None

    @property
    def current(self) -> Revision:
        return self._revisions[self._cursor]

    @property
    def revisions(self) -> tuple[Revision, ...]:
        return self._revisions

    @property
    def cursor(self) -> int:
        return self._cursor

    @property
    def next_revision_id(self) -> int:
        return self._next_revision_id

    @property
    def draft(self) -> Revision | None:
        return self._draft

    @property
    def can_undo(self) -> bool:
        return self._cursor > 0

    @property
    def can_redo(self) -> bool:
        return self._cursor + 1 < len(self._revisions)

    def preview(self, state: ReviewState, reason: str) -> Revision:
        _validate_transition(self.current.state, state)
        self._draft = Revision(self._next_revision_id, state, reason)
        return self._draft

    def cancel(self) -> None:
        self._draft = None

    def apply(self) -> Revision:
        if self._draft is not None and self._draft.state != self.current.state:
            self._revisions = self._revisions[:self._cursor + 1] + (self._draft,)
            self._cursor += 1
            self._next_revision_id += 1
        self.cancel()
        return self.current

    def undo(self) -> Revision:
        if self._draft is not None:
            raise ValueError("Apply or cancel the draft before undo")
        if self.can_undo:
            self._cursor -= 1
        return self.current

    def redo(self) -> Revision:
        if self._draft is not None:
            raise ValueError("Apply or cancel the draft before redo")
        if self.can_redo:
            self._cursor += 1
        return self.current
