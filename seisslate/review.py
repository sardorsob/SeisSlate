"""Immutable review snapshots and the edits that produce new revisions."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from math import isfinite
from typing import Iterable, Literal
from weakref import ReferenceType, ref as weakref

from .evidence import (
    ROLES, IntervalAction, IntervalEdit, PickDecision, RawConfirmation, RawEvidence, RawEvidenceView,
    RawReplay, confirm_pick_evidence, inspect_raw_evidence, require_reason, validate_span,
)
from .evidence import Role as Role
from .groups import GroupWindow
from .rms import RMS_METHOD, RmsEvidence, RmsReplay, RmsReview, RmsWindows, inspect_rms_review, measure_review
from .sources import SampleRef, SampleSpan, SourceCatalog


ReviewMode = Literal["paired", "single-channel"]
PairStatus = Literal["confirmed", "ambiguous", "unmatched"]
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
    group_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.blow_id, str) or not self.blow_id.strip():
            raise ValueError("Blow ID is required")
        if self.pair_status not in _PAIR_STATUSES:
            raise ValueError("Unknown pair status")
        if type(self.included) is not bool:
            raise ValueError("Requested fit inclusion must be boolean")
        if self.group_id is not None and (type(self.group_id) is not str or not self.group_id.strip()):
            raise ValueError("Group ID must be nonempty text or null")


@dataclass(frozen=True)
class ReviewState:
    blows: tuple[Blow, ...]
    masks: tuple[SampleSpan, ...] = ()
    receivers: tuple[Receiver, ...] = ()
    timing_evidence: str = ""
    groups: tuple[GroupWindow, ...] = ()
    pick_decisions: tuple[PickDecision, ...] = ()
    interval_actions: tuple[IntervalAction, ...] = ()
    raw_evidence: tuple[RawEvidence, ...] = ()
    mode: ReviewMode = "paired"
    rms: RmsReview | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "blows", tuple(self.blows))
        object.__setattr__(self, "masks", _merge_spans(self.masks))
        object.__setattr__(self, "receivers", tuple(self.receivers))
        object.__setattr__(self, "groups", tuple(self.groups))
        object.__setattr__(self, "pick_decisions", tuple(self.pick_decisions))
        object.__setattr__(self, "interval_actions", tuple(self.interval_actions))
        object.__setattr__(self, "raw_evidence", tuple(self.raw_evidence))
        if self.mode == "single-channel":
            if self.blows or self.receivers or self.timing_evidence or self.groups or self.pick_decisions or self.raw_evidence:
                raise ValueError("A single-channel review cannot contain paired review records")
            if self.rms is None:
                raise ValueError("A single-channel review needs an RMS payload")
            windows = self.rms.windows
            identity = windows.source_hash, windows.channel
            if any(type(value) is not str or not value.strip() for value in identity):
                raise ValueError("A single-channel RMS review needs source and channel identity")
            epoch = windows.epoch_evidence
            for item in (windows.event, windows.background, windows.counts_evidence, epoch.reference if epoch else None):
                if item is not None and (item.source_hash, item.channel) != identity:
                    raise ValueError("RMS windows and metadata must identify the selected source and channel")
        elif self.mode != "paired":
            raise ValueError("Unsupported review mode")
        elif self.rms is not None:
            raise ValueError("A paired review must have a null RMS payload")

        if len({blow.blow_id for blow in self.blows}) != len(self.blows):
            raise ValueError("Blow IDs must be unique")
        group_ids = {group.group_id for group in self.groups}
        if len(group_ids) != len(self.groups):
            raise ValueError("Group IDs must be unique")
        for blow in self.blows:
            if blow.group_id is not None and blow.group_id not in group_ids:
                raise ValueError(f"Blow {blow.blow_id} links to unknown group {blow.group_id!r}")
        if len({receiver.role for receiver in self.receivers}) != len(self.receivers):
            raise ValueError("Receiver roles must be unique")
        if len(self.receivers) == 2:
            first, second = self.receivers
            if first.channel == second.channel and (
                not first.source_hash or not second.source_hash or first.source_hash == second.source_hash
            ):
                raise ValueError("Receivers with the same channel ID need distinct source fingerprints")
        records = {record.evidence_id: record for record in self.raw_evidence}
        if len(records) != len(self.raw_evidence):
            raise ValueError("Raw evidence IDs must be unique")
        if self.pick_decisions:
            if len({decision.target for decision in self.pick_decisions}) != len(self.pick_decisions):
                raise ValueError("Pick decision targets must be unique")
            blows = {blow.blow_id: blow for blow in self.blows}
            for decision in self.pick_decisions:
                target_blow = blows.get(decision.blow_id)
                if target_blow is None or getattr(target_blow, decision.role).reviewed_ref != decision.reviewed_ref:
                    raise ValueError("Pick decision must match the reviewed sample for its blow and receiver")
                if decision.evidence_id is not None:
                    record = records.get(decision.evidence_id)
                    if record is None or record.confirmation.decision != replace(decision, evidence_id=None):
                        raise ValueError("Pick decision must match its linked raw evidence")

    @classmethod
    def for_single_channel(cls, source_hash: str, channel: str) -> ReviewState:
        return cls((), mode="single-channel", rms=RmsReview(RmsWindows(source_hash, channel)))

    @property
    def single_channel_identity(self) -> tuple[str, str] | None:
        if self.rms is None:
            return None
        return self.rms.windows.source_hash, self.rms.windows.channel

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


def validate_single_channel_span(state: ReviewState, catalog: SourceCatalog, span: SampleSpan) -> None:
    validate_span(catalog, span)
    segment = catalog.segment(span.segment_id)
    if (segment.source_hash, segment.channel) != state.single_channel_identity:
        raise ValueError("Sample span is outside the selected single-channel recording")


def validate_single_channel(catalog: SourceCatalog, state: ReviewState) -> None:
    if len(catalog.sources) != 1 or not any(
        (segment.source_hash, segment.channel) == state.single_channel_identity for segment in catalog.segments
    ):
        raise ValueError("A single-channel review must identify a channel in exactly one source")


def set_rms_windows(state: ReviewState, catalog: SourceCatalog, windows: RmsWindows) -> ReviewState:
    """Stage changed window intent and clear the previous human disposition."""
    if state.rms is not None and state.rms.windows == windows:
        return state
    return set_rms_review(state, catalog, RmsReview(windows))


def set_rms_review(state: ReviewState, catalog: SourceCatalog, review: RmsReview) -> ReviewState:
    """Stage an explicit human disposition without carrying a historical baseline into the draft."""
    validate_single_channel(catalog, state)
    if state.single_channel_identity != (review.windows.source_hash, review.windows.channel):
        raise ValueError("A review edit cannot switch its single-channel source or mode")
    if review.method_version != RMS_METHOD:
        raise ValueError("Unsupported RMS method version for a new review")
    staged = replace(review, baseline=None)
    if state.rms is not None and staged == replace(state.rms, baseline=None):
        return state
    return replace(state, rms=staged)


def _validate_role(role: Role) -> None:
    if role not in ROLES:
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


def validate_pick_decision(state: ReviewState, catalog: SourceCatalog, decision: PickDecision) -> None:
    """Bind a pick decision and its support to the exact receiver recording."""
    pick = getattr(state.blow(decision.blow_id), decision.role)
    if pick.reviewed_ref != decision.reviewed_ref:
        raise ValueError("Pick decision must match the reviewed sample for its blow and receiver")
    for ref in (decision.reviewed_ref, decision.lower_ref, decision.upper_ref):
        if ref is not None:
            catalog.segment(ref)

    receiver = state.receiver(decision.role)
    prior = pick.original_ref or pick.reviewed_ref
    channel = receiver.channel if receiver else prior.channel if prior else None
    source_hash = (receiver.source_hash if receiver else "") or (prior.source_hash if prior else "")
    identities = {
        (segment.source_hash, segment.channel) for segment in catalog.segments
        if segment.channel == channel and (not source_hash or segment.source_hash == source_hash)
    }
    if channel is not None and len(identities) != 1:
        raise ValueError("Pick decision receiver must identify one source channel")
    point = decision.reviewed_ref
    if point is not None and (point.source_hash, point.channel) not in identities:
        raise ValueError("Pick decision is outside the requested receiver's recording")
    span = decision.inspection_span
    if span is not None:
        validate_span(catalog, span)
        segment = catalog.segment(span.segment_id)
        if (segment.source_hash, segment.channel) not in identities:
            raise ValueError("Inspection span is outside the requested receiver's recording")
        if point is None:
            intended = decision.intended_utc_ns
            start = segment.start_ns + span.start * segment.period_ns
            stop = segment.start_ns + span.stop * segment.period_ns
            if intended is None or not start <= intended < stop:
                raise ValueError("Inspection span must cover the intended UTC of an unresolved decision")


def set_pick_decision(state: ReviewState, catalog: SourceCatalog, decision: PickDecision) -> ReviewState:
    """Review the current point or record why a receiver has no reviewed point."""
    validate_pick_decision(state, catalog, decision)
    decisions = {previous.target: previous for previous in state.pick_decisions}
    decisions[decision.target] = decision
    return replace(state, pick_decisions=tuple(decisions.values()))


def move_pick(
    state: ReviewState, catalog: SourceCatalog, blow_id: str, role: Role, ref: SampleRef | None,
    *, decision: PickDecision | None = None,
) -> ReviewState:
    """Move or clear a pick, invalidating its decision unless explicitly replaced."""
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
    target = (blow_id, role)
    if decision is not None and decision.target != target:
        raise ValueError("A replacement decision must target the moved pick")
    if ref != pick.reviewed_ref:
        reviewed = replace(pick, reviewed_ref=ref)
        updated = replace(blow, original=reviewed) if role == "original" else replace(blow, calibration=reviewed)
        state = replace(
            state,
            blows=tuple(updated if other.blow_id == blow_id else other for other in state.blows),
            pick_decisions=tuple(previous for previous in state.pick_decisions if previous.target != target),
        )
    return state if decision is None else set_pick_decision(state, catalog, decision)


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


def _subtract_spans(spans: Iterable[SampleSpan], removed: Iterable[SampleSpan]) -> tuple[SampleSpan, ...]:
    by_segment: dict[str, list[SampleSpan]] = defaultdict(list)
    for span in removed:
        by_segment[span.segment_id].append(span)
    remaining = []
    for span in spans:
        start = span.start
        for excluded in by_segment.get(span.segment_id, ()):
            if excluded.stop <= start:
                continue
            if excluded.start >= span.stop:
                break
            if start < excluded.start:
                remaining.append(SampleSpan(span.segment_id, start, excluded.start))
            start = excluded.stop
        if start < span.stop:
            remaining.append(SampleSpan(span.segment_id, start, span.stop))
    return tuple(remaining)


def _edit_masks(
    masks: tuple[SampleSpan, ...], selection: tuple[SampleSpan, ...], action: IntervalEdit,
) -> tuple[SampleSpan, ...]:
    return _merge_spans(masks + selection) if action == "mask" else _subtract_spans(masks, selection)


def _mask_delta(
    before: tuple[SampleSpan, ...], after: tuple[SampleSpan, ...], action: IntervalEdit,
) -> tuple[SampleSpan, ...]:
    return _subtract_spans(after, before) if action == "mask" else _subtract_spans(before, after)


def _edit_intervals(
    state: ReviewState, spans: Iterable[SampleSpan], action: IntervalEdit,
    catalog: SourceCatalog | None, reason: str | None,
) -> ReviewState:
    selection = _merge_spans(spans)
    if state.rms is not None and (catalog is None or reason is None):
        raise ValueError("A single-channel interval edit requires a catalog and a reason")
    if reason is not None or catalog is not None:
        if reason is None or catalog is None:
            raise ValueError("A reasoned interval edit requires a catalog and a reason")
        require_reason(reason, "An interval action")
        for span in selection:
            if state.rms is not None:
                validate_single_channel_span(state, catalog, span)
            else:
                validate_span(catalog, span)

    masks = _edit_masks(state.masks, selection, action)
    if masks == state.masks:
        return state
    actions = state.interval_actions
    if reason is not None:
        changed = _mask_delta(state.masks, masks, action)
        actions += (IntervalAction(action, changed, reason),)
    rms = state.rms
    if rms is not None:
        rms = RmsReview(rms.windows, method_version=rms.method_version)
    return replace(state, masks=masks, interval_actions=actions, rms=rms)


def mask_spans(
    state: ReviewState, spans: Iterable[SampleSpan], *, catalog: SourceCatalog | None = None,
    reason: str | None = None,
) -> ReviewState:
    return _edit_intervals(state, spans, "mask", catalog, reason)


def restore_spans(
    state: ReviewState, spans: Iterable[SampleSpan], *, catalog: SourceCatalog | None = None,
    reason: str | None = None,
) -> ReviewState:
    return _edit_intervals(state, spans, "restore", catalog, reason)


def set_inclusion(state: ReviewState, blow_id: str, included: bool, reason: str) -> ReviewState:
    if not included:
        require_reason(reason, "A fit exclusion")
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


def add_groups(state: ReviewState, groups: Iterable[GroupWindow]) -> ReviewState:
    """Stage a group metadata batch as one review edit."""
    additions = tuple(groups)
    seen = {group.group_id for group in state.groups}
    for group in additions:
        if group.group_id in seen:
            raise ValueError(f"Line {group.source_row}: duplicate group ID {group.group_id!r}")
        seen.add(group.group_id)
    return replace(state, groups=state.groups + additions)


def set_group_reviewed(state: ReviewState, group_id: str, reviewed: bool) -> ReviewState:
    """Record the analyst's review of a group alongside ordinary history."""
    if not any(group.group_id == group_id for group in state.groups):
        raise ValueError(f"Unknown group ID {group_id!r}")
    return replace(state, groups=tuple(
        replace(group, reviewed=reviewed) if group.group_id == group_id else group for group in state.groups
    ))


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
    if before.mode != after.mode or before.single_channel_identity != after.single_channel_identity:
        raise ValueError("A review edit cannot switch its single-channel source or mode")
    if after.rms is not None and after.masks != before.masks:
        if after.rms.disposition != "candidate":
            raise ValueError("Changed masks require a new RMS review")
        if len(after.interval_actions) == len(before.interval_actions):
            raise ValueError("A single-channel mask change requires a recorded interval action")
    if before.receivers != after.receivers or before.timing_evidence != after.timing_evidence:
        raise ValueError("A review edit cannot rewrite source or timing evidence")
    if len(after.groups) < len(before.groups):
        raise ValueError("A review edit cannot remove imported groups")
    for old_group, new_group in zip(before.groups, after.groups):
        if old_group is not new_group and old_group != replace(new_group, reviewed=old_group.reviewed):
            raise ValueError(f"A review edit cannot rewrite imported group metadata for {old_group.group_id!r}")
    action_count = len(before.interval_actions)
    if after.interval_actions[:action_count] != before.interval_actions:
        raise ValueError("A review edit cannot rewrite recorded interval actions")
    if len(after.interval_actions) > action_count:
        masks = before.masks
        for action in after.interval_actions[action_count:]:
            updated = _edit_masks(masks, _merge_spans(action.spans), action.action)
            if _mask_delta(masks, updated, action.action) != action.spans:
                raise ValueError("An interval action must describe exactly the samples it changes")
            masks = updated
        if masks != after.masks:
            raise ValueError("Interval actions must account for the mask changes in their revision")
    after_by_id = {blow.blow_id: blow for blow in after.blows}
    for old in before.blows:
        new = after_by_id.get(old.blow_id)
        if new is None:
            raise ValueError(f"A review edit cannot remove blow {old.blow_id}")
        for role in ROLES:
            before_pick = getattr(old, role)
            after_pick = getattr(new, role)
            if before_pick.imported_ns != after_pick.imported_ns or before_pick.original_ref != after_pick.original_ref:
                raise ValueError(f"A review edit cannot rewrite imported picks for blow {old.blow_id}")


def _validate_applied_revision(before: ReviewState | None, revision: Revision) -> None:
    state = revision.state
    if state.rms is not None:
        if before is None:
            initial_rms = RmsReview(
                RmsWindows(state.rms.windows.source_hash, state.rms.windows.channel), method_version=state.rms.method_version,
            )
            if state.rms != initial_rms or state.masks:
                raise ValueError("The initial single-channel revision must be empty of RMS edits and masks")
        elif state.rms.baseline is None:
            raise ValueError("An applied single-channel revision requires an RMS baseline")
    previous_records = before.raw_evidence if before is not None else ()
    if before is None and state.interval_actions:
        raise ValueError("The initial revision cannot contain recorded interval actions")
    if state.raw_evidence[:len(previous_records)] != previous_records:
        raise ValueError("A review edit cannot rewrite captured raw evidence")
    new_records = state.raw_evidence[len(previous_records):]
    if any(record.applied_revision_id != revision.revision_id for record in new_records):
        raise ValueError("New raw evidence must originate in its applied revision")
    previous_decisions = set(before.pick_decisions) if before is not None else set()
    new_links = set()
    for decision in state.pick_decisions:
        if decision.evidence_id is None:
            if decision.status == "accepted" or decision.inspection_span is not None:
                raise ValueError("This pick decision requires a confirmed raw capture")
        elif decision not in previous_decisions:
            new_links.add(decision.evidence_id)
    if new_links != {record.evidence_id for record in new_records}:
        raise ValueError("A new evidence link must match a decision captured in the same revision")


def _overlaps_support(support: SampleSpan, span: SampleSpan) -> bool:
    return support.segment_id == span.segment_id and max(support.start, span.start) < min(support.stop, span.stop)


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
        _validate_applied_revision(None, timeline[0])
        for previous, following in zip(timeline, timeline[1:]):
            if type(following.revision_id) is not int or following.revision_id <= previous.revision_id:
                raise ValueError("Revision IDs must increase")
            _validate_transition(previous.state, following.state)
            _validate_applied_revision(previous.state, following)
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
        self._confirmed_draft: tuple[ReferenceType[ReviewHistory], Revision, RawConfirmation] | None = None

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
        previous, current = self.current.state.rms, state.rms
        if (previous is not None and current is not None and current.disposition == "accepted"
                and previous.windows != current.windows):
            raise ValueError("Changed windows or metadata need an explicit RMS review before acceptance")
        return self._preview(state, reason)

    def preview_rms_review(self, state: ReviewState, reason: str) -> Revision:
        """Stage a deliberate window disposition, including acceptance of newly reviewed support."""
        if state.rms is None:
            raise ValueError("An explicit RMS review requires a single-channel state")
        return self._preview(state, reason)

    def _preview(self, state: ReviewState, reason: str) -> Revision:
        _validate_transition(self.current.state, state)
        if state.raw_evidence != self.current.state.raw_evidence:
            raise ValueError("Raw evidence can only be captured at Apply")
        self._draft = Revision(self._next_revision_id, state, reason)
        self._confirmed_draft = None
        return self._draft

    def cancel(self) -> None:
        self._draft = None
        self._confirmed_draft = None

    def apply(self) -> Revision:
        if self._draft is not None and self._draft.state.rms is not None:
            raise ValueError("Use apply_rms to apply single-channel edits")
        return self._apply(self._draft)

    def apply_rms(self, catalog: SourceCatalog, *, checked_evidence: Iterable[RmsEvidence] = ()) -> Revision:
        """Capture the applied numerics independently of the staged human disposition."""
        draft = self._draft
        if draft is None:
            return self.current
        state = draft.state
        review = state.rms
        if review is None:
            raise ValueError("RMS apply requires a single-channel draft")
        if state == self.current.state:
            return self._apply(draft)
        if review.method_version != RMS_METHOD:
            raise ValueError("Unsupported RMS method version for a new review")
        validate_single_channel(catalog, state)
        for span in state.masks:
            validate_single_channel_span(state, catalog, span)
        result = measure_review(catalog, state, review.windows, checked_evidence=checked_evidence)
        state = replace(state, rms=replace(review, baseline=result))
        return self._apply(replace(draft, state=state))

    def replay_rms(self, catalog: SourceCatalog, *, checked_evidence: Iterable[RmsEvidence] = ()) -> RmsReplay:
        """Recompute the current single-channel revision without editing its baseline."""
        state = self.current.state
        if state.rms is None:
            raise ValueError("RMS replay requires a single-channel review")
        validate_single_channel(catalog, state)
        return inspect_rms_review(catalog, state, state.rms, checked_evidence=checked_evidence)

    def confirm_current_pick_evidence(
        self, catalog: SourceCatalog, decision: PickDecision, view: RawEvidenceView,
    ) -> RawConfirmation:
        """Bind the reviewer's confirmation to this draft until it is replaced or closed."""
        draft = self._draft
        if draft is None or decision not in draft.state.pick_decisions:
            raise ValueError("The decision to confirm must belong to the current review draft")
        confirmation = confirm_pick_evidence(catalog, decision, view)
        self._confirmed_draft = weakref(self), draft, confirmation
        return confirmation

    def apply_with_evidence(
        self, catalog: SourceCatalog, confirmation: RawConfirmation, current_view: RawEvidenceView | None, *,
        captured_utc_ns: int, software_versions: tuple[tuple[str, str], ...],
    ) -> Revision:
        """Revalidate one confirmed draft and commit it together with its raw capture."""
        draft = self._draft
        live = self._confirmed_draft
        if (
            draft is None or live is None or live[0]() is not self
            or live[1] is not draft or live[2] is not confirmation
        ):
            raise ValueError("Confirm the current review draft before Apply")
        if current_view is None or current_view != confirmation.view:
            raise ValueError("The raw view changed. Confirm the current raw view before Apply")
        if confirm_pick_evidence(catalog, confirmation.decision, current_view) != confirmation:
            raise ValueError("The raw samples or grid changed after confirmation")

        record = RawEvidence(
            f"raw-{draft.revision_id}", draft.revision_id, confirmation, captured_utc_ns, software_versions,
        )
        staged = replace(draft.state, raw_evidence=draft.state.raw_evidence + (record,))
        linked = replace(confirmation.decision, evidence_id=record.evidence_id)
        return self._apply(replace(draft, state=set_pick_decision(staged, catalog, linked)))

    def _apply(self, draft: Revision | None) -> Revision:
        if draft is not None and draft.state != self.current.state:
            _validate_applied_revision(self.current.state, draft)
            self._revisions = self._revisions[:self._cursor + 1] + (draft,)
            self._cursor += 1
            self._next_revision_id += 1
        self.cancel()
        return self.current

    def replay_pick_evidence(
        self, catalog: SourceCatalog, blow_id: str, role: Role, *, view: RawEvidenceView | None = None,
    ) -> RawReplay:
        """Report the current decision's raw replay and later mask changes independently."""
        decision = next((item for item in self.current.state.pick_decisions if item.target == (blow_id, role)), None)
        if decision is None or decision.evidence_id is None:
            return RawReplay()
        record = next(item for item in self.current.state.raw_evidence if item.evidence_id == decision.evidence_id)
        replay = inspect_raw_evidence(catalog, record, view=view)
        return replace(replay, masks_changed=self._masks_changed_after(record))

    def _masks_changed_after(self, record: RawEvidence) -> bool:
        support = record.confirmation.view.span
        for index in range(1, self._cursor + 1):
            revision = self._revisions[index]
            if revision.revision_id <= record.applied_revision_id:
                continue
            before, after = self._revisions[index - 1].state, revision.state
            new_actions = after.interval_actions[len(before.interval_actions):]
            if any(_overlaps_support(support, span) for action in new_actions for span in action.spans):
                return True
            if before.masks != after.masks:
                changed = _subtract_spans(before.masks, after.masks) + _subtract_spans(after.masks, before.masks)
                if any(_overlaps_support(support, span) for span in changed):
                    return True
        return False

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
