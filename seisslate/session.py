"""Versioned local review sessions and applied-revision CSV exports."""

from __future__ import annotations

import csv
import json
import os
import re
from collections.abc import Callable, Iterable, Mapping
from copy import copy
from dataclasses import asdict
from fractions import Fraction
from hashlib import file_digest, sha256
from math import ceil, isfinite
from pathlib import Path, PurePosixPath, PureWindowsPath
from sys import float_info
from tempfile import NamedTemporaryFile
from typing import IO, Any, cast

from .analysis import METHOD_VERSION, analyze
from .evidence import (
    IntervalAction, IntervalEdit, PickDecision, PickStatus, RawConfirmation, RawEvidence, RawEvidenceView, RawSamples,
    validate_span,
)
from .groups import GroupWindow
from .review import (
    Blow, PairStatus, Pick, Receiver, ReviewHistory, ReviewMode, ReviewState, Revision, Role,
    validate_pick_decision, validate_single_channel, validate_single_channel_span,
)
from .rms import (
    RMS_METHOD, RMS_REASON_ORDER, EpochEvidence, EpochStatus, RmsDisposition, RmsEvidence, RmsReason, RmsResult,
    RmsReview, RmsUnits, RmsWindow, RmsWindowResult, RmsWindows, UtcNs, validate_baseline_provenance,
)
from .sources import SampleRef, SampleSpan, SourceCatalog, SourceIdentityError, load_source


SCHEMA_VERSION = 3
_LEGACY_SESSION_KEYS = {"schema_version", "method_version", "sources", "revisions", "cursor", "next_revision_id"}
_SESSION_KEYS = {1: _LEGACY_SESSION_KEYS, 2: _LEGACY_SESSION_KEYS, 3: _LEGACY_SESSION_KEYS | {"mode"}}
_V1_STATE_KEYS = {"blows", "masks", "receivers", "timing_evidence"}
_V2_STATE_KEYS = _V1_STATE_KEYS | {"groups"}
_STATE_KEYS = {1: _V1_STATE_KEYS, 2: _V2_STATE_KEYS, 3: _V2_STATE_KEYS | {
    "pick_decisions", "interval_actions", "raw_evidence", "rms",
}}
_V1_BLOW_KEYS = {
    "blow_id", "original", "calibration", "distance_original_m", "distance_calibration_m",
    "included", "reason", "pair_status", "pair_reason",
}
_BLOW_KEYS = _V1_BLOW_KEYS | {"group_id"}
_GROUP_KEYS = {
    "group_id", "start_ns", "stop_ns", "start_utc", "end_utc", "duration_s", "source_file", "source_sha256",
    "source_row", "time_basis", "source_position_id", "original_channel", "calibration_channel", "timing_note",
    "geometry_note", "reviewed",
}


class SessionSourceError(ValueError):
    def __init__(self, message: str, path: Path, expected_hash: str) -> None:
        super().__init__(f"{message} {path}")
        self.path = path
        self.expected_hash = expected_hash


class MissingSourceError(SessionSourceError):
    """A saved recording is unavailable at its recorded location."""


class SourceMismatchError(SessionSourceError):
    """A recording does not have the saved fingerprint and size."""


def _object(raw: object, label: str, expected: set[str]) -> dict[str, Any]:
    if not isinstance(raw, dict) or raw.keys() != expected:
        raise ValueError(f"Unexpected fields in {label}")
    return raw


def _require[T](value: object, kind: type[T], field: str) -> T:
    if type(value) is not kind:
        raise ValueError(f"{field} must be of type {kind.__name__}")
    return cast(T, value)


def _nullable[T](value: object, kind: type[T], field: str) -> T | None:
    return None if value is None else _require(value, kind, field)


def _distance(value: object) -> float | None:
    if value is None:
        return None
    if type(value) is not int and type(value) is not float:
        raise ValueError("Receiver distances must be finite and nonnegative")
    if not 0 <= value <= float_info.max:
        raise ValueError("Receiver distances must be finite and nonnegative")
    return float(value)


def _ref(raw: object) -> SampleRef | None:
    if raw is None:
        return None
    value = _object(raw, "SampleRef", {"source_hash", "channel", "segment_id", "sample_index"})
    return SampleRef(
        _require(value["source_hash"], str, "source fingerprint"),
        _require(value["channel"], str, "channel"),
        _require(value["segment_id"], str, "segment ID"),
        _require(value["sample_index"], int, "sample index"),
    )


def _pick(raw: object) -> Pick:
    value = _object(raw, "Pick", {"imported_ns", "original_ref", "reviewed_ref"})
    return Pick(
        _nullable(value["imported_ns"], int, "imported UTC nanoseconds"),
        _ref(value["original_ref"]),
        _ref(value["reviewed_ref"]),
    )


def _blow(raw: object, schema_version: int) -> Blow:
    value = _object(raw, "Blow", _V1_BLOW_KEYS if schema_version == 1 else _BLOW_KEYS)
    return Blow(
        _require(value["blow_id"], str, "blow ID"),
        _pick(value["original"]),
        _pick(value["calibration"]),
        _distance(value["distance_original_m"]),
        _distance(value["distance_calibration_m"]),
        _require(value["included"], bool, "requested inclusion"),
        _require(value["reason"], str, "review reason"),
        cast(PairStatus, _require(value["pair_status"], str, "pair status")),
        _require(value["pair_reason"], str, "pair reason"),
        _nullable(value.get("group_id"), str, "group ID"),
    )


def _span(raw: object) -> SampleSpan:
    value = _object(raw, "SampleSpan", {"segment_id", "start", "stop"})
    return SampleSpan(
        _require(value["segment_id"], str, "segment ID"),
        _require(value["start"], int, "span start"),
        _require(value["stop"], int, "span stop"),
    )


def _receiver(raw: object) -> Receiver:
    value = _object(raw, "Receiver", {"role", "channel", "source_hash"})
    return Receiver(
        cast(Role, _require(value["role"], str, "receiver role")),
        _require(value["channel"], str, "receiver channel"),
        _require(value["source_hash"], str, "receiver source fingerprint"),
    )


def _decision(raw: object) -> PickDecision:
    value = _object(raw, "PickDecision", {
        "blow_id", "role", "reviewed_ref", "status", "lower_ref", "upper_ref", "reason", "reviewer",
        "intended_utc_ns", "inspection_span", "evidence_id",
    })
    span = value["inspection_span"]
    return PickDecision(
        _require(value["blow_id"], str, "blow ID"),
        cast(Role, _require(value["role"], str, "decision receiver role")),
        _ref(value["reviewed_ref"]),
        cast(PickStatus, _require(value["status"], str, "decision status")),
        _ref(value["lower_ref"]),
        _ref(value["upper_ref"]),
        _require(value["reason"], str, "decision reason"),
        _require(value["reviewer"], str, "decision reviewer"),
        _nullable(value["intended_utc_ns"], int, "intended UTC nanoseconds"),
        None if span is None else _span(span),
        _nullable(value["evidence_id"], str, "decision evidence ID"),
    )


def _interval_action(raw: object) -> IntervalAction:
    value = _object(raw, "IntervalAction", {"action", "spans", "reason"})
    return IntervalAction(
        cast(IntervalEdit, _require(value["action"], str, "interval action")),
        tuple(_span(span) for span in _require(value["spans"], list, "interval action spans")),
        _require(value["reason"], str, "interval action reason"),
    )


def _bounds(raw: object) -> tuple[SampleRef, SampleRef] | None:
    if raw is None:
        return None
    items = _require(raw, list, "raw view bounds")
    if len(items) != 2:
        raise ValueError("Raw view bounds require two sample references")
    lower, upper = map(_ref, items)
    if lower is None or upper is None:
        raise ValueError("Raw view bounds require two sample references")
    return lower, upper


def _raw_view(raw: object) -> RawEvidenceView:
    value = _object(raw, "RawEvidenceView", {
        "source_hash", "channel", "span", "display_start_ns", "display_stop_ns", "point", "bounds", "ordinate_scale",
    })
    return RawEvidenceView(
        _require(value["source_hash"], str, "raw view source fingerprint"),
        _require(value["channel"], str, "raw view channel"),
        _span(value["span"]),
        _require(value["display_start_ns"], int, "raw display start UTC"),
        _require(value["display_stop_ns"], int, "raw display stop UTC"),
        _ref(value["point"]),
        _bounds(value["bounds"]),
        _require(value["ordinate_scale"], str, "raw view ordinate scale"),
    )


def _raw_samples(raw: object) -> RawSamples:
    value = _object(raw, "RawSamples", {
        "segment_start_ns", "period_numerator", "period_denominator", "dtype", "sample_count", "digest",
    })
    return RawSamples(
        _require(value["segment_start_ns"], int, "raw segment start UTC"),
        _require(value["period_numerator"], int, "raw sample period numerator"),
        _require(value["period_denominator"], int, "raw sample period denominator"),
        _require(value["dtype"], str, "raw sample dtype"),
        _require(value["sample_count"], int, "raw sample count"),
        _require(value["digest"], str, "raw sample digest"),
    )


def _raw_version(raw: object) -> tuple[str, str]:
    items = _require(raw, list, "raw software version pair")
    if len(items) != 2:
        raise ValueError("Raw software version must contain a name and a version")
    return _require(items[0], str, "software name"), _require(items[1], str, "software version")


def _raw_evidence(raw: object) -> RawEvidence:
    value = _object(raw, "RawEvidence", {
        "evidence_id", "applied_revision_id", "confirmation", "captured_utc_ns", "software_versions", "method_id",
    })
    confirmation = _object(value["confirmation"], "RawConfirmation", {"decision", "view", "samples"})
    return RawEvidence(
        _require(value["evidence_id"], str, "raw evidence ID"),
        _require(value["applied_revision_id"], int, "raw applied revision ID"),
        RawConfirmation(
            _decision(confirmation["decision"]), _raw_view(confirmation["view"]), _raw_samples(confirmation["samples"]),
        ),
        _require(value["captured_utc_ns"], int, "capture UTC nanoseconds"),
        tuple(_raw_version(item) for item in _require(value["software_versions"], list, "software versions")),
        _require(value["method_id"], str, "raw evidence method ID"),
    )


def _rational(raw: object) -> Fraction:
    value = _object(raw, "RMS rational", {"numerator", "denominator"})
    numerator = _require(value["numerator"], int, "RMS rational numerator")
    denominator = _require(value["denominator"], int, "RMS rational denominator")
    if denominator <= 0:
        raise ValueError("RMS rational denominator must be positive")
    fraction = Fraction(numerator, denominator)
    if fraction.denominator != denominator:
        raise ValueError("RMS rational must use reduced terms")
    return fraction


def _rational_record(value: UtcNs) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def _rms_window(raw: object) -> RmsWindow | None:
    if raw is None:
        return None
    value = _object(raw, "RMS window intent", {"source_hash", "channel", "start_ns", "stop_ns"})
    return RmsWindow(
        _require(value["source_hash"], str, "RMS source hash"),
        _require(value["channel"], str, "RMS channel"),
        _rational(value["start_ns"]), _rational(value["stop_ns"]),
    )


def _rms_metadata(raw: object) -> RmsEvidence:
    value = _object(raw, "RMS metadata evidence", {
        "source_hash", "channel", "start_ns", "stop_ns", "locator", "sha256", "provenance",
    })
    return RmsEvidence(
        _require(value["source_hash"], str, "RMS evidence source hash"),
        _require(value["channel"], str, "RMS evidence channel"),
        _rational(value["start_ns"]), _rational(value["stop_ns"]),
        _require(value["locator"], str, "RMS evidence locator"),
        _require(value["sha256"], str, "RMS evidence SHA-256"),
        _require(value["provenance"], str, "RMS evidence provenance"),
    )


def _rms_epoch(raw: object) -> EpochEvidence | None:
    if raw is None:
        return None
    value = _object(raw, "RMS epoch evidence", {"reference", "verification", "conflicting"})
    return EpochEvidence(
        _rms_metadata(value["reference"]),
        _require(value["verification"], str, "RMS epoch verification"),
        _require(value["conflicting"], bool, "RMS epoch conflict"),
    )


def _rms_windows(raw: object) -> RmsWindows:
    value = _object(raw, "RMS windows", {
        "source_hash", "channel", "event", "background", "epoch_evidence", "counts_evidence",
    })
    return RmsWindows(
        _require(value["source_hash"], str, "RMS selected source hash"),
        _require(value["channel"], str, "RMS selected channel"),
        _rms_window(value["event"]), _rms_window(value["background"]), _rms_epoch(value["epoch_evidence"]),
        None if value["counts_evidence"] is None else _rms_metadata(value["counts_evidence"]),
    )


def _rms_number(value: object) -> float | None:
    if value is None:
        return None
    if (type(value) is not int and type(value) is not float) or (
        not -float_info.max <= value <= float_info.max or float(value) != value
    ):
        raise ValueError("RMS baseline values must be finite, representable numbers or null")
    return float(value)


def _rms_reasons(raw: object) -> tuple[RmsReason, ...]:
    reasons = tuple(_require(reason, str, "RMS reason") for reason in _require(raw, list, "RMS reasons"))
    ordered = tuple(reason for reason in RMS_REASON_ORDER if reason in reasons)
    if ordered != reasons:
        raise ValueError("RMS reasons must be known, unique and in canonical order")
    return ordered


def _rms_window_result(raw: object) -> RmsWindowResult:
    value = _object(raw, "RMS per-window baseline", {"span", "mean", "rms", "reasons"})
    span = None if value["span"] is None else _span(value["span"])
    mean, rms, reasons = _rms_number(value["mean"]), _rms_number(value["rms"]), _rms_reasons(value["reasons"])
    if reasons:
        if mean is not None or rms is not None:
            raise ValueError("Unavailable RMS windows must have null numeric values")
    elif span is None or mean is None or rms is None or rms < 0:
        raise ValueError("An available RMS window needs a valid span, mean and nonnegative RMS")
    return RmsWindowResult(span, mean, rms, reasons)


def _rms_result(raw: object) -> RmsResult | None:
    if raw is None:
        return None
    value = _object(raw, "RMS baseline", {
        "event", "background", "reasons", "contrast", "samples_per_window", "duration_ns", "epoch_status", "units",
    })
    epoch = _require(value["epoch_status"], str, "RMS epoch status")
    if epoch not in ("unknown", "unverified", "independently_verified", "conflicting"):
        raise ValueError("Unknown RMS epoch status")
    units = _require(value["units"], str, "RMS units")
    if units not in ("raw counts", "raw units unknown"):
        raise ValueError("Unknown RMS units")
    event = _rms_window_result(value["event"])
    background = _rms_window_result(value["background"])
    reasons = _rms_reasons(value["reasons"])
    contrast = _rms_number(value["contrast"])
    count = _nullable(value["samples_per_window"], int, "RMS sample count")
    duration = None if value["duration_ns"] is None else _rational(value["duration_ns"])
    if not set(event.reasons + background.reasons).issubset(reasons):
        raise ValueError("RMS baseline must include its window reasons")
    if contrast is not None:
        if reasons or contrast < 0 or event.rms is None or background.rms is None or background.rms <= 0:
            raise ValueError("Available RMS contrast needs two valid RMS values and no reasons")
    elif not reasons:
        raise ValueError("Unavailable RMS contrast needs a reason")
    if (count is None) != (duration is None) or count is not None and count <= 0 or duration is not None and duration <= 0:
        raise ValueError("RMS sample count and duration must both be positive or null")
    return RmsResult(
        event, background, reasons, contrast, count, duration, cast(EpochStatus, epoch), cast(RmsUnits, units),
    )


def _rms_review(raw: object) -> RmsReview:
    value = _object(raw, "single-channel RMS payload", {
        "method_version", "windows", "disposition", "reviewer", "reason", "baseline",
    })
    return RmsReview(
        _rms_windows(value["windows"]),
        cast(RmsDisposition, _require(value["disposition"], str, "RMS disposition")),
        _require(value["reviewer"], str, "RMS reviewer"),
        _require(value["reason"], str, "RMS decision reason"),
        _rms_result(value["baseline"]),
        _require(value["method_version"], str, "RMS method version"),
    )


def _revision(raw: object, schema_version: int, mode: ReviewMode) -> Revision:
    value = _object(raw, "Revision", {"revision_id", "state", "reason"})
    state = _object(value["state"], "ReviewState", _STATE_KEYS[schema_version])
    if mode == "paired" and state.get("rms") is not None:
        raise ValueError("A paired session must have a null RMS payload in every revision")
    return Revision(
        _require(value["revision_id"], int, "revision ID"),
        ReviewState(
            mode=mode,
            rms=_rms_review(state["rms"]) if mode == "single-channel" else None,
            blows=tuple(_blow(blow, schema_version) for blow in _require(state["blows"], list, "observations")),
            masks=tuple(_span(span) for span in _require(state["masks"], list, "masks")),
            receivers=tuple(_receiver(receiver) for receiver in _require(state["receivers"], list, "receivers")),
            timing_evidence=_require(state["timing_evidence"], str, "timing evidence"),
            groups=tuple(
                GroupWindow(**_object(raw_group, "GroupWindow", _GROUP_KEYS))
                for raw_group in _require(state.get("groups", []), list, "groups")
            ),
            pick_decisions=tuple(_decision(raw) for raw in _require(state.get("pick_decisions", []), list, "decisions")),
            interval_actions=tuple(
                _interval_action(raw) for raw in _require(state.get("interval_actions", []), list, "interval actions")
            ),
            raw_evidence=tuple(_raw_evidence(raw) for raw in _require(state.get("raw_evidence", []), list, "raw evidence")),
        ),
        _require(value["reason"], str, "revision reason"),
    )


def _validate_rms_spans(catalog: SourceCatalog, state: ReviewState, review: RmsReview, result: RmsResult) -> None:
    for intent, measured in zip(
        (review.windows.event, review.windows.background), (result.event, result.background), strict=True,
    ):
        span = measured.span
        if span is None:
            continue
        validate_single_channel_span(state, catalog, span)
        if intent is None:
            raise ValueError("An RMS span requires a window intent")
        if review.method_version == RMS_METHOD:
            segment = catalog.segment(span.segment_id)
            if not (
                segment.start_ns <= intent.start_ns < intent.stop_ns
                <= segment.start_ns + segment.npts * segment.period_ns
                and ceil((intent.start_ns - segment.start_ns) / segment.period_ns) == span.start
                and ceil((intent.stop_ns - segment.start_ns) / segment.period_ns) == span.stop
            ):
                raise ValueError("RMS span does not match its window intent and sample grid")

    validate_baseline_provenance(review.windows, result)
    if review.method_version != RMS_METHOD:
        return
    event, background = result.event.span, result.background.span
    count = duration = None
    if event is not None and background is not None and event.segment_id == background.segment_id:
        if event.stop - event.start == background.stop - background.start:
            count = event.stop - event.start
            duration = count * catalog.segment(event.segment_id).period_ns
    if result.samples_per_window != count or result.duration_ns != duration:
        raise ValueError("RMS span lengths do not match the baseline sample count and duration")


def _validate_references(catalog: SourceCatalog, history: ReviewHistory) -> None:
    channels = {(segment.source_hash, segment.channel) for segment in catalog.segments}
    if history.current.state.rms is not None:
        validate_single_channel(catalog, history.current.state)
    checked_actions = 0
    for revision in history.revisions:
        state = revision.state
        assigned: dict[str, tuple[str, str]] = {}
        for receiver in state.receivers:
            matching = {
                identity for identity in channels
                if identity[1] == receiver.channel and (not receiver.source_hash or identity[0] == receiver.source_hash)
            }
            if len(matching) != 1:
                raise ValueError(f"Receiver {receiver.role} must identify one source channel")
            assigned[receiver.role] = matching.pop()
        for blow in state.blows:
            _distance(blow.distance_original_m)
            _distance(blow.distance_calibration_m)
            for role, pick in (("original", blow.original), ("calibration", blow.calibration)):
                for ref in (pick.original_ref, pick.reviewed_ref):
                    if ref is None:
                        continue
                    catalog.segment(ref)
                    identity = ref.source_hash, ref.channel
                    if role in assigned and identity != assigned[role]:
                        raise ValueError(f"{blow.blow_id} has a pick outside its assigned {role} receiver")
                if pick.original_ref is not None:
                    ref = pick.original_ref
                    if pick.imported_ns is None or ref not in catalog.lookup(
                        ref.channel, pick.imported_ns, source_hash=ref.source_hash
                    ).candidates:
                        raise ValueError(f"{blow.blow_id} has an invalid original sample mapping")
        if state.rms is not None and state.rms.baseline is not None:
            _validate_rms_spans(catalog, state, state.rms, state.rms.baseline)
        for span in state.masks:
            if state.rms is not None:
                validate_single_channel_span(state, catalog, span)
            elif span.stop > catalog.segment(span.segment_id).npts:
                raise ValueError("Mask extends beyond its source segment")
        for decision in state.pick_decisions:
            validate_pick_decision(state, catalog, decision)
        for action in state.interval_actions[checked_actions:]:
            for span in action.spans:
                if state.rms is not None:
                    validate_single_channel_span(state, catalog, span)
                else:
                    validate_span(catalog, span)
        checked_actions = len(state.interval_actions)
        for record in state.raw_evidence:
            # History has already checked immutable carry-forward. Verify each origin once.
            if record.applied_revision_id == revision.revision_id:
                view = record.confirmation.view
                validate_span(catalog, view.span)
                segment = catalog.segment(view.span.segment_id)
                if (view.source_hash, view.channel) != (segment.source_hash, segment.channel):
                    raise ValueError("Raw evidence view does not identify its source channel")


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value = dict(pairs)
    if len(value) != len(pairs):
        raise ValueError("Duplicate JSON fields in session")
    return value


def _load_catalog(
    raw: object, parent: Path, relink: Mapping[str, Path | str], *, handoff: bool = False,
) -> SourceCatalog:
    sources = []
    seen = set()
    for entry in _require(raw, list, "sources"):
        value = _object(entry, "source", {"path", "sha256", "size_bytes"})
        fingerprint = _require(value["sha256"], str, "source fingerprint")
        size_bytes = _require(value["size_bytes"], int, "source size")
        recorded_path = _require(value["path"], str, "source path")
        if not re.fullmatch(r"[0-9a-f]{64}", fingerprint) or size_bytes <= 0 or fingerprint in seen:
            raise ValueError("Invalid or duplicate source identity")
        if not recorded_path or "\0" in recorded_path:
            raise ValueError("Invalid source path")
        seen.add(fingerprint)
        if fingerprint in relink:
            candidate = Path(relink[fingerprint])
        elif handoff:
            raise ValueError("A saved recording is absent from the handoff manifest")
        else:
            candidate = parent / recorded_path
            if recorded_path.replace("\\", "/").startswith("//") or candidate.as_posix().startswith("//"):
                raise MissingSourceError("Relink the network recording explicitly.", candidate, fingerprint)
        try:
            source = load_source(candidate, expected_sha256=fingerprint, expected_size_bytes=size_bytes)
        except OSError as error:
            raise MissingSourceError("Could not open the saved recording.", candidate, fingerprint) from error
        except SourceIdentityError as error:
            raise SourceMismatchError("The recording differs from the saved source.", candidate, fingerprint) from error
        sources.append(source)
    if relink.keys() - seen:
        raise ValueError("Relink references a fingerprint absent from the session")
    return SourceCatalog(tuple(sources))


def load_session(
    path: Path | str, *, relink: Mapping[str, Path | str] | None = None
) -> tuple[SourceCatalog, ReviewHistory]:
    """Open the applied review and history after verifying recordings by fingerprint."""
    path = Path(path)
    with path.open(encoding="utf-8") as file:
        raw = json.load(file, object_pairs_hook=_unique_keys)
    return _decode_session(raw, path.parent.resolve(), relink or {})


def _decode_session(
    raw: object, parent: Path, relink: Mapping[str, Path | str], *, handoff: bool = False,
) -> tuple[SourceCatalog, ReviewHistory]:
    if not isinstance(raw, dict):
        raise ValueError("Expected a session object")
    schema_version = _require(raw.get("schema_version"), int, "schema version")
    fields = _SESSION_KEYS.get(schema_version)
    if fields is None:
        raise ValueError("Unsupported session schema version")
    document = _object(raw, "session", fields)
    mode = _require(document["mode"], str, "session mode") if schema_version == 3 else "paired"
    if mode not in ("paired", "single-channel"):
        raise ValueError("Unsupported session mode")
    if _require(document["method_version"], str, "method version") != METHOD_VERSION:
        raise ValueError("Unsupported analysis method version")
    revisions = tuple(
        _revision(raw, schema_version, cast(ReviewMode, mode))
        for raw in _require(document["revisions"], list, "revisions")
    )
    if not revisions:
        raise ValueError("Session must contain an initial revision")
    history = ReviewHistory(
        revisions[0].state,
        revisions=revisions,
        cursor=_require(document["cursor"], int, "revision cursor"),
        next_revision_id=_require(document["next_revision_id"], int, "next revision ID"),
    )
    catalog = _load_catalog(document["sources"], parent, relink, handoff=handoff)
    _validate_references(catalog, history)
    return catalog, history


def _same_file(path: Path, other: Path) -> bool:
    try:
        return path.samefile(other)
    except FileNotFoundError:
        return False


def _file_sha256(path: Path) -> str:
    with path.open("rb") as file:
        return file_digest(file, "sha256").hexdigest()


def _retained_fingerprints(history: ReviewHistory) -> set[str]:
    fingerprints: set[str] = set()
    for revision in history.revisions:
        state = revision.state
        fingerprints.update(group.source_sha256 for group in state.groups)
        if state.rms is not None:
            windows = state.rms.windows
            if windows.counts_evidence is not None:
                fingerprints.add(windows.counts_evidence.sha256)
            if windows.epoch_evidence is not None:
                fingerprints.add(windows.epoch_evidence.reference.sha256)
    return fingerprints


def _protect_sources(
    path: Path, catalog: SourceCatalog, log_fingerprints: set[str], session_path: Path | None,
) -> None:
    resolved = path.resolve()
    if session_path is not None and (resolved == session_path.resolve() or _same_file(path, session_path)):
        raise ValueError("Choose an export destination outside the active review session")
    for source in catalog.sources:
        if resolved == source.path.resolve():
            raise ValueError("Choose a destination outside the recordings")
        if _same_file(path, source.path):
            raise ValueError("Destination aliases a source recording")
    if log_fingerprints and path.is_file():
        try:
            fingerprint = _file_sha256(path)
        except FileNotFoundError:
            return
        if fingerprint in log_fingerprints:
            raise ValueError("Destination matches an imported group log or RMS evidence. Choose another file.")


def _atomic_write(
    path: Path, catalog: SourceCatalog, history: ReviewHistory, write: Callable[[IO[str]], None],
    *, session_path: Path | str | None = None,
) -> None:
    log_fingerprints = _retained_fingerprints(history)
    active_session = None if session_path is None else Path(session_path)
    _protect_sources(path, catalog, log_fingerprints, active_session)
    temporary = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=path.parent, prefix=".seisslate-", delete=False
        ) as file:
            temporary = Path(file.name)
            write(file.file)
            file.flush()
            os.fsync(file.fileno())
        _protect_sources(path, catalog, log_fingerprints, active_session)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _recorded_path(source: Path, parent: Path) -> str:
    try:
        return Path(os.path.relpath(source, parent)).as_posix()
    except ValueError:
        # Windows cannot express a relative path across drives or network shares.
        return source.as_posix()


def _rms_record(review: RmsReview) -> dict[str, Any]:
    record = asdict(review)
    windows = record["windows"]
    epoch = windows["epoch_evidence"]
    for interval in (windows["event"], windows["background"], windows["counts_evidence"],
                     epoch["reference"] if epoch is not None else None):
        if interval is not None:
            for endpoint in ("start_ns", "stop_ns"):
                interval[endpoint] = _rational_record(interval[endpoint])
    if review.baseline is not None and review.baseline.duration_ns is not None:
        record["baseline"]["duration_ns"] = _rational_record(review.baseline.duration_ns)
    return record


def _revision_record(revision: Revision) -> dict[str, Any]:
    state = revision.state
    return {
        "revision_id": revision.revision_id,
        "state": {
            "blows": [asdict(item) for item in state.blows],
            "masks": [asdict(item) for item in state.masks],
            "receivers": [asdict(item) for item in state.receivers],
            "timing_evidence": state.timing_evidence,
            "groups": [asdict(item) for item in state.groups],
            "pick_decisions": [asdict(item) for item in state.pick_decisions],
            "interval_actions": [asdict(item) for item in state.interval_actions],
            "raw_evidence": [asdict(item) for item in state.raw_evidence],
            "rms": None if state.rms is None else _rms_record(state.rms),
        },
        "reason": revision.reason,
    }


def save_session(path: Path | str, catalog: SourceCatalog, history: ReviewHistory) -> None:
    """Atomically save committed history and its applied cursor. Draft edits stay in memory."""
    path = Path(path)
    _validate_references(catalog, history)
    parent = path.parent.resolve()
    sources = [
        {
            "path": _recorded_path(source.path, parent),
            "sha256": source.sha256,
            "size_bytes": source.size_bytes,
        }
        for source in catalog.sources
    ]
    document = {
        "schema_version": SCHEMA_VERSION,
        "mode": history.current.state.mode,
        "method_version": METHOD_VERSION,
        "sources": sources,
        "revisions": [_revision_record(revision) for revision in history.revisions],
        "cursor": history.cursor,
        "next_revision_id": history.next_revision_id,
    }

    def write(file: IO[str]) -> None:
        json.dump(document, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write("\n")

    _atomic_write(path, catalog, history, write)


_PICK_COLUMNS = (
    "imported_utc_ns", "reviewed_utc_ns", "mapping_offset_ns", "reviewed_offset_ns",
    "source_hash", "channel", "segment_id", "sample_index",
)
_CSV_COLUMNS = (
    "revision_id", "method_version", "blow_id", "status", "reason",
    "requested_included", "effective_included", "distance_original_m", "distance_calibration_m",
    "lag_ns", "lag_s", "dx_m", "residual_m", "fit_status", "fit_slope_m_per_s", "fit_intercept_m",
    *(f"{role}_{column}" for role in ("original", "calibration") for column in _PICK_COLUMNS),
)


def _csv_text(value: str) -> str:
    if value.startswith(("\t", "\r", "\n")) or value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def _csv_number(value: float | None) -> float | None:
    return value if value is not None and isfinite(value) and value >= 0 else None


def _pick_cells(catalog: SourceCatalog, pick: Pick, mapping_offset: int | None) -> tuple:
    ref = pick.reviewed_ref
    if ref is None:
        return pick.imported_ns, None, mapping_offset, None, "", "", "", None
    time_ns = catalog.time_ns(ref)
    reviewed_offset = None if pick.imported_ns is None else time_ns - pick.imported_ns
    return (
        pick.imported_ns, time_ns, mapping_offset, reviewed_offset,
        ref.source_hash, _csv_text(ref.channel), _csv_text(ref.segment_id), ref.sample_index,
    )


def export_csv(
    path: Path | str, catalog: SourceCatalog, history: ReviewHistory, *, session_path: Path | str | None = None,
) -> None:
    """Atomically export freshly computed observations from the applied revision."""
    snapshot = copy(history)
    revision = snapshot.current
    fit = analyze(catalog, snapshot.revisions[0].state, revision.state).reviewed

    def write(file: IO[str]) -> None:
        writer = csv.writer(file)
        writer.writerow(_CSV_COLUMNS)
        for blow, row in zip(revision.state.blows, fit.observations, strict=True):
            writer.writerow((
                revision.revision_id, METHOD_VERSION, _csv_text(row.blow_id), row.status, _csv_text(row.reason),
                str(row.requested_included).lower(), str(row.effective_included).lower(),
                _csv_number(blow.distance_original_m), _csv_number(blow.distance_calibration_m),
                row.lag_ns, row.lag_s, row.dx_m, row.residual_m, fit.status, fit.slope_m_per_s, fit.intercept_m,
                *_pick_cells(catalog, blow.original, row.original_mapping_offset_ns),
                *_pick_cells(catalog, blow.calibration, row.calibration_mapping_offset_ns),
            ))

    _atomic_write(Path(path), catalog, snapshot, write, session_path=session_path)


_EVIDENCE_CAPTURE_COLUMNS = (
    "evidence_id", "evidence_revision_id", "evidence_method_id", "evidence_captured_utc_ns",
    "evidence_source_hash", "evidence_channel", "evidence_segment_id", "evidence_span_start",
    "evidence_span_stop_exclusive", "evidence_raw_digest", "evidence_display_start_utc_ns",
    "evidence_display_stop_utc_ns", "evidence_ordinate_scale",
)
_EVIDENCE_COLUMNS = (
    "revision_id", "blow_id", "role", "decision_status", "decision_reason", "reviewer", "intended_utc_ns",
    "imported_utc_ns",
    *(f"{kind}_{column}" for kind in ("original", "reviewed")
      for column in ("sample_utc_ns", "source_hash", "channel", "segment_id", "sample_index")),
    "lower_bound_utc_ns", "lower_bound_sample_index", "upper_bound_utc_ns", "upper_bound_sample_index",
    *_EVIDENCE_CAPTURE_COLUMNS,
    "source_identity", "raw_samples", "grid_matches", "method_supported", "recorded_view_valid",
    "view_restoration", "mask_stale", "attestation_current",
)


def _sample_identity_cells(catalog: SourceCatalog, ref: SampleRef | None) -> tuple:
    if ref is None:
        return (None,) * 5
    return (catalog.time_ns(ref), _csv_text(ref.source_hash), _csv_text(ref.channel),
            _csv_text(ref.segment_id), ref.sample_index)


def _bound_cells(catalog: SourceCatalog, ref: SampleRef | None) -> tuple[int | None, int | None]:
    return (None, None) if ref is None else (catalog.time_ns(ref), ref.sample_index)


def _capture_cells(record: RawEvidence | None) -> tuple:
    if record is None:
        return (None,) * len(_EVIDENCE_CAPTURE_COLUMNS)
    view = record.confirmation.view
    return (
        _csv_text(record.evidence_id), record.applied_revision_id, _csv_text(record.method_id), record.captured_utc_ns,
        _csv_text(view.source_hash), _csv_text(view.channel), _csv_text(view.span.segment_id),
        view.span.start, view.span.stop, record.confirmation.samples.digest,
        view.display_start_ns, view.display_stop_ns, _csv_text(view.ordinate_scale),
    )


def _csv_flag(value: bool | None) -> str:
    return "" if value is None else str(value).lower()


def export_evidence_csv(
    path: Path | str, catalog: SourceCatalog, history: ReviewHistory, *, session_path: Path | str | None = None,
) -> None:
    """Export applied pick evidence and replay checks, one row per blow and receiver role."""
    snapshot = copy(history)
    revision = snapshot.current
    decisions = {decision.target: decision for decision in revision.state.pick_decisions}
    records = {record.evidence_id: record for record in revision.state.raw_evidence}

    def write(file: IO[str]) -> None:
        writer = csv.writer(file)
        writer.writerow(_EVIDENCE_COLUMNS)
        for blow in revision.state.blows:
            targets: tuple[tuple[Role, Pick], ...] = (("original", blow.original), ("calibration", blow.calibration))
            for role, pick in targets:
                decision = decisions.get((blow.blow_id, role))
                record = records.get(decision.evidence_id) if decision and decision.evidence_id else None
                replay = snapshot.replay_pick_evidence(catalog, blow.blow_id, role)
                writer.writerow((
                    revision.revision_id, _csv_text(blow.blow_id), role,
                    decision.status if decision else "legacy-unassessed",
                    _csv_text(decision.reason) if decision else "", _csv_text(decision.reviewer) if decision else "",
                    decision.intended_utc_ns if decision else None, pick.imported_ns,
                    *_sample_identity_cells(catalog, pick.original_ref), *_sample_identity_cells(catalog, pick.reviewed_ref),
                    *_bound_cells(catalog, decision.lower_ref if decision else None),
                    *_bound_cells(catalog, decision.upper_ref if decision else None),
                    *_capture_cells(record),
                    replay.source_identity, replay.raw_samples, _csv_flag(replay.grid_matches),
                    _csv_flag(replay.method_supported), _csv_flag(replay.recorded_view_valid),
                    "not_checked" if record else "not_recorded",
                    _csv_flag(replay.masks_changed if record else None), _csv_flag(replay.attestation_current if record else None),
                ))

    _atomic_write(Path(path), catalog, snapshot, write, session_path=session_path)


def _rms_interval_cells(record: RmsWindow | RmsEvidence | None, prefix: str) -> dict[str, str | int | None]:
    cells: dict[str, str | int | None] = {
        f"{prefix}_source_hash": _csv_text(record.source_hash) if record else "",
        f"{prefix}_channel": _csv_text(record.channel) if record else "",
    }
    for name, endpoint in (("start", record.start_ns if record else None), ("stop_exclusive", record.stop_ns if record else None)):
        rational = Fraction(endpoint) if endpoint is not None else None
        cells[f"{prefix}_{name}_utc_ns_numerator"] = rational.numerator if rational is not None else None
        cells[f"{prefix}_{name}_utc_ns_denominator"] = rational.denominator if rational is not None else None
    return cells


def _rms_evidence_cells(record: RmsEvidence | None, prefix: str) -> dict[str, str | int | None]:
    return {
        **_rms_interval_cells(record, prefix),
        **{f"{prefix}_{name}": _csv_text(getattr(record, name)) if record else ""
           for name in ("locator", "sha256", "provenance")},
    }


def _rms_result_cells(result: RmsResult | None, prefix: str) -> dict[str, str | float | int | None]:
    duration = result.duration_ns if result else None
    cells = {
        f"{prefix}_available": _csv_flag(result.available if result else None),
        f"{prefix}_qualified": _csv_flag(result.qualified if result else None),
        f"{prefix}_reasons": "|".join(result.reasons) if result else "",
        f"{prefix}_epoch_status": result.epoch_status if result else "",
        f"{prefix}_units": result.units if result else "",
        f"{prefix}_samples_per_window": result.samples_per_window if result else None,
        f"{prefix}_duration_ns_numerator": duration.numerator if duration is not None else None,
        f"{prefix}_duration_ns_denominator": duration.denominator if duration is not None else None,
        f"{prefix}_contrast": result.contrast if result else None,
    }
    for name, window in (("event", result.event if result else None), ("background", result.background if result else None)):
        span = window.span if window else None
        cells.update({
            f"{prefix}_{name}_segment_id": _csv_text(span.segment_id) if span else "",
            f"{prefix}_{name}_span_start": span.start if span else None,
            f"{prefix}_{name}_span_stop_exclusive": span.stop if span else None,
            f"{prefix}_{name}_mean": window.mean if window else None,
            f"{prefix}_{name}_rms": window.rms if window else None,
            f"{prefix}_{name}_reasons": "|".join(window.reasons) if window else "",
        })
    return cells


def export_rms_csv(
    path: Path | str, catalog: SourceCatalog, history: ReviewHistory, *, session_path: Path | str | None = None,
) -> None:
    """Export one applied RMS review, with separate historical and replay measurements."""
    snapshot = copy(history)
    revision = snapshot.current
    review = revision.state.rms
    if review is None:
        raise ValueError("RMS export requires a single-channel review")
    replay = snapshot.replay_rms(catalog)
    windows = review.windows
    epoch = windows.epoch_evidence
    cells = {
        "schema_version": SCHEMA_VERSION,
        "rms_csv_schema_version": 1,
        "revision_id": revision.revision_id,
        "method_version": _csv_text(review.method_version),
        "source_hash": _csv_text(windows.source_hash),
        "channel": _csv_text(windows.channel),
        "disposition": review.disposition,
        "reviewer": _csv_text(review.reviewer),
        "decision_reason": _csv_text(review.reason),
        "replay_status": replay.status,
        "structure_matches": _csv_flag(replay.structure_matches),
        "ready": _csv_flag(replay.ready),
        **_rms_interval_cells(windows.event, "event_intent"),
        **_rms_interval_cells(windows.background, "background_intent"),
        **_rms_result_cells(replay.baseline, "baseline"),
        **_rms_result_cells(replay.current, "current"),
        **_rms_evidence_cells(epoch.reference if epoch else None, "epoch_evidence"),
        "epoch_verification": _csv_text(epoch.verification) if epoch else "",
        "epoch_conflicting": _csv_flag(epoch.conflicting if epoch else None),
        **_rms_evidence_cells(windows.counts_evidence, "counts_evidence"),
    }
    comparisons = {check.quantity: check for check in replay.comparisons}
    for quantity in ("event_mean", "event_rms", "background_mean", "background_rms", "contrast"):
        check = comparisons.get(quantity)
        cells[f"{quantity}_delta"] = check.delta if check else None
        cells[f"{quantity}_tolerance"] = check.tolerance if check else None
        cells[f"{quantity}_matches"] = _csv_flag(check.matches if check else None)

    def write(file: IO[str]) -> None:
        writer = csv.writer(file)
        writer.writerow(cells.keys())
        writer.writerow(cells.values())

    _atomic_write(Path(path), catalog, snapshot, write, session_path=session_path)


def _handoff_path(record: object, root: Path) -> tuple[Path, str]:
    value = _object(record, "handoff file", {"path", "sha256"})
    name = _require(value["path"], str, "handoff relative path")
    digest = _require(value["sha256"], str, "handoff SHA-256")
    relative = PurePosixPath(name)
    if (
        not relative.parts or relative.is_absolute() or relative.as_posix() != name or ".." in relative.parts
        or any(char in name for char in ("\\", ":", "\0")) or PureWindowsPath(name).drive
        or re.fullmatch(r"[0-9a-f]{64}", digest) is None
    ):
        raise ValueError("Each handoff file needs a portable relative path and a SHA-256 fingerprint")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("A handoff file resolves outside the handoff directory")
    return path, digest


def _handoff_record(path: Path, root: Path, digest: str) -> dict[str, str]:
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("Place the session, waveforms and selected logs under the handoff directory")
    record = {"path": resolved.relative_to(root).as_posix(), "sha256": digest}
    _handoff_path(record, root)
    return record


def write_handoff_manifest(
    path: Path | str, session_path: Path | str, *, logs: Iterable[Path | str] = (),
) -> None:
    """Index a saved review, its recordings, and only the explicitly selected retained logs."""
    path, saved = Path(path), Path(session_path)
    root = path.parent.resolve()
    contents = saved.read_bytes()
    session_record = _handoff_record(saved, root, sha256(contents).hexdigest())
    catalog, history = _decode_session(
        json.loads(contents.decode("utf-8"), object_pairs_hook=_unique_keys), saved.parent.resolve(), {},
    )
    source_records = [_handoff_record(source.path, root, source.sha256) for source in catalog.sources]
    retained = _retained_fingerprints(history)
    log_records = []
    seen_paths = {session_record["path"], *(record["path"] for record in source_records)}
    for log in map(Path, logs):
        digest = _file_sha256(log)
        if digest not in retained:
            raise ValueError("A selected log does not match evidence retained in the saved session")
        record = _handoff_record(log, root, digest)
        if record["path"] in seen_paths:
            raise ValueError("Duplicate handoff file path")
        seen_paths.add(record["path"])
        log_records.append(record)
    document = {"schema_version": 1, "session": session_record, "sources": source_records, "logs": log_records}

    def write(file: IO[str]) -> None:
        json.dump(document, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write("\n")

    _atomic_write(path, catalog, history, write, session_path=saved)


def load_handoff(path: Path | str) -> tuple[SourceCatalog, ReviewHistory]:
    """Verify a portable handoff and open its saved revision without trusting evidence locators."""
    path = Path(path)
    root = path.parent.resolve()
    with path.open(encoding="utf-8") as file:
        raw = json.load(file, object_pairs_hook=_unique_keys)
    manifest = _object(raw, "handoff manifest", {"schema_version", "session", "sources", "logs"})
    if _require(manifest["schema_version"], int, "handoff schema version") != 1:
        raise ValueError("Unsupported handoff schema version")
    saved, digest = _handoff_path(manifest["session"], root)
    contents = saved.read_bytes()
    if sha256(contents).hexdigest() != digest:
        raise ValueError(f"Session SHA-256 mismatch for {saved.name}")
    seen_paths = {saved, path.resolve()}
    recordings: dict[str, Path | str] = {}
    for record in _require(manifest["sources"], list, "handoff sources"):
        source, source_digest = _handoff_path(record, root)
        if source in seen_paths or source_digest in recordings:
            raise ValueError("Duplicate handoff source path or fingerprint")
        seen_paths.add(source)
        recordings[source_digest] = source
    catalog, history = _decode_session(
        json.loads(contents.decode("utf-8"), object_pairs_hook=_unique_keys), saved.parent, recordings, handoff=True,
    )
    retained = _retained_fingerprints(history)
    for record in _require(manifest["logs"], list, "handoff logs"):
        log, digest = _handoff_path(record, root)
        if log in seen_paths:
            raise ValueError("Duplicate handoff file path")
        seen_paths.add(log)
        if digest not in retained:
            raise ValueError("A selected log does not match evidence retained in the saved session")
        try:
            actual = _file_sha256(log)
        except FileNotFoundError as error:
            raise ValueError(f"Missing handoff log {log.name}") from error
        if actual != digest:
            raise ValueError(f"Log SHA-256 mismatch for {log.name}")
    return catalog, history
