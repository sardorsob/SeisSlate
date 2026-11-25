"""Versioned local review sessions and applied-revision CSV exports."""

from __future__ import annotations

import csv
import json
import os
import re
from collections.abc import Callable, Mapping
from dataclasses import asdict
from hashlib import file_digest
from math import isfinite
from pathlib import Path
from sys import float_info
from tempfile import NamedTemporaryFile
from typing import IO, Any, cast

from .analysis import METHOD_VERSION, analyze
from .evidence import (
    IntervalAction, IntervalEdit, PickDecision, PickStatus, RawConfirmation, RawEvidence, RawEvidenceView, RawSamples,
    validate_span,
)
from .groups import GroupWindow
from .review import Blow, PairStatus, Pick, Receiver, ReviewHistory, ReviewState, Revision, Role, validate_pick_decision
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


def _revision(raw: object, schema_version: int) -> Revision:
    value = _object(raw, "Revision", {"revision_id", "state", "reason"})
    state = _object(value["state"], "ReviewState", _STATE_KEYS[schema_version])
    if state.get("rms") is not None:
        raise ValueError("A paired session must have a null RMS payload in every revision")
    return Revision(
        _require(value["revision_id"], int, "revision ID"),
        ReviewState(
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


def _validate_references(catalog: SourceCatalog, history: ReviewHistory) -> None:
    channels = {(segment.source_hash, segment.channel) for segment in catalog.segments}
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
        for span in state.masks:
            if span.stop > catalog.segment(span.segment_id).npts:
                raise ValueError("Mask extends beyond its source segment")
        for decision in state.pick_decisions:
            validate_pick_decision(state, catalog, decision)
        for action in state.interval_actions[checked_actions:]:
            for span in action.spans:
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


def _load_catalog(raw: object, parent: Path, relink: Mapping[str, Path | str]) -> SourceCatalog:
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
    if not isinstance(raw, dict):
        raise ValueError("Expected a session object")
    schema_version = _require(raw.get("schema_version"), int, "schema version")
    fields = _SESSION_KEYS.get(schema_version)
    if fields is None:
        raise ValueError("Unsupported session schema version")
    document = _object(raw, "session", fields)
    if schema_version == 3:
        mode = _require(document["mode"], str, "session mode")
        if mode == "single-channel":
            raise ValueError("Open this session in a SeisSlate version that supports single-channel RMS review")
        if mode != "paired":
            raise ValueError("Unsupported session mode")
    if _require(document["method_version"], str, "method version") != METHOD_VERSION:
        raise ValueError("Unsupported analysis method version")
    revisions = tuple(_revision(raw, schema_version) for raw in _require(document["revisions"], list, "revisions"))
    if not revisions:
        raise ValueError("Session must contain an initial revision")
    history = ReviewHistory(
        revisions[0].state,
        revisions=revisions,
        cursor=_require(document["cursor"], int, "revision cursor"),
        next_revision_id=_require(document["next_revision_id"], int, "next revision ID"),
    )
    catalog = _load_catalog(document["sources"], path.parent.resolve(), relink or {})
    _validate_references(catalog, history)
    return catalog, history


def _protect_sources(path: Path, catalog: SourceCatalog, log_fingerprints: set[str]) -> None:
    resolved = path.resolve()
    for source in catalog.sources:
        if resolved == source.path.resolve():
            raise ValueError("Choose a destination outside the recordings")
        try:
            same_file = path.samefile(source.path)
        except FileNotFoundError:
            same_file = False
        if same_file:
            raise ValueError("Destination aliases a source recording")
    if log_fingerprints and path.is_file():
        try:
            with path.open("rb") as file:
                fingerprint = file_digest(file, "sha256").hexdigest()
        except FileNotFoundError:
            return
        if fingerprint in log_fingerprints:
            raise ValueError("Destination matches an imported group log. Choose another file.")


def _atomic_write(
    path: Path, catalog: SourceCatalog, history: ReviewHistory, write: Callable[[IO[str]], None],
) -> None:
    log_fingerprints = {group.source_sha256 for revision in history.revisions for group in revision.state.groups}
    _protect_sources(path, catalog, log_fingerprints)
    temporary = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=path.parent, prefix=".seisslate-", delete=False
        ) as file:
            temporary = Path(file.name)
            write(file.file)
            file.flush()
            os.fsync(file.fileno())
        _protect_sources(path, catalog, log_fingerprints)
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


def _paired_revision_record(revision: Revision) -> dict[str, Any]:
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
            "rms": None,
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
        "mode": "paired",
        "method_version": METHOD_VERSION,
        "sources": sources,
        "revisions": [_paired_revision_record(revision) for revision in history.revisions],
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


def export_csv(path: Path | str, catalog: SourceCatalog, history: ReviewHistory) -> None:
    """Atomically export freshly computed observations from the applied revision."""
    revision = history.current
    fit = analyze(catalog, history.revisions[0].state, revision.state).reviewed

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

    _atomic_write(Path(path), catalog, history, write)
