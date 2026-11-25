"""Versioned local review sessions and applied-revision CSV exports."""

from __future__ import annotations

import csv
import json
import os
import re
from collections.abc import Callable, Mapping
from dataclasses import asdict, fields
from hashlib import file_digest
from math import isfinite
from pathlib import Path
from sys import float_info
from tempfile import NamedTemporaryFile
from typing import IO, Any, cast

from .analysis import METHOD_VERSION, analyze
from .groups import GroupWindow
from .review import Blow, PairStatus, Pick, Receiver, ReviewHistory, ReviewState, Revision, Role
from .sources import SampleRef, SampleSpan, SourceCatalog, SourceIdentityError, load_source


SCHEMA_VERSION = 2
_V1_STATE_KEYS = {"blows", "masks", "receivers", "timing_evidence"}
_V2_STATE_KEYS = _V1_STATE_KEYS | {"groups"}


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


def _record(raw: object, cls: type, *, omitted: tuple[str, ...] = ()) -> dict[str, Any]:
    return _object(raw, cls.__name__, {field.name for field in fields(cls)}.difference(omitted))


def _require[T](value: object, kind: type[T], field: str) -> T:
    if type(value) is not kind:
        raise ValueError(f"{field} must be of type {kind.__name__}")
    return cast(T, value)


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
    value = _record(raw, SampleRef)
    return SampleRef(
        _require(value["source_hash"], str, "source fingerprint"),
        _require(value["channel"], str, "channel"),
        _require(value["segment_id"], str, "segment ID"),
        _require(value["sample_index"], int, "sample index"),
    )


def _pick(raw: object) -> Pick:
    value = _record(raw, Pick)
    imported_ns = value["imported_ns"]
    return Pick(
        None if imported_ns is None else _require(imported_ns, int, "imported UTC nanoseconds"),
        _ref(value["original_ref"]),
        _ref(value["reviewed_ref"]),
    )


def _blow(raw: object, schema_version: int) -> Blow:
    value = _record(raw, Blow, omitted=("group_id",) if schema_version == 1 else ())
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
        value.get("group_id"),
    )


def _span(raw: object) -> SampleSpan:
    value = _record(raw, SampleSpan)
    return SampleSpan(
        _require(value["segment_id"], str, "segment ID"),
        _require(value["start"], int, "mask start"),
        _require(value["stop"], int, "mask stop"),
    )


def _receiver(raw: object) -> Receiver:
    value = _record(raw, Receiver)
    return Receiver(
        cast(Role, _require(value["role"], str, "receiver role")),
        _require(value["channel"], str, "receiver channel"),
        _require(value["source_hash"], str, "receiver source fingerprint"),
    )


def _revision(raw: object, schema_version: int) -> Revision:
    value = _record(raw, Revision)
    state = _object(value["state"], "ReviewState", _V1_STATE_KEYS if schema_version == 1 else _V2_STATE_KEYS)
    return Revision(
        _require(value["revision_id"], int, "revision ID"),
        ReviewState(
            tuple(_blow(blow, schema_version) for blow in _require(state["blows"], list, "observations")),
            tuple(_span(span) for span in _require(state["masks"], list, "masks")),
            tuple(_receiver(receiver) for receiver in _require(state["receivers"], list, "receivers")),
            _require(state["timing_evidence"], str, "timing evidence"),
            tuple(
                GroupWindow(**_record(raw_group, GroupWindow))
                for raw_group in _require(state.get("groups", []), list, "groups")
            ),
        ),
        _require(value["reason"], str, "revision reason"),
    )


def _validate_references(catalog: SourceCatalog, history: ReviewHistory) -> None:
    channels = {(segment.source_hash, segment.channel) for segment in catalog.segments}
    for revision in history.revisions:
        assigned: dict[str, tuple[str, str]] = {}
        for receiver in revision.state.receivers:
            matching = {
                identity for identity in channels
                if identity[1] == receiver.channel and (not receiver.source_hash or identity[0] == receiver.source_hash)
            }
            if len(matching) != 1:
                raise ValueError(f"Receiver {receiver.role} must identify one source channel")
            assigned[receiver.role] = matching.pop()
        for blow in revision.state.blows:
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
        for span in revision.state.masks:
            if span.stop > catalog.segment(span.segment_id).npts:
                raise ValueError("Mask extends beyond its source segment")


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
    document = _object(raw, "session", {
        "schema_version", "method_version", "sources", "revisions", "cursor", "next_revision_id",
    })
    schema_version = _require(document["schema_version"], int, "schema version")
    if schema_version not in (1, SCHEMA_VERSION):
        raise ValueError("Unsupported session schema version")
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


def save_session(path: Path | str, catalog: SourceCatalog, history: ReviewHistory) -> None:
    """Atomically save committed history and its applied cursor. Draft edits stay in memory."""
    path = Path(path)
    revisions = []
    for revision in history.revisions:
        if revision.state.pick_decisions or revision.state.interval_actions:
            raise ValueError("Session version 2 cannot save pick decisions or interval actions")
        record = asdict(revision)
        record["state"] = {key: val for key, val in record["state"].items() if key in _V2_STATE_KEYS}
        revisions.append(record)
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
        "method_version": METHOD_VERSION,
        "sources": sources,
        "revisions": revisions,
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
