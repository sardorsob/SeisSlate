"""Prepare, review and reopen a bounded channel-by-band observation exercise.

Run as python -m tools.landslide_pilot from the SeisSlate checkout.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import csv
from dataclasses import dataclass
from fractions import Fraction
from hashlib import file_digest
import json
from math import ulp
from pathlib import Path
import re
import shutil
from typing import cast

import numpy as np
from obspy import Stream, Trace, UTCDateTime

from seisslate.landslide import BoundaryName, Status, Worksheet, WorksheetRow, build_rows
from seisslate.sources import NS_PER_SECOND, SampleRef, SourceCatalog, load_source, utc_ns


MANIFEST = "manifest.json"
TABLE = "review.csv"
WAVEFORMS = "snuffler.mseed"
PACK_VERSION = "seisslate-landslide-pack-v1"
REVIEW_VERSION = "seisslate-landslide-review-v1"
REVIEW_EVIDENCE_KEYS = {
    "Snuffler + ObsPy": {"table_sha256", "markers_sha256", "previous_review_sha256", "baseline"},
    "Qt worksheet": {"previous_review_sha256"},
}
BOUNDS: tuple[BoundaryName, ...] = ("onset", "end")
STATUSES: tuple[Status, ...] = (
    "unreviewed", "identifiable", "ambiguous", "no_identifiable_signal", "insufficient_support", "unavailable",
)
TABLE_COLUMNS = (
    "channel", "band", "snuffler_nslc", "onset_status", "onset_reason", "end_status", "end_reason",
)
REVIEW_CASE_KEYS = ("case_id", "channels", "start_utc", "stop_utc")


@dataclass(frozen=True)
class Pack:
    root: Path
    manifest: dict
    digest: str
    rows: tuple[WorksheetRow, ...]
    start_ns: int
    stop_ns: int

    @property
    def traces(self) -> Path:
        return self.root / WAVEFORMS


def digest_file(path: Path) -> str:
    with path.open("rb") as file:
        return file_digest(file, "sha256").hexdigest()


def write_json(path: Path, payload: dict) -> None:
    contents = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as file:
        file.write(contents)


def read_json(path: Path) -> dict:
    def no_duplicate_keys(pairs):
        result = dict(pairs)
        if len(result) != len(pairs):
            raise ValueError(f"Duplicate JSON keys in {path}")
        return result

    payload = json.loads(path.read_bytes(), object_pairs_hook=no_duplicate_keys)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def resolve_input(root: Path, name: str) -> Path:
    path = Path(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Invalid pack-relative path {name!r}")
    resolved = (root / path).resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ValueError(f"Pack-relative path escapes its root: {name!r}")
    return resolved


def row_aliases(channels: list[str]) -> dict[tuple[str, str], str]:
    """Reserve the MiniSEED location code for review band and raw view."""
    aliases = {}
    used = set()
    for index, original in enumerate(channels, start=1):
        if not re.fullmatch(r"[A-Z0-9]{1,2}\.[A-Z0-9]{1,5}\.[A-Z0-9]{0,2}\.[A-Z0-9]{1,3}", original):
            raise ValueError(f"Expected a MiniSEED 2 NSLC identifier: {original!r}")
        network, station, _, channel = original.split(".")
        clean_station = station
        while (network, clean_station, channel) in used:
            clean_station = f"R{index:04d}"
            index += 1
        used.add((network, clean_station, channel))
        for band in ("HF", "LP", "RW"):
            aliases[original, band] = ".".join((network, clean_station, band, channel))
    return aliases


def make_trace(alias: str, segment, first: int, samples) -> Trace:
    network, station, location, channel = alias.split(".")
    return Trace(samples.astype(np.float64, copy=True), header={
        "network": network, "station": station, "location": location, "channel": channel,
        "starttime": UTCDateTime(ns=segment.time_ns(first)), "sampling_rate": segment.sampling_rate_hz,
    })


def _write_prepared_traces(rows, aliases, start_ns, stop_ns, output):
    stream = Stream()
    for row in rows:
        if row.band == "HF":
            for segment in row.segments:
                first = segment.lower_bound(start_ns)
                stop = segment.lower_bound(stop_ns)
                if first < stop:
                    stream += make_trace(aliases[row.channel, "RW"], segment, first, segment.data[first:stop])
        for waveform in row.waveforms:
            support = waveform.support
            if support is not None:
                relative_start = support.start - waveform.first_index
                relative_stop = support.stop - waveform.first_index
                stream += make_trace(
                    aliases[row.channel, row.band], waveform.segment, support.start,
                    waveform.values[relative_start:relative_stop],
                )
    if not stream:
        raise ValueError("No raw waveform covers the requested channels and review window")
    stream.write(str(output), format="MSEED", encoding="FLOAT64")
    # MiniSEED 2 can round timestamps or encode a different rate from the input.
    # The reviewed plots and the displayed preparation must have the same grid.
    actual = load_source(output)
    expected = sorted((trace.id, trace.stats.starttime.ns, trace.stats.sampling_rate, trace.data.tobytes())
                      for trace in stream)
    observed = sorted((seg.channel, seg.start_ns, seg.sampling_rate_hz, seg.data.tobytes())
                      for seg in actual.segments)
    if expected != observed:
        raise ValueError("Prepared traces did not preserve the waveform sample starts, rates or values")


def prepare(case_file: Path, output: Path) -> None:
    """Make a fresh pack of pinned source bytes, band waveforms and a blind table."""
    case = read_json(case_file)
    required = set(REVIEW_CASE_KEYS) | {"sources"}
    if not required <= case.keys() or case.keys() - required - {"provenance", "reference_sha256"}:
        raise ValueError("Case fields are case_id, channels, start_utc, stop_utc, sources and optional "
                         "provenance and reference_sha256")
    if not isinstance(case["case_id"], str) or not case["case_id"].strip():
        raise ValueError("Case ID must be nonempty")
    if not isinstance(case.get("provenance", {}), dict):
        raise ValueError("Provenance must be an object")
    if (not isinstance(case["channels"], list) or not case["channels"]
            or not all(isinstance(channel, str) for channel in case["channels"])
            or len(set(case["channels"])) != len(case["channels"])):
        raise ValueError("List unique source NSLC identifiers")
    start_ns, stop_ns = utc_ns(case["start_utc"]), utc_ns(case["stop_utc"])
    if start_ns >= stop_ns:
        raise ValueError("Review start must precede review stop")
    aliases = row_aliases(case["channels"])
    if not isinstance(case["sources"], list) or not case["sources"]:
        raise ValueError("List at least one pinned source file")
    for specification in case["sources"]:
        if not isinstance(specification, dict) or specification.keys() != {"path", "sha256"}:
            raise ValueError("Each source must give path and sha256")
    sources = tuple(load_source(case_file.parent / spec["path"], expected_sha256=spec["sha256"])
                    for spec in case["sources"])
    rows = build_rows(SourceCatalog(sources), case["channels"], start_ns=start_ns, stop_ns=stop_ns)
    footprint = Worksheet(rows).to_dict()
    output.mkdir()
    try:
        (output / "sources").mkdir()
        inventories = []
        for index, source in enumerate(sources, start=1):
            name = f"sources/source-{index:02d}.mseed"
            shutil.copyfile(source.path, output / name)
            if digest_file(output / name) != source.sha256:
                raise ValueError(f"Source bytes changed during preparation: {source.path}")
            inventories.append({"path": name, "sha256": source.sha256, "size_bytes": source.size_bytes})
        _write_prepared_traces(rows, aliases, start_ns, stop_ns, output / WAVEFORMS)
        with (output / TABLE).open("x", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=TABLE_COLUMNS)
            writer.writeheader()
            for row in rows:
                writer.writerow({
                    "channel": row.channel, "band": row.band,
                    "snuffler_nslc": aliases[row.channel, row.band],
                    "onset_status": row.initial_status, "end_status": row.initial_status,
                })
        fields = {key: case[key] for key in (*REVIEW_CASE_KEYS, "provenance", "reference_sha256") if key in case}
        write_json(output / MANIFEST, {
            **fields, "schema": PACK_VERSION, "sources": inventories,
            "derived_sha256": digest_file(output / WAVEFORMS),
            "processing": footprint["processing"], "rows": footprint["rows"],
        })
    except Exception:
        shutil.rmtree(output)
        raise


def open_pack(path: Path) -> Pack:
    root = path.resolve(strict=True)
    manifest_path = resolve_input(root, MANIFEST)
    manifest = read_json(manifest_path)
    if manifest.get("schema") != PACK_VERSION:
        raise ValueError("Unsupported landslide pack schema")
    sources = tuple(load_source(
        resolve_input(root, item["path"]), expected_sha256=item["sha256"], expected_size_bytes=item["size_bytes"],
    ) for item in manifest["sources"])
    if digest_file(resolve_input(root, WAVEFORMS)) != manifest["derived_sha256"]:
        raise ValueError("Prepared Snuffler waveform SHA-256 differs")
    start, stop = utc_ns(manifest["start_utc"]), utc_ns(manifest["stop_utc"])
    rows = build_rows(SourceCatalog(sources), manifest["channels"], start_ns=start, stop_ns=stop)
    footprint = Worksheet(rows).to_dict()
    if any(footprint[key] != manifest[key] for key in ("rows", "processing")):
        raise ValueError("Prepared grids, support or processing differ from replay")
    return Pack(root, manifest, digest_file(manifest_path), rows, start, stop)


def snap_sample(row: WorksheetRow, target_ns: int | Fraction, uncertainty_ns: int | Fraction = 1) -> SampleRef:
    """Require a unique nearest sample inside eligible support, excluding midpoints."""
    matches = []
    for waveform in row.waveforms:
        support, segment = waveform.support, waveform.segment
        if support is None:
            continue
        position = (target_ns - segment.start_ns) / segment.period_ns
        index = round(position)
        if (
            support.start <= index < support.stop
            and segment.time_ns(support.start) - uncertainty_ns <= target_ns
            <= segment.time_ns(support.stop - 1) + uncertainty_ns
            and abs(position - index) * segment.period_ns < segment.period_ns / 2 - uncertainty_ns
        ):
            matches.append(segment.ref(index))
    if len(matches) != 1:
        raise ValueError(f"{row.channel} {row.band}: selection has no unique nearest supported sample")
    return matches[0]


def _snap_printed_utc(row: WorksheetRow, printed_utc: str) -> SampleRef:
    from tools.compare_measurements import marker_time

    date, time = printed_utc.split()
    stamp, uncertainty = marker_time(date, time)
    # The viewer projects a mouse coordinate in binary64 before formatting it.
    uncertainty += Fraction(ulp(stamp / NS_PER_SECOND)) * NS_PER_SECOND
    return snap_sample(row, stamp, uncertainty)


def _read_marker_picks(pack: Pack, markers_path: Path) -> dict:

    aliases = row_aliases(pack.manifest["channels"])
    by_alias = {aliases[row.channel, row.band]: row for row in pack.rows}
    contents = markers_path.read_text(encoding="utf-8").splitlines()
    if not contents or contents[0].strip() != "# Snuffler Markers File Version 0.3":
        raise ValueError("Expected a Snuffler 0.3 marker file")
    picks = {}
    for line in contents[1:]:
        tokens = line.split()
        if not tokens or (len(tokens) in (4, 7) and tokens[-2] == "0"):
            continue  # optional unclassified navigation marks
        if len(tokens) != 4 or tokens[2] not in ("1", "2"):
            raise ValueError("Use one plain point marker on one band trace, kind 1 (onset) or 2 (end)")
        date, time, kind, alias = tokens
        row = by_alias.get(alias)
        if row is None:
            raise ValueError("A boundary marker must name one prepared band trace")
        boundary = BOUNDS[int(kind) - 1]
        key = (row.channel, row.band, boundary)
        if key in picks:
            raise ValueError(f"Multiple {boundary} markers on {row.channel} {row.band}")
        printed_utc = f"{date} {time}"
        picks[key] = (_snap_printed_utc(row, printed_utc), printed_utc)
    return picks


def _duration(row: WorksheetRow, onset: SampleRef | None, end: SampleRef | None) -> tuple[str, float | None]:
    """Independently compute baseline elapsed time on the original ObsPy sample grid."""
    if onset is None or end is None:
        return "missing_boundary", None
    first, last = onset.sample_index, end.sample_index
    for waveform in row.waveforms:
        support = waveform.support
        if (support and onset.segment_id == end.segment_id == support.segment_id
                and support.start <= min(first, last) <= max(first, last) < support.stop):
            if last <= first:
                return "unordered", None
            return "eligible", float((last - first) * waveform.segment.period_ns / NS_PER_SECOND)
    return "separate_support", None


def save_review(
    pack: Pack, worksheet: Worksheet, output: Path, *, method: str, evidence: dict, reviewer: str = "",
) -> None:
    write_json(output, {
        "schema": REVIEW_VERSION, "case_id": pack.manifest["case_id"], "pack_sha256": pack.digest,
        "method": method, "reviewer": reviewer, "evidence": evidence, "worksheet": worksheet.to_dict(),
    })


def _audit_baseline_row(row, result, timestamps):
    if not isinstance(timestamps, dict) or set(timestamps) != set(BOUNDS):
        raise ValueError("Baseline evidence requires both marker times")
    for boundary, timestamp in timestamps.items():
        if timestamp is not None and not isinstance(timestamp, str):
            raise ValueError("Invalid stored marker time")
        sample = _snap_printed_utc(row, timestamp) if timestamp is not None else None
        if sample != getattr(result, boundary).sample:
            raise ValueError(f"Stored marker time does not reproduce for {row.channel} {row.band} {boundary}")
    duration_status, duration = _duration(row, result.onset.sample, result.end.sample)
    if (duration_status, duration) != (result.duration_status, result.duration_seconds):
        raise ValueError(f"Independent duration does not reproduce for {row.channel} {row.band}")
    return {
        "channel": row.channel, "band": row.band, "marker_utc": timestamps,
        "duration_status": duration_status, "observed_signal_duration_seconds": duration,
    }


def reopen(pack_path: Path | Pack, review_path: Path) -> Worksheet:
    pack = pack_path if isinstance(pack_path, Pack) else open_pack(pack_path)
    review = read_json(review_path)
    method = review.get("method")
    if (review.get("schema") != REVIEW_VERSION or review.get("pack_sha256") != pack.digest
            or not isinstance(method, str) or method not in REVIEW_EVIDENCE_KEYS
            or review.get("case_id") != pack.manifest["case_id"]):
        raise ValueError("Review does not belong to this prepared pack")
    metadata = review.get("evidence")
    if not isinstance(metadata, dict) or metadata.keys() != REVIEW_EVIDENCE_KEYS[method]:
        raise ValueError("Evidence fields do not match the review method")
    for key in REVIEW_EVIDENCE_KEYS[method] - {"baseline"}:
        value = metadata[key]
        if value is None and key == "previous_review_sha256":
            continue
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError(f"Invalid review evidence digest {key}")
    worksheet = Worksheet.from_dict(pack.rows, review["worksheet"])
    if method == "Snuffler + ObsPy":
        baseline = metadata["baseline"]
        if not isinstance(baseline, list) or len(baseline) != len(pack.rows):
            raise ValueError("Missing independent baseline evidence")
        for row, evidence in zip(pack.rows, baseline, strict=True):
            result = worksheet.result(row.channel, row.band)
            if not isinstance(evidence, dict) or evidence != _audit_baseline_row(row, result, evidence.get("marker_utc")):
                raise ValueError(f"Baseline evidence does not reproduce for {row.channel} {row.band}")
    return worksheet


def score(
    pack_path: Path, table: Path, markers: Path, output: Path, *, reviewer: str = "", previous: Path | None = None,
) -> None:
    """Apply statuses and snapped markers, and audit each duration against baseline arithmetic."""
    pack = open_pack(pack_path)
    worksheet = reopen(pack, previous) if previous else Worksheet(pack.rows)
    rows: dict[tuple[str, str], WorksheetRow] = {(row.channel, row.band): row for row in pack.rows}
    aliases = row_aliases(pack.manifest["channels"])
    picks = _read_marker_picks(pack, markers)
    seen = set()
    comparisons = {}
    with table.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None or sorted(reader.fieldnames) != sorted(TABLE_COLUMNS):
            raise ValueError("Use the prepared review.csv fields")
        for record in reader:
            key = (record["channel"], record["band"])
            row = rows.get(key)
            if row is None or key in seen or record["snuffler_nslc"] != aliases[key] or None in record:
                raise ValueError("Review table contains a duplicate or altered trace identity")
            seen.add(key)
            timestamps = {}
            for boundary in BOUNDS:
                status, reason = record[f"{boundary}_status"], record[f"{boundary}_reason"]
                sample, timestamp = picks.get((*key, boundary), (None, None))
                if status not in STATUSES or reason is None:
                    raise ValueError("Review table contains an incomplete row or unsupported status")
                if (sample is not None) != (status == "identifiable"):
                    raise ValueError(f"{row.channel} {row.band} {boundary}: identifiable status requires one marker")
                current = getattr(worksheet.result(row.channel, row.band), boundary)
                if (current.status, current.reason, current.sample) != (status, reason, sample):
                    worksheet.apply(worksheet.preview(row.channel, row.band, boundary, cast(Status, status), reason, sample))
                timestamps[boundary] = timestamp
            comparisons[key] = _audit_baseline_row(row, worksheet.result(row.channel, row.band), timestamps)
    if seen != rows.keys():
        raise ValueError("Review table is missing a prepared row")
    save_review(pack, worksheet, output, method="Snuffler + ObsPy", reviewer=reviewer, evidence={
        "table_sha256": digest_file(table), "markers_sha256": digest_file(markers),
        "previous_review_sha256": digest_file(previous) if previous else None,
        "baseline": [comparisons[key] for key in rows],
    })


def snuffle(pack_path: Path, output: Path, initial_markers: Path | None = None) -> None:
    """Launch the stock viewer with the prepared band traces and its filter controls reset."""
    from pyrocko import pile
    from pyrocko.gui.snuffler import snuffle as stock_snuffler
    from pyrocko.gui.snuffler.marker import Marker

    if output.exists():
        raise FileExistsError(output)
    pack = open_pack(pack_path)
    markers = Marker.load_markers(str(initial_markers)) if initial_markers is not None else []

    def configure(window):
        viewer = window.get_view()
        viewer.menuitem_demean.setChecked(False)
        viewer.menuitem_degap.setChecked(False)
        viewer.menuitem_fft_filtering.setChecked(False)
        viewer.menuitem_remove_resp.setChecked(False)
        viewer.menuitem_allowdownsampling.setChecked(False)
        viewer.lowpass_change(None)
        viewer.highpass_change(None)
        viewer.set_time_range(pack.start_ns / NS_PER_SECOND, pack.stop_ns / NS_PER_SECOND)

    prepared = pile.make_pile([str(pack.traces)], fileformat="mseed")
    if not prepared.nslc_ids:
        raise ValueError("Snuffler could not load the prepared traces")
    _, reviewed = stock_snuffler(
        prepared, markers=markers, opengl=False,
        want_markers=True, launch_hook=configure, ntracks=min(12, 3 * len(pack.manifest["channels"])),
    )
    # Pyrocko owns its filename-only export API. Stage into a temporary file, then
    # publish exclusively so a source or previous review cannot be overwritten.
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as folder:
        staged = Path(folder) / "review.markers"
        Marker.save_markers(reviewed, str(staged), fdigits=9)
        with output.open("xb") as target:
            target.write(staged.read_bytes())


def reveal(pack_path: Path, review: Path, reference_csv: Path, output: Path) -> None:
    """Write source CSV rows beside an already saved independent review, without grading them."""
    pack = open_pack(pack_path)
    worksheet = reopen(pack, review)
    if any(boundary.status == "unreviewed"
           for row in worksheet.rows for boundary in (worksheet.result(row.channel, row.band).onset,
                                                      worksheet.result(row.channel, row.band).end)):
        raise ValueError("Finish the independent review before revealing published rows")
    reference_sha = digest_file(reference_csv)
    if "reference_sha256" in pack.manifest and pack.manifest["reference_sha256"] != reference_sha:
        raise ValueError("Published CSV SHA-256 differs from the case pin")
    selected = set(pack.manifest["channels"])
    with reference_csv.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        keys = {"Network", "Station", "LocationCode", "Channel"}
        keys.update(f"{prefix}{band}{suffix}"
                    for band in ("HF", "LP")
                    for prefix, suffix in (("detect_", ""), ("starttime", "_UTC"), ("endtime", "_UTC")))
        if reader.fieldnames is None or not keys <= set(reader.fieldnames):
            raise ValueError("Expected the USGS example CSV and its detection and timestamp columns")
        published = []
        for record in reader:
            channel = ".".join(record[field].strip() for field in ("Network", "Station", "LocationCode", "Channel"))
            if channel in selected:
                published.append(record)
    write_json(output, {
        "case_id": pack.manifest["case_id"], "independent_review_sha256": digest_file(review),
        "published_csv_sha256": reference_sha, "published_rows": published,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    paths = {
        "prepare": ("case_file", "output"),
        "snuffle": ("pack_path", "output"),
        "score": ("pack_path", "table", "markers", "output"),
        "reopen": ("pack_path", "review"),
        "reveal": ("pack_path", "review", "reference_csv", "output"),
        "worksheet": ("pack_path", "output"),
    }
    for command, arguments in paths.items():
        subparser = commands.add_parser(command)
        for argument in arguments:
            subparser.add_argument(argument, type=Path)
        if command == "snuffle":
            subparser.add_argument("--initial-markers", type=Path)
        if command == "score":
            subparser.add_argument("--previous", type=Path, help="Previous applied review by this reviewer")
        if command in ("score", "worksheet"):
            subparser.add_argument("--reviewer", default="", help="Optional reviewer label")
        if command == "worksheet":
            subparser.add_argument("--review", type=Path, help="Open an applied review from this pack")
    values = vars(parser.parse_args())
    command = values.pop("command")
    try:
        if command == "reopen":
            worksheet = reopen(values["pack_path"], values["review"])
            print(json.dumps(worksheet.to_dict(), indent=2, allow_nan=False))
            return
        if command == "worksheet":
            from tools.landslide_worksheet import launch

            launch(**values)
            return
        handlers: dict[str, Callable[..., None]] = {"prepare": prepare, "snuffle": snuffle, "score": score, "reveal": reveal}
        handlers[command](**values)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    main()
