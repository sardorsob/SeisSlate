"""Produce a raw-sample RMS reference for segment-pinned MiniSEED 2 with integral rates.

A source inventory must independently specify the segment grid and sample hashes.
Cases without an expected result produce reference values with unassessed status.
"""

from __future__ import annotations

import argparse
import calendar
import csv
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, localcontext
from fractions import Fraction
from hashlib import sha256
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import re
import shlex
from sys import float_info
from typing import Any

import numpy as np
from pyrocko import mseed_ext


PYROCKO_VERSION = "2026.11.5"
NS_PER_SECOND = 1_000_000_000
MAX_WINDOW_SAMPLES = 1_000_000
MARKER_FORMAT = "Snuffler 0.3 named phase intervals"
REASONS = (
    "missing_window", "cross_segment", "out_of_bounds", "unequal_length",
    "too_short", "too_large", "overlapping_windows", "ambiguous_coverage",
    "masked_samples", "nonfinite_samples", "zero_background", "numeric_failure",
)
NUMERIC_FIELDS = {
    "event_mean", "event_rms", "background_mean", "background_rms", "contrast",
}
Interval = tuple[Fraction, Fraction]
Mask = tuple[int, int, int]


@dataclass(frozen=True)
class Segment:
    index: int
    channel: str
    start: int
    period: Fraction
    samples: np.ndarray

    @property
    def stop(self) -> Fraction:
        return self.start + len(self.samples) * self.period

    def grid_index(self, utc_ns: Fraction) -> Fraction:
        return (utc_ns - self.start) / self.period


@dataclass(frozen=True)
class Window:
    support: tuple[int, int, int] | None = None
    mean: float | None = None
    rms: float | None = None
    variance: Fraction | None = None
    reasons: tuple[str, ...] = ()

    def row(self, name: str) -> dict[str, Any]:
        return {
            f"{name}_segment": self.support[0] if self.support else None,
            f"{name}_indices": list(self.support[1:]) if self.support else None,
            f"{name}_mean": self.mean,
            f"{name}_rms": self.rms,
            f"{name}_reasons": list(self.reasons),
        }


def require_fields(value: dict, keys: str, context: str, *, optional: str = "") -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    missing = set(keys.split()).difference(value)
    unknown = set(value).difference(keys.split(), optional.split())
    if missing or unknown:
        raise ValueError(f"{context} fields: missing {sorted(missing)}, unknown {sorted(unknown)}")


def unique_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate manifest field {key!r}")
        result[key] = value
    return result


def reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"Nonfinite manifest token {value}")


def sample_sha256(samples: np.ndarray) -> str:
    """Hash sample bytes in little-endian order, independent of host byte order."""
    canonical = samples.astype(samples.dtype.newbyteorder("<"), copy=False)
    return sha256(canonical.tobytes()).hexdigest()


def read_source(path: Path, spec: dict) -> tuple[dict, tuple[Segment, ...]]:
    require_fields(spec, "path sha256 size_bytes segments", "Source")
    if version("pyrocko") != PYROCKO_VERSION or mseed_ext.HPTMODULUS != 1_000_000:
        raise ValueError(f"This reference requires Pyrocko {PYROCKO_VERSION} with microsecond MiniSEED ticks")
    path = path / spec["path"]
    source_hash = sha256(path.read_bytes()).hexdigest()
    if (type(spec["size_bytes"]) is not int or path.stat().st_size != spec["size_bytes"]
            or source_hash != spec["sha256"]):
        raise ValueError(f"Source size or SHA-256 differs for {path.name}")

    # The pinned segment and sample inventory must detect a reader merge or split.
    # Use libmseed's integer timestamps to avoid an epoch float round trip.
    traces = sorted(
        mseed_ext.get_traces(str(path), True, 0, 0),
        key=lambda item: (item[1:5], item[5], -item[7], item[8].size),
    )
    if not isinstance(spec["segments"], list) or not traces or len(traces) != len(spec["segments"]):
        raise ValueError("Pyrocko segment inventory differs from the source manifest")
    segments = []
    verified = []
    for index, (raw, expected) in enumerate(zip(traces, spec["segments"], strict=True)):
        require_fields(expected, "channel start_ns sampling_rate_hz npts dtype sample_sha256", "Segment")
        _, net, sta, loc, component, start_ticks, end_ticks, read_rate, samples, *_ = raw
        if samples.ndim != 1 or samples.size == 0 or samples.dtype.kind not in "iuf":
            raise ValueError(f"Unsupported MiniSEED samples in segment {index}")
        # MiniSEED 2 factor/multiplier rates near an integer are distinguishable
        # in binary64, and blockette 100 values already use binary32. Non-integral
        # decoded rates need an independently supported exact-rate reader.
        if not 0 < read_rate <= NS_PER_SECOND or not read_rate.is_integer():
            raise ValueError(f"Pyrocko segment {index} sampling_rate_hz is outside the integral-rate profile")
        rate = int(read_rate)
        expected_rate = expected["sampling_rate_hz"]
        if type(expected_rate) not in (int, str) or rate != Fraction(expected_rate):
            raise ValueError(f"Pyrocko segment {index} sampling_rate_hz differs from the source manifest")
        period = Fraction(NS_PER_SECOND, rate)
        start = int(start_ticks) * 1000
        if abs(start + (samples.size - 1) * period - int(end_ticks) * 1000) > 500:
            raise ValueError(f"Pyrocko segment {index} end time differs from its recorded grid")

        found = {
            "channel": ".".join((net, sta, loc, component)), "start_ns": start,
            "npts": samples.size, "dtype": samples.dtype.name, "sample_sha256": sample_sha256(samples),
        }
        for key, value in found.items():
            if type(value) is not type(expected[key]) or value != expected[key]:
                raise ValueError(f"Pyrocko segment {index} {key} differs from the source manifest")
        segments.append(Segment(index, found["channel"], start, period, samples))
        verified.append({
            **found, "index": index, "segment_id": f"{source_hash}:{index}",
            "sampling_rate_hz": str(rate), "period_ns": str(period),
        })
    if sha256(path.read_bytes()).hexdigest() != source_hash:
        raise ValueError(f"Source SHA-256 changed during decoding of {path.name}")
    return {**spec, "segments": verified}, tuple(segments)


def decimal_value(value: Fraction) -> Decimal:
    return Decimal(value.numerator) / Decimal(value.denominator)


def finite_float(value: Fraction | Decimal) -> float:
    rounded = float(value)
    if not math.isfinite(rounded) or (value and rounded == 0):
        raise ArithmeticError("Reference result lies outside float64 range")
    return rounded


def sqrt_float(squared: Fraction) -> float:
    if squared == 0:
        return 0.0
    # Decimal locates the near binary64 value. Rational midpoints settle ties,
    # including the two boundaries where a nonzero result cannot be represented.
    lower_limit = Fraction(math.ulp(0.0)) / 2
    upper_limit = Fraction(float_info.max) + Fraction(math.ulp(float_info.max)) / 2
    if squared <= lower_limit**2 or squared >= upper_limit**2:
        raise ArithmeticError("Reference root lies outside float64 range")
    with localcontext() as context:
        context.prec = 96
        rounded = float(decimal_value(squared).sqrt())
    rounded = min(float_info.max, max(math.ulp(0.0), rounded))
    even = int(rounded / math.ulp(rounded)) % 2 == 0
    for direction in (0.0, math.inf):
        neighbor = math.nextafter(rounded, direction)
        if not math.isfinite(neighbor):
            continue
        midpoint = ((Fraction(rounded) + Fraction(neighbor)) / 2)**2
        if (squared < midpoint if direction == 0 else squared > midpoint) or (squared == midpoint and not even):
            return neighbor
    return rounded


def exact_statistics(samples: np.ndarray) -> tuple[float, float, Fraction]:
    """Use exact moments of the represented values, then round the Decimal square root."""
    total = Fraction(0)
    squares = Fraction(0)
    for sample in samples:
        represented = float(sample)
        if not math.isfinite(represented) or sample.item() != represented:
            raise ArithmeticError("Sample is outside float64 precision or range")
        exact = Fraction.from_float(represented)
        total += exact
        squares += exact * exact
    mean = total / len(samples)
    variance = squares / len(samples) - mean * mean
    return finite_float(mean), sqrt_float(variance), variance


def measure_window(segments: tuple[Segment, ...], interval: Interval | None, masks: tuple[Mask, ...]) -> Window:
    if interval is None:
        return Window(reasons=("missing_window",))

    start, stop = interval
    inside_start: Fraction = max(start, min(Fraction(segment.start) for segment in segments))
    inside_stop = min(stop, max(segment.stop for segment in segments))
    intersections = sorted(
        (max(start, Fraction(segment.start)), min(stop, segment.stop)) for segment in segments
        if start < segment.stop and segment.start < stop
    )
    covering = [segment for segment in segments if segment.start <= inside_start < inside_stop <= segment.stop]
    issues = []
    if not covering and inside_start < inside_stop:
        issues.append("cross_segment")
    if inside_start != start or inside_stop != stop:
        issues.append("out_of_bounds")
    if any(right > next_left for (_, right), (next_left, _) in zip(intersections, intersections[1:])):
        issues.append("ambiguous_coverage")
    if issues:
        return Window(reasons=tuple(issues))

    segment = covering[0]
    lo, hi = (math.ceil(segment.grid_index(bound)) for bound in interval)
    count = hi - lo
    if count == 0:
        return Window(reasons=("too_short",))
    support = segment.index, lo, hi
    if count < 2:
        issues.append("too_short")
    if count > MAX_WINDOW_SAMPLES:
        issues.append("too_large")
    if any(index == segment.index and mask_start < hi and lo < mask_stop for index, mask_start, mask_stop in masks):
        issues.append("masked_samples")
    samples = segment.samples[lo:hi]
    if count <= MAX_WINDOW_SAMPLES and not np.isfinite(samples).all():
        issues.append("nonfinite_samples")
    if issues:
        return Window(support, reasons=tuple(issues))
    try:
        mean, rms, variance = exact_statistics(samples)
    except ArithmeticError:
        return Window(support, reasons=("numeric_failure",))
    return Window(support, mean, rms, variance)


def measure(
    segments: tuple[Segment, ...], event: Interval | None, background: Interval | None, masks: tuple[Mask, ...],
) -> dict:
    e, b = (measure_window(segments, request, masks) for request in (event, background))
    reasons = set(e.reasons + b.reasons)
    count = None
    duration = None
    if e.support and b.support:
        event_segment, e_start, e_stop = e.support
        background_segment, b_start, b_stop = b.support
        if event_segment != background_segment:
            reasons.add("cross_segment")
        if e_stop - e_start != b_stop - b_start:
            reasons.add("unequal_length")
        elif event_segment == background_segment:
            segment = next(segment for segment in segments if segment.index == event_segment)
            count = e_stop - e_start
            duration = str(count * segment.period)
    if event and background and max(event[0], background[0]) < min(event[1], background[1]):
        reasons.add("overlapping_windows")
    if b.variance == 0:
        reasons.add("zero_background")

    contrast = None
    if not reasons:
        assert e.variance is not None and b.variance
        try:
            contrast = sqrt_float(e.variance / b.variance)
        except ArithmeticError:
            reasons.add("numeric_failure")
    return {
        **e.row("event"), **b.row("background"), "contrast": contrast,
        "samples_per_window": count, "duration_ns": duration,
        "available": contrast is not None, "reasons": [reason for reason in REASONS if reason in reasons],
        "units": "raw units unknown", "epoch_status": "unknown",
    }


def compare_value(observed: Any, expected: Any, relative_only: bool, *, numeric: bool = True) -> dict:
    bound = None
    delta = None
    passed = type(observed) is type(expected) and observed == expected
    if numeric and type(observed) in (int, float) and type(expected) in (int, float):
        scale = abs(expected) if relative_only else max(1, abs(expected))
        bound = 1e-12 * scale if expected else 0.0
        delta = abs(observed - expected)
        passed = delta <= bound and (expected == 0 or not relative_only or observed != 0)
        if not math.isfinite(delta):
            delta = "outside-float64"
    return {"expected": expected, "observed": observed, "delta": delta, "bound": bound, "passed": passed}


def marker_time(date: str, time: str) -> tuple[int, Fraction]:
    match = re.fullmatch(r"([0-9]{2}:[0-9]{2}:[0-9]{2})\.([0-9]{1,9})", time)
    if match is None:
        raise ValueError("Marker UTC time needs explicit fractional precision of 1–9 decimal digits")
    whole, fractional = match.groups()
    utc = calendar.timegm(datetime.strptime(f"{date} {whole}", "%Y-%m-%d %H:%M:%S").timetuple())
    quantum_ns = 10 ** (9 - len(fractional))
    timestamp = utc * NS_PER_SECOND + int(fractional) * quantum_ns
    # Serialization rounds both the displayed decimal and its binary64 epoch input.
    radius = Fraction(quantum_ns, 2) + Fraction(math.ulp(float(Fraction(timestamp, NS_PER_SECOND)))) * NS_PER_SECOND / 2
    return timestamp, radius


def marker_interval(
    row: list[str], segments: tuple[Segment, ...], channel: str,
) -> tuple[str, Interval, dict]:
    if len(row) != 15 or row[0] != "phase:":
        raise ValueError("Unsupported Snuffler marker fields; select exactly two named phase intervals")
    _, start_date, start_time, end_date, end_time, length, kind, trace, event_id, event_date, event_time, name, polarity, automatic, uncertainty = row
    if name not in {"E", "B"}:
        raise ValueError("Snuffler interval name must be E or B")
    if trace != channel or not re.fullmatch(r"[0-9]+", kind):
        raise ValueError("Snuffler marker must target exactly the selected channel with an integer kind")
    if (event_id, event_date, event_time, polarity, uncertainty) != ("None",) * 5 or automatic not in {"False", "None"}:
        raise ValueError("Unsupported Snuffler event, polarity, automatic or uncertainty fields")
    if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?", length) is None:
        raise ValueError("Invalid Snuffler interval duration")
    printed = (marker_time(start_date, start_time), marker_time(end_date, end_time))
    candidates: list[Interval] = []
    for segment in segments:
        bounds = []
        for timestamp, radius in printed:
            if not segment.start - radius <= timestamp <= segment.stop + radius:
                break
            if 2 * radius >= segment.period:
                raise ValueError(f"Snuffler marker {name} precision cannot identify one boundary")
            first = math.ceil(segment.grid_index(timestamp - radius))
            last = math.floor(segment.grid_index(timestamp + radius))
            if first != last or not 0 <= first <= len(segment.samples):
                break
            bounds.append(segment.start + first * segment.period)
        if len(bounds) == 2 and bounds[0] < bounds[1]:
            candidates.append((bounds[0], bounds[1]))
    if len(candidates) != 1:
        raise ValueError(f"Snuffler marker {name} is off-grid, outside one segment, or has ambiguous boundaries")
    interval = candidates.pop()
    # The native duration is derived from two binary64 endpoints before their display rounding.
    clock_error = sum(Fraction(math.ulp(float(bound / NS_PER_SECOND))) for bound in interval)
    duration = Fraction(length)
    length_error = Fraction(math.ulp(float(duration)))
    if abs((interval[1] - interval[0]) / NS_PER_SECOND - duration) > (clock_error + length_error) / 2:
        raise ValueError(f"Snuffler marker {name} duration does not support the restored grid boundaries")
    return name, interval, {
        "printed_utc": [f"{start_date} {start_time}", f"{end_date} {end_time}"],
        "saved_duration_s": length,
        "endpoint_adjustment_ns": [str(bound - saved[0]) for bound, saved in zip(interval, printed, strict=True)],
    }


def read_markers(path: Path, segments: tuple[Segment, ...], channel: str) -> tuple[Interval, Interval, dict]:
    contents = path.read_bytes()
    lines = contents.decode("utf-8").splitlines()
    if not lines or lines[0] != "# Snuffler Markers File Version 0.3":
        raise ValueError("Expected a Snuffler marker file version 0.3")
    intervals = {}
    rows = {}
    for line in lines[1:]:
        if not line.strip():
            continue
        name, interval, row = marker_interval(shlex.split(line), segments, channel)
        if name in intervals:
            raise ValueError(f"Duplicate Snuffler interval name {name}")
        intervals[name] = interval
        rows[name] = row
    if set(intervals) != {"E", "B"}:
        raise ValueError("Expected exactly one E and one B Snuffler interval")
    return intervals["E"], intervals["B"], {"format": MARKER_FORMAT, "sha256": sha256(contents).hexdigest(), "rows": rows}


def parse_interval(value: Any) -> Interval | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 2 or any(type(bound) not in (int, str) for bound in value):
        raise ValueError("Expected two exact UTC nanosecond bounds in each interval")
    start, stop = map(Fraction, value)
    if start >= stop:
        raise ValueError("Expected ordered UTC nanosecond bounds")
    return start, stop


def evaluate_manifest(
    path: Path, *, case_id: str | None = None, marker_path: Path | None = None, confirm_grid_intent: bool = False,
) -> dict:
    if marker_path and (not case_id or not confirm_grid_intent):
        raise ValueError("Snuffler intervals require --case and --confirm-grid-intent")
    manifest_bytes = path.read_bytes()
    manifest = json.loads(manifest_bytes, object_pairs_hook=unique_fields, parse_constant=reject_nonfinite_json)
    require_fields(manifest, "version sources cases", "Input manifest")
    if type(manifest["version"]) is not int or manifest["version"] != 1:
        raise ValueError("Expected input manifest version 1")
    cases = []
    seen: set[str] = set()
    sources: dict[str, tuple[dict, tuple[Segment, ...]]] = {}
    for case in manifest["cases"]:
        require_fields(case, "id source channel event_ns background_ns masks relative_only", "Case", optional="expected")
        name = case["id"]
        if not isinstance(name, str) or not name or name in seen:
            raise ValueError("Case IDs must be unique nonempty text")
        seen.add(name)
        if case_id and name != case_id:
            continue
        if type(case["relative_only"]) is not bool:
            raise ValueError(f"{name}: relative_only must be boolean")
        source_name = case["source"]
        if source_name not in sources:
            sources[source_name] = read_source(path.parent, manifest["sources"][source_name])
        source, all_segments = sources[source_name]
        channel = case["channel"]
        segments = tuple(segment for segment in all_segments if segment.channel == channel)
        if not segments:
            raise ValueError(f"{name}: selected channel is absent from the verified source")
        masks = tuple(tuple(mask) for mask in case["masks"])
        by_index = {segment.index: segment for segment in segments}
        for mask in masks:
            if len(mask) != 3 or any(type(value) is not int for value in mask):
                raise ValueError(f"{name}: a mask needs a segment index and two integer sample indices")
            index, start, stop = mask
            if index not in by_index or not 0 <= start < stop <= len(by_index[index].samples):
                raise ValueError(f"{name}: mask falls outside the selected channel's sample support")
        event = parse_interval(case["event_ns"])
        background = parse_interval(case["background_ns"])
        marker_record = None
        if marker_path:
            restored_event, restored_background, marker_record = read_markers(marker_path, segments, channel)
            if "expected" in case and (event, background) != (restored_event, restored_background):
                raise ValueError(f"{name}: marker indices differ from the predeclared fixture support")
            event, background = restored_event, restored_background
        measured = measure(segments, event, background, masks)
        comparison = {}
        if "expected" in case:
            expected = case["expected"]
            require_fields(expected, " ".join(measured), f"{name} expected outcome")
            comparison = {
                field: compare_value(value, expected[field], case["relative_only"], numeric=field in NUMERIC_FIELDS)
                for field, value in measured.items()
            }
        status = "unassessed"
        if comparison:
            status = "passed" if all(field["passed"] for field in comparison.values()) else "failed"
        cases.append({
            "comparison_status": status, "id": name, "source": source, "channel": channel, "masks": masks,
            "relative_only": case["relative_only"], "marker_file": marker_record,
            "requests_ns": {
                field: [str(bound) for bound in interval] if interval else None
                for field, interval in (("event", event), ("background", background))
            },
            "reference": measured, "comparison": comparison,
        })
    if not cases:
        raise ValueError("No matching comparison cases")
    statuses = {case["comparison_status"] for case in cases}
    if "failed" in statuses:
        status = "failed"
    elif len(statuses) == 1:
        status = statuses.pop()
    else:
        status = "partial"
    return {
        "schema_version": 1, "method": "demeaned-rms-v1", "input_sha256": sha256(manifest_bytes).hexdigest(),
        "dependencies": {"python": platform.python_version(), "numpy": np.__version__, "pyrocko": version("pyrocko")},
        "comparison_status": status,
        "passed": status == "passed" if status in {"failed", "passed"} else None,
        "cases": cases,
    }


def write_results(output_dir: Path, bundle: dict) -> None:
    # Require a new destination so a comparison cannot overwrite source evidence.
    output_dir.mkdir()
    (output_dir / "sample-manifest.json").write_text(
        json.dumps(bundle, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    with (output_dir / "reference.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "case", "source_sha256", "channel", "event_request_ns", "background_request_ns", "masks",
            *bundle["cases"][0]["reference"],
        ])
        writer.writeheader()
        for case in bundle["cases"]:
            row = {
                "case": case["id"], "source_sha256": case["source"]["sha256"], "channel": case["channel"],
                "event_request_ns": case["requests_ns"]["event"], "background_request_ns": case["requests_ns"]["background"],
                "masks": case["masks"], **case["reference"],
            }
            writer.writerow({key: json.dumps(value, separators=(",", ":")) if isinstance(value, (tuple, list)) else value
                             for key, value in row.items()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path, help="Create this result directory")
    parser.add_argument("--case", help="Run one manifest case")
    parser.add_argument("--markers", type=Path, help="Snuffler 0.3 file with named E and B intervals")
    parser.add_argument("--confirm-grid-intent", action="store_true", help="Declare that saved intervals were intended to be exact grid boundaries")
    args = parser.parse_args()
    try:
        bundle = evaluate_manifest(
            args.manifest, case_id=args.case, marker_path=args.markers, confirm_grid_intent=args.confirm_grid_intent,
        )
        write_results(args.output_dir, bundle)
    except (ValueError, TypeError, KeyError, ArithmeticError, OSError, mseed_ext.MSeedError) as error:
        parser.error(str(error))
    if bundle["passed"] is False:
        raise SystemExit("The independent reference disagrees with the declared fixture outcomes")


if __name__ == "__main__":
    main()
