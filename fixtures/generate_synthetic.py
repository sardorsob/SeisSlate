"""Write a reproducible paired MiniSEED fixture and the notebook fit baseline."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from obspy import Stream, Trace, UTCDateTime


ORIGIN = UTCDateTime("2000-01-01T00:00:00Z")
RATE_HZ = 1000
TRANSECT_M = 44.2
LAG_BOUNDS_S = (-0.06, 0.08)
# Each index here is elapsed milliseconds from ORIGIN at 1000 Hz.
# Blow, CALB onset, ORIG truth, ORIG reported pick, x1 (m), case
BLOWS = (
    ("B01", 400, 360, 360, 13.1, "clear"),
    ("B02", 800, 815, 815, 26.85, "close first"),
    ("B03", 880, 910, 910, 30.6, "close second"),
    ("B04", 1250, 1225, 1345, 16.85, "mistimed pick"),
    ("B05", 1650, 1590, 1590, 8.1, "strict lower bound"),
    ("B06", 1950, 2030, 2030, 43.1, "strict upper bound"),
    ("B07", 2145, 2165, None, 28.1, "original onset in gap"),
    ("B08", 2500, 2500, 2500, 23.1, "zero lag"),
)
COVERAGE = {"original": ((0, 2150), (2200, 3000)), "calibration": ((0, 3000),)}
PULSE = (750, 1050, 430, -600, -510, 190, 140, -80, -40)


def utc(index):
    return str(ORIGIN + index / RATE_HZ)


def write_source(output, role, station, arrivals):
    samples = ((np.arange(3000, dtype=np.int32) * 13 + 7) % 23) - 11
    for index in arrivals:
        samples[index : index + len(PULSE)] += PULSE

    traces = []
    segments = []
    for start, stop in COVERAGE[role]:
        header = {
            "network": "XX",
            "station": station,
            "location": "",
            "channel": "EHZ",
            "sampling_rate": RATE_HZ,
            "starttime": ORIGIN + start / RATE_HZ,
        }
        traces.append(Trace(samples[start:stop].copy(), header))
        segments.append({"start_utc": utc(start), "npts": stop - start})

    path = output / f"{role}.mseed"
    Stream(traces).write(str(path), format="MSEED", encoding="INT32", reclen=512, byteorder=">")
    data = path.read_bytes()
    return {
        "path": path.name,
        "size_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "trace_id": traces[0].id,
        "sampling_rate_hz": RATE_HZ,
        "segments": segments,
    }


def baseline_rows():
    rows = []
    for blow_id, calibration, truth, original, x1, case in BLOWS:
        dt = None if original is None else (original - calibration) / RATE_HZ
        segment_start = 0 if original is None or original < 2150 else 2200
        x2 = TRANSECT_M - x1
        rows.append({
            "blow_id": blow_id,
            "case": case,
            "original_truth_utc": utc(truth),
            "original_pick_utc": None if original is None else utc(original),
            "original_segment_start_utc": None if original is None else utc(segment_start),
            "original_sample_index": None if original is None else original - segment_start,
            "calibration_pick_utc": utc(calibration),
            "calibration_sample_index": calibration,
            "distance_original_m": x1,
            "distance_calibration_m": round(x2, 10),
            "dt_s": dt,
            "dx_m": round(x1 - x2, 10),
        })
    return rows


def write_baseline(output):
    rows = baseline_rows()
    paired = [row for row in rows if row["dt_s"] is not None]
    dt = np.array([row["dt_s"] for row in paired])
    dx = np.array([row["dx_m"] for row in paired])

    # Notebook cell 11, retaining the intercept in both fits.
    slope_all, intercept_all = np.polyfit(dt, dx, 1)
    lower, upper = LAG_BOUNDS_S
    mask = (dt > lower) & (dt < upper)
    slope_strict, intercept_strict = np.polyfit(dt[mask], dx[mask], 1)

    # Notebook cell 4 defines residual as observed distance minus predicted distance.
    residual_all = dx - (slope_all * dt + intercept_all)
    residual_strict = dx - (slope_strict * dt + intercept_strict)
    for row in rows:
        row.update(strict_included="false", all_residual_m=None, strict_residual_m=None)
    for row, included, all_m, strict_m in zip(paired, mask, residual_all, residual_strict, strict=True):
        row.update(
            strict_included=str(bool(included)).lower(),
            all_residual_m=round(float(all_m), 10) or 0.0,
            strict_residual_m=round(float(strict_m), 10) or 0.0,
        )

    with (output / "baseline.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    return {
        "all": {
            "slope_m_per_s": round(float(slope_all), 10),
            "intercept_m": round(float(intercept_all), 10),
            "blow_ids": [row["blow_id"] for row in paired],
        },
        "strict": {
            "slope_m_per_s": round(float(slope_strict), 10),
            "intercept_m": round(float(intercept_strict), 10),
            "blow_ids": [row["blow_id"] for row in paired if row["strict_included"] == "true"],
        },
    }


def generate(output):
    output.mkdir(parents=True)
    original = write_source(output, "original", "ORIG", [blow[2] for blow in BLOWS])
    calibration = write_source(output, "calibration", "CALB", [blow[1] for blow in BLOWS])
    expected = {
        "synthetic": True,
        "origin_utc": utc(0),
        "transect_length_m": TRANSECT_M,
        "sources": {"original": original, "calibration": calibration},
        "notebook": {
            "path": "fluvial_seismology/notebooks/instrument_calibration/estimate_n11_hammer_p_wave_velocity.ipynb",
            "last_change_commit": "b1e156a6117c609ba1cdb72f981bcfeafd793106",
            "sha256": "ac7cf3ae7d61337831066252a055521f21abca7f7c9c5088e57cef49bacfc139",
            "fit_cell_zero_based": 11,
            "residual_cell_zero_based": 4,
        },
        "strict_lag_bounds_exclusive_s": list(LAG_BOUNDS_S),
        "fits": write_baseline(output),
        "snuffler_probe": {
            "blow_id": "B01",
            "original_segment_start_utc": utc(0),
            "original_sample_index": 360,
            "original_pick_utc": utc(360),
            "nudged_sample_index": 361,
            "nudged_pick_utc": utc(361),
            "calibration_pick_utc": utc(400),
        },
    }
    with (output / "expected.json").open("w", newline="\n", encoding="utf-8") as file:
        json.dump(expected, file, indent=2)
        file.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new output directory")
    generate(parser.parse_args().output)
