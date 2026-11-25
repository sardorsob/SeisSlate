"""Measure fixed synthetic SeisSlate workloads in clean and warmed processes."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from hashlib import file_digest
from importlib import import_module
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory
from time import perf_counter_ns
from typing import Callable

pilot_check = import_module("tools.check_pilot" if __package__ else "check_pilot")


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
BASELINE = "0f76033"
WARM_REPEATS = 20
TIMEOUT_SECONDS = 240
RESULT_PREFIX = "SEISSLATE_BENCHMARK "
HISTORY_STEPS = 1000
CAPTURE_EVERY = 20
LARGE = {"single_1h_100hz": (360_000, 100), "single_24h_200hz": (17_280_000, 200)}
TARGETS = {
    ("hammer", "launch_to_interactive", "cold"): 5000,
    ("hammer", "apply_refresh", "warm_p95"): 250,
    ("rms_cap", "compute", "warm_p95"): 250,
    ("history_1000", "save", "warm_p95"): 3000,
    ("history_1000", "reopen", "warm_p95"): 3000,
}
RUNS = (
    ("hammer", "apply_refresh"),
    ("hammer", "load"),
    ("hammer", "raw_capture"),
    ("single_1h_100hz", "load"),
    ("single_24h_200hz", "load"),
    ("rms_cap", "compute"),
    ("history_1000", "save"),
    ("history_1000", "reopen"),
)
BASELINE_RUNS = {("hammer", "apply_refresh"), ("hammer", "load")}


def milliseconds(start: int) -> float:
    return (perf_counter_ns() - start) / 1_000_000


def peak_rss() -> int:
    if sys.platform != "win32":
        import resource

        amount = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(amount if sys.platform == "darwin" else amount * 1024)

    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = (("size", wintypes.DWORD), ("page_fault_count", wintypes.DWORD)) + tuple(
            (name, ctypes.c_size_t) for name in (
                "peak_working_set", "working_set", "quota_peak_paged_pool", "quota_paged_pool",
                "quota_peak_nonpaged_pool", "quota_nonpaged_pool", "pagefile", "peak_pagefile",
            )
        )

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    read = ctypes.WinDLL("psapi", use_last_error=True).GetProcessMemoryInfo
    read.argtypes = (wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD)
    read.restype = wintypes.BOOL
    counters = Counters()
    counters.size = ctypes.sizeof(counters)
    if not read(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.size):
        raise ctypes.WinError(ctypes.get_last_error())
    return int(counters.peak_working_set)


def sha(path: Path) -> str:
    with path.open("rb") as handle:
        return file_digest(handle, "sha256").hexdigest()


def check(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def paired_desktop(fixture: Path, *, start_ns: int | None, count: int) -> tuple[list[dict], dict]:
    from PyQt5.QtCore import QEvent, QObject, QT_VERSION_STR, Qt
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication
    from seisslate.example import load_example
    from seisslate.ui import launch_workbench

    initial = load_example(fixture)
    rows: list[dict] = []
    metadata: dict = {}

    class Paints(QObject):
        def __init__(self):
            super().__init__()
            self.count = 0

        def eventFilter(self, watched, event):
            if event.type() == QEvent.Paint:
                self.count += 1
            return False

    def exercise(workbench):
        app = QApplication.instance()
        metadata.update(qt_backend=app.platformName(), qt_version=QT_VERSION_STR,
                        matplotlib_canvas=type(workbench.canvas).__name__)
        workbench.window.resize(1200, 840)
        app.processEvents()
        check(workbench.window.isVisible() and workbench.viewer.isVisible(), "Desktop waveform is not visible")
        check(workbench.table.item(0, 0).text() == "B01", "Desktop review is not interactive")
        if start_ns is not None:
            rows.append({"duration_ms": milliseconds(start_ns), "peak_rss_bytes": peak_rss(), "error": None})
            return

        waveform, chart = Paints(), Paints()
        workbench.viewer.installEventFilter(waveform)
        workbench.canvas.installEventFilter(chart)
        for _ in range(count):
            def apply():
                before = waveform.count, chart.count
                QTest.mouseClick(workbench.buttons["apply"], Qt.LeftButton)
                app.processEvents()
                return before

            def consistent(before):
                point = workbench.history.current.state.blow("B01").original.reviewed_ref
                row = next(obs for obs in workbench.result.reviewed.observations if obs.blow_id == "B01")
                check(point.sample_index == 361 and row.lag_ns == -39_000_000, "Unexpected applied pick or fit observation")
                check(workbench.table.item(0, 2).text() == "-39.000", "Applied lag table is stale")
                check(waveform.count > before[0] and chart.count > before[1], "Waveform or result plot did not repaint")

            QTest.mouseClick(workbench.buttons["later"], Qt.LeftButton)
            app.processEvents()
            check(workbench.history.draft is not None, "Cannot stage the benchmark pick")
            rows.append(observe(apply, consistent))
            if workbench.history.can_undo:
                QTest.mouseClick(workbench.buttons["undo"], Qt.LeftButton)
                app.processEvents()
            elif workbench.history.draft is not None:
                workbench.cancel()
        workbench.viewer.removeEventFilter(waveform)
        workbench.canvas.removeEventFilter(chart)

    launch_workbench(initial, fixture=fixture, check=exercise)
    return rows, metadata


def observe(action: Callable, verify: Callable | None = None) -> dict:
    start = perf_counter_ns()
    try:
        result = action()
        elapsed = milliseconds(start)
        if verify is not None:
            verify(result)
    except Exception as error:
        return {"duration_ms": None, "peak_rss_bytes": peak_rss(), "error": f"{type(error).__name__}: {error}"}
    return {"duration_ms": elapsed, "peak_rss_bytes": peak_rss(), "error": None}


def prepare_source(path: Path, samples: int, rate: int) -> dict:
    import numpy as np
    from obspy import Stream, Trace, UTCDateTime

    start = perf_counter_ns()
    pattern = np.array([-2, 2, -1, 1, 0, 12, 4, 2], dtype=np.int32)
    data = np.tile(pattern, samples // len(pattern))
    stream = Stream([Trace(data, header={
        "network": "XX", "station": "BENCH", "channel": "EHZ",
        "starttime": UTCDateTime(2000, 1, 1), "sampling_rate": rate,
    })])
    stream.write(str(path), format="MSEED", encoding="STEIM2", reclen=4096)
    return {"prepare_ms": milliseconds(start), "samples": samples, "sample_rate_hz": rate,
            "size_bytes": path.stat().st_size, "sha256": sha(path)}


def open_signal(path: Path, samples: int, rate: int):
    from seisslate.sources import load_source

    source = load_source(path)
    (segment,) = source.segments
    check(segment.npts == samples and segment.sampling_rate_hz == rate, "Synthetic source grid or count changed")
    return source


def prepare_capture(catalog, history, blow_id: str, *, reason: str):
    from seisslate.evidence import PickDecision, RawEvidenceView
    from seisslate.review import set_pick_decision
    from seisslate.sources import SampleSpan

    point = history.current.state.blow(blow_id).original.reviewed_ref
    segment = catalog.segment(point)
    first = max(0, point.sample_index - 50)
    stop = min(segment.npts, point.sample_index + 51)
    span = SampleSpan(segment.segment_id, first, stop)
    decision = PickDecision(
        blow_id, "original", point, "accepted", point, point, reason=reason,
        reviewer="Benchmark", inspection_span=span,
    )
    view = RawEvidenceView(
        point.source_hash, point.channel, span, segment.time_ns(first),
        segment.time_ns(stop) if stop < segment.npts else segment.stop_ns, point, (point, point),
    )
    history.preview(set_pick_decision(history.current.state, catalog, decision), reason)
    return decision, view


def capture(catalog, history, decision, view, revision: int):
    confirmation = history.confirm_current_pick_evidence(catalog, decision, view)
    captured = int(datetime(2000, 1, 2, tzinfo=timezone.utc).timestamp()) * 10**9 + revision
    return history.apply_with_evidence(
        catalog, confirmation, view, captured_utc_ns=captured,
        software_versions=(("python", platform.python_version()),),
    )


def prepare_history(fixture: Path, path: Path) -> dict:
    from seisslate.example import load_example
    from seisslate.review import nudge_pick
    from seisslate.session import save_session

    catalog, history = load_example(fixture)
    sweep = tuple(blow.blow_id for blow in history.current.state.blows
                  if blow.original.reviewed_ref is not None and blow.calibration.reviewed_ref is not None)
    check(bool(sweep), "Hammer fixture has no reviewed pairs")
    def point_index(blow_id: str) -> int:
        ref = history.current.state.blow(blow_id).original.reviewed_ref
        if ref is None:
            raise RuntimeError("Benchmark onset has no source sample")
        return ref.sample_index

    original = {blow_id: point_index(blow_id) for blow_id in sweep}
    sizes = {}
    growth = {}
    start = perf_counter_ns()
    for revision in range(1, HISTORY_STEPS + 1):
        if revision % CAPTURE_EVERY == 0:
            blow_id = sweep[(revision // CAPTURE_EVERY - 1) % len(sweep)]
            decision, view = prepare_capture(catalog, history, blow_id, reason=f"Confirm observed onset, review {revision}")
            capture(catalog, history, decision, view, revision)
        else:
            blow_id = sweep[(revision - 1) % len(sweep)]
            cursor = point_index(blow_id)
            step = 1 if cursor == original[blow_id] else -1
            history.preview(nudge_pick(history.current.state, catalog, blow_id, "original", step),
                            f"{blow_id} later onset sample" if step > 0 else f"{blow_id} restore earlier onset sample")
            history.apply()
        if revision in {100, 500, HISTORY_STEPS}:
            growth[str(revision)] = len(history.current.state.raw_evidence)
            save_session(path, catalog, history)
            sizes[str(revision)] = path.stat().st_size
    check(history.current.revision_id == HISTORY_STEPS, "History did not apply all review revisions")
    return {
        "prepare_ms_including_growth_saves": milliseconds(start), "applied_revisions": HISTORY_STEPS,
        "total_snapshots": len(history.revisions), "evidence_records_at_revision": growth,
        "bytes_at_revision": sizes, "source_blow_ids": list(sweep), "sha256": sha(path),
    }


def repeat_workload(root: Path, temporary: Path, workload: str, metric: str, count: int) -> tuple[list[dict], dict]:
    fixture = root / "fixtures" / "synthetic"
    if metric == "apply_refresh":
        return paired_desktop(fixture, start_ns=None, count=count)
    if workload == "hammer":
        from seisslate.example import load_example

        if metric == "load":
            return [observe(lambda: load_example(fixture)) for _ in range(count)], {}
        rows = []
        for _ in range(count):
            catalog, history = load_example(fixture)
            decision, view = prepare_capture(catalog, history, "B01", reason="Confirm observed onset")
            rows.append(observe(
                lambda: capture(catalog, history, decision, view, 1),
                lambda result: check(len(result.state.raw_evidence) == 1, "Raw evidence did not persist at Apply"),
            ))
        return rows, {}
    if workload in LARGE:
        samples, rate = LARGE[workload]
        return [observe(lambda: open_signal(temporary / f"{workload}.mseed", samples, rate))
                for _ in range(count)], {}
    if workload == "rms_cap":
        from seisslate.review import ReviewState
        from seisslate.rms import RmsWindow, RmsWindows, compute_rms_contrast
        from seisslate.sources import SourceCatalog

        samples, rate = LARGE["single_24h_200hz"]
        source = open_signal(temporary / "single_24h_200hz.mseed", samples, rate)
        catalog = SourceCatalog((source,))
        (segment,) = catalog.segments
        identity = segment.source_hash, segment.channel
        cuts = [segment.start_ns + index * segment.period_ns for index in (0, 1_000_000, 2_000_000)]
        windows = RmsWindows(*identity, RmsWindow(*identity, *cuts[:2]), RmsWindow(*identity, *cuts[1:]))
        state = ReviewState.for_single_channel(*identity)

        def available(result):
            check(result.available and result.samples_per_window == 1_000_000 and result.contrast == 1,
                  "The full cap computation did not yield the expected result")

        return [observe(lambda: compute_rms_contrast(catalog, state, windows), available)
                for _ in range(count)], {"samples_per_window": 1_000_000}

    from seisslate.session import load_session, save_session

    target = temporary / "history.json"

    def loaded(result):
        _, history = result
        check(history.current.revision_id == HISTORY_STEPS
              and len(history.current.state.raw_evidence) == HISTORY_STEPS // CAPTURE_EVERY,
              "History replay omitted revisions or evidence")

    if metric == "reopen":
        return [observe(lambda: load_session(target), loaded) for _ in range(count)], {}
    catalog, history = load_session(target)
    loaded((catalog, history))
    out = temporary / "timed-save.json"
    size = target.stat().st_size

    def matching_size(_):
        check(out.stat().st_size == size, "A timed save changed the prepared session byte count")

    rows = [observe(lambda: save_session(out, catalog, history), matching_size) for _ in range(count)]
    if out.exists() and sha(out) != sha(target):
        rows[-1].update(duration_ms=None, error="Saved session bytes differ from the prepared session")
    return rows, {"session_bytes": out.stat().st_size if out.exists() else None}


def worker(args: argparse.Namespace) -> None:
    root = Path(args.checkout)
    temporary = Path(args.workspace)
    sys.path.insert(0, str(root))
    if args.metric == "prepare":
        if args.workload == "history_1000":
            data = prepare_history(root / "fixtures" / "synthetic", temporary / "history.json")
        else:
            data = prepare_source(temporary / f"{args.workload}.mseed", *LARGE[args.workload])
        body = {"preparation": data, "rows": [], "metadata": {}}
    elif args.metric == "launch_to_interactive":
        rows, metadata = paired_desktop(root / "fixtures" / "synthetic", start_ns=args.started, count=1)
        body = {"rows": rows, "metadata": metadata}
    else:
        rows, metadata = repeat_workload(root, temporary, args.workload, args.metric, args.count)
        body = {"rows": rows, "metadata": metadata}
    print(RESULT_PREFIX + json.dumps(body, allow_nan=False), flush=True)


def isolated(
    checkout: Path, workspace: Path, config: Path, workload: str, metric: str, count: int,
) -> dict:
    env = {**os.environ, "PYROCKO_DIR": str(config / "pyrocko"), "MPLCONFIGDIR": str(config / "matplotlib")}
    env.pop("PYTHONOPTIMIZE", None)
    args = [
        sys.executable, "-B", "-u", "-X", "utf8", str(SCRIPT), "--worker", "--checkout", str(checkout),
        "--workspace", str(workspace), "--workload", workload, "--metric", metric, "--count", str(count),
    ]
    if metric == "launch_to_interactive":
        args.extend(("--started", str(perf_counter_ns())))
    try:
        result = subprocess.run(
            args, cwd=checkout, env=env, stdin=subprocess.DEVNULL, capture_output=True,
            encoding="utf-8", errors="replace", timeout=TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return {"error": f"{type(error).__name__}: {error}", "rows": [], "metadata": {}}
    lines = [line for line in result.stdout.splitlines() if line.startswith(RESULT_PREFIX)]
    if result.returncode or len(lines) != 1:
        diagnostics = (result.stderr + "\n" + result.stdout).strip()[-2500:]
        return {"error": f"Worker exit {result.returncode}: {diagnostics}", "rows": [], "metadata": {}}
    return json.loads(lines[0][len(RESULT_PREFIX):])


def collect(checkout: Path, workspace: Path, repeats: int, *, baseline: bool) -> tuple[list[dict], dict]:
    label = "baseline" if baseline else "candidate"
    config = workspace / f"{label}-desktop-config"
    rows: list[dict] = []
    metadata: dict[str, dict] = {}
    workloads = RUNS if not baseline else tuple(run for run in RUNS if run in BASELINE_RUNS)

    def record(workload: str, metric: str, result: dict, iterations: range) -> None:
        metadata[f"{workload}.{metric}"] = result["metadata"] or metadata.get(f"{workload}.{metric}", {})
        result_rows = result["rows"]
        for offset, index in enumerate(iterations):
            observation = result_rows[offset] if offset < len(result_rows) else {
                "error": result.get("error", "Missing worker observation"), "duration_ms": None, "peak_rss_bytes": None,
            }
            rows.append({
                "build": label, "workload": workload, "metric": metric, "phase": "cold" if index == 0 else "warm",
                "replicate": index, **observation,
            })

    print(f"{label}: hammer launch in {repeats + 1} isolated processes", flush=True)
    for index in range(repeats + 1):
        result = isolated(checkout, workspace, config, "hammer", "launch_to_interactive", 1)
        record("hammer", "launch_to_interactive", result, range(index, index + 1))
    for workload, metric in workloads:
        print(f"{label}: {workload} {metric}", flush=True)
        result = isolated(checkout, workspace, config, workload, metric, repeats + 1)
        record(workload, metric, result, range(repeats + 1))
    return rows, metadata


def nearest_rank(values, probability):
    return values[math.ceil(probability * len(values)) - 1] if values else None


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        grouped.setdefault((row.get("build", "candidate"), row["workload"], row["metric"]), []).append(row)
    result = []
    for (build, workload, metric), observations in grouped.items():
        cold = next(row for row in observations if row["phase"] == "cold")
        attempts = [row for row in observations if row["phase"] == "warm"]
        successful = [row for row in attempts if row["error"] is None]
        warm = sorted(row["duration_ms"] for row in successful)
        warm_memory = sorted(row["peak_rss_bytes"] for row in successful if row.get("peak_rss_bytes") is not None)
        entry = {
            "build": build, "workload": workload, "metric": metric, "cold_ms": cold["duration_ms"],
            "cold_error": cold["error"], "warm_attempts": len(attempts), "warm_successes": len(warm),
            "warm_failures": len(attempts) - len(warm),
            "warm_median_ms": nearest_rank(warm, 0.5), "warm_p95_ms": nearest_rank(warm, 0.95),
            "cold_peak_rss_bytes": cold.get("peak_rss_bytes"),
            "warm_median_peak_rss_bytes": nearest_rank(warm_memory, 0.5),
            "warm_p95_peak_rss_bytes": nearest_rank(warm_memory, 0.95),
        }
        if build == "candidate":
            for field in ("cold", "warm_p95"):
                target = TARGETS.get((workload, metric, field))
                if target is not None:
                    observed = entry[field + "_ms"]
                    entry["target"] = {
                        "statistic": field, "limit_ms": target,
                        "met": observed is not None and observed <= target,
                    }
        result.append(entry)
    return result


def compare_baseline(summary: list[dict]) -> list[dict]:
    candidates = {(entry["workload"], entry["metric"]): entry for entry in summary if entry["build"] == "candidate"}
    comparison = []
    for entry in summary:
        if entry["build"] != "baseline":
            continue
        candidate = candidates[(entry["workload"], entry["metric"])]
        for statistic in ("cold", "warm_p95"):
            before, after = entry[statistic + "_ms"], candidate[statistic + "_ms"]
            change = (after / before - 1) * 100 if before and after is not None else None
            comparison.append({
                "workload": entry["workload"], "metric": entry["metric"], "statistic": statistic,
                "baseline_ms": before, "candidate_ms": after, "change_percent": change,
                "investigate": change is not None and change > 20,
            })
    return comparison


def checkout_baseline(root: Path, baseline: Path, storage: Path, baseline_ref: str) -> str:
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "--verify", "--end-of-options", f"{baseline_ref}^{{commit}}"],
                                       text=True, timeout=15).strip()
    with storage.open("wb") as output:
        subprocess.run(["git", "-C", str(root), "archive", revision], stdout=output, check=True, timeout=30)
    baseline.mkdir()
    with tarfile.open(storage) as archive:
        archive.extractall(baseline, filter="data")
    return revision


def environment() -> dict:
    interpreter, host = pilot_check.host_architectures()
    return {
        "os": platform.platform(), "host_architecture": host, "interpreter_architecture": interpreter,
        "cpu": platform.processor(), "logical_cpu_count": os.cpu_count(), "python": platform.python_version(),
        "dependencies": {name: version(name) for name in pilot_check.DEPENDENCIES},
        "requested_qt_environment": {name: os.environ.get(name) for name in ("QT_QPA_PLATFORM", "QT_SCALE_FACTOR")},
    }


def source_identity(root: Path) -> dict:
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True, timeout=15).strip()
    dirty = bool(subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], timeout=15))
    runtime = [*sorted((root / "seisslate").glob("*.py")), root / "pyproject.toml", SCRIPT,
               Path(__file__).with_name("check_pilot.py")]
    return {"commit": commit, "dirty": dirty, "runtime_sha256": {str(path.relative_to(root)): sha(path) for path in runtime}}


def run(output: Path, repeats: int, baseline_ref: str) -> int:
    output.mkdir()
    identity = source_identity(ROOT)
    with TemporaryDirectory(prefix="seisslate-benchmark-") as directory:
        scratch = Path(directory)
        preparation = {}
        for workload in (*LARGE, "history_1000"):
            print(f"prepare: {workload}", flush=True)
            prepared = isolated(ROOT, scratch, scratch / "preparation", workload, "prepare", 1)
            preparation[workload] = prepared.get("preparation", {"error": prepared.get("error", "Missing preparation")})

        rows, metadata = collect(ROOT, scratch, repeats, baseline=False)
        archive = scratch / "baseline"
        revision = checkout_baseline(ROOT, archive, scratch / "baseline.tar", baseline_ref)
        fixture_files = ("original.mseed", "calibration.mseed", "expected.json", "baseline.csv")
        fixture_sha = {name: sha(ROOT / "fixtures" / "synthetic" / name) for name in fixture_files}
        check(all(sha(archive / "fixtures" / "synthetic" / name) == digest for name, digest in fixture_sha.items()),
              "The candidate and baseline hammer fixture bytes differ")
        baseline_rows, baseline_metadata = collect(archive, scratch, repeats, baseline=True)
        rows.extend(baseline_rows)
        summary = summarize(rows)
        end_identity = source_identity(ROOT)
        drift = identity["runtime_sha256"] != end_identity["runtime_sha256"]
        bundle = {
            "schema_version": 1, "started_utc": STARTED_UTC, "candidate": identity,
            "candidate_at_completion": end_identity, "runtime_changed_during_run": drift,
            "baseline_commit": revision, "environment": environment(),
            "warm_repeats_per_metric": repeats, "acceptance_repeat_count": repeats == WARM_REPEATS,
            "cold_scope": "First operation in a new process, with prerequisites prepared and OS file cache uncontrolled.",
            "warm_launch_scope": "New process for each observation with reused Qt and Matplotlib configuration caches.",
            "peak_rss_scope": "Process high-water mark through the observation, including prerequisites.",
            "warm_summary": "Nearest-rank p50 and p95 over successful warm observations only.",
            "history_workload": "Hammer fixture pick nudges and raw confirmations every twentieth revision.",
            "raw_capture_scope": "Confirm and apply a staged pick and its already selected raw sample span.",
            "hammer_fixture_sha256": fixture_sha, "preparation": preparation,
            "worker_metadata": {"candidate": metadata, "baseline": baseline_metadata},
            "summary": summary, "baseline_comparison": compare_baseline(summary), "observations": rows,
        }
        (output / "benchmark.json").write_text(json.dumps(bundle, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        with (output / "timings.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    failures = sum(row["error"] is not None for row in rows)
    misses = sum(not row["target"]["met"] for row in summary if "target" in row)
    print(f"Saved {len(rows)} observations with {failures} failures and {misses} unmet engineering targets.")
    if drift:
        print("Candidate runtime files changed during collection. Rerun on the frozen build.")
    return int(bool(drift or failures or (misses and repeats == WARM_REPEATS)))


STARTED_UTC = datetime.now(timezone.utc).isoformat(timespec="seconds")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="Create a new directory for the benchmark bundle")
    parser.add_argument("--warm-repeats", type=int, default=WARM_REPEATS)
    parser.add_argument("--baseline-ref", default=BASELINE, help="Git commit to compare on the same hammer fixture")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    for name in ("checkout", "workspace", "workload", "metric"):
        parser.add_argument("--" + name, help=argparse.SUPPRESS)
    for name in ("count", "started"):
        parser.add_argument("--" + name, type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return 0
    if args.output_dir is None or not 1 <= args.warm_repeats <= WARM_REPEATS:
        parser.error(f"Provide --output-dir and 1–{WARM_REPEATS} warm repetitions")
    try:
        return run(args.output_dir, args.warm_repeats, args.baseline_ref)
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
