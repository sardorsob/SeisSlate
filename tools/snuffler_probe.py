"""Exercise exact source-sample nudging in Snuffler on the checked-in fixture.

Run with --check to drive the real marker table and nudge button through Qt.
Run without --check to repeat the interaction in the visible window.
Run with --raw-view-check to audit the display inputs and review-session reopen.
"""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import platform


def utc(timestamp):
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat(
        timespec="microseconds"
    ).replace("+00:00", "Z")


def fixture_time(iso_time):
    return datetime.fromisoformat(iso_time).timestamp()


def sample_for_marker(source_pile, marker):
    """Resolve an on-grid point to one unprocessed MiniSEED segment."""
    timestamp = marker.get_tmin()
    nslcs = marker.get_nslc_ids()
    if marker.get_tmax() != timestamp or len(nslcs) != 1:
        raise ValueError("Select one point marker with a single trace ID.")

    matches = []
    for trace in source_pile.iter_traces(
        trace_selector=lambda trace: trace.nslc_id == nslcs[0]
    ):
        index = round((timestamp - trace.tmin) / trace.deltat)
        sample_time = trace.tmin + index * trace.deltat
        if 0 <= index < trace.data_len() and math.isclose(
            sample_time, timestamp, rel_tol=0, abs_tol=math.ulp(timestamp)
        ):
            matches.append((trace, index))

    if len(matches) != 1:
        raise ValueError("The pick does not resolve to exactly one source sample.")
    return matches[0]


def check_raw_view():
    """Compare stock wiggle draw inputs with immutable samples on the pinned viewer."""
    from importlib.metadata import version
    from tempfile import TemporaryDirectory
    from unittest.mock import patch

    import numpy as np
    from pyrocko import info
    from pyrocko.gui.qt_compat import qc, qw
    from pyrocko.gui.snuffler import pile_viewer, pile_viewer_waterfall, snuffling
    from PyQt5.QtTest import QTest

    from seisslate.example import load_example
    from seisslate.ui import launch_workbench

    assert version("pyrocko") == "2026.11.5", "Use the pinned SeisSlate Pyrocko to reproduce this audit"
    print(json.dumps({
        "platform": platform.platform(), "python": platform.python_version(),
        "pyrocko": info.version, "pyrocko_commit": info.git_sha1,
        "qt": qc.qVersion(), "pyqt": qc.PYQT_VERSION_STR,
        "upstream_sources": {
            module.__name__: {
                "path": module.__file__, "sha256": sha256(Path(module.__file__).read_bytes()).hexdigest(),
            }
            for module in (pile_viewer, pile_viewer_waterfall, snuffling)
        },
    }), flush=True)

    class ReversePolarity(snuffling.Snuffling):
        def setup(self):
            self.set_name("Probe polarity reversal")
            self.set_have_post_process_hook(True)

        def post_process_hook(self, traces):
            for trace in traces:
                trace.set_ydata(-trace.get_ydata())
            return traces

    def exercise(workbench):
        viewer = workbench.viewer
        container = viewer.parent()
        segment = workbench.segments[0]
        nslc = tuple(segment.channel.split("."))
        source_bytes = segment.data.tobytes()
        qw.QApplication.processEvents()
        workbench.select_blow("B01")

        def draw_inputs(label):
            observed = []
            prepare = viewer.prepare_cutout2

            def observe(*args, **kwargs):
                traces = prepare(*args, **kwargs)
                observed.extend(trace.copy() for trace in traces if trace.nslc_id == nslc)
                return traces

            with patch.object(viewer, "prepare_cutout2", side_effect=observe):
                assert not viewer.grab().isNull(), "Qt did not render the stock viewer"
            expected_start = segment.start_ns / 1e9
            expected_stop = segment.stop_ns / 1e9
            trace = next((trace for trace in observed if expected_start <= trace.tmin < expected_stop), None)
            assert trace is not None, f"No wiggle draw input in {segment.segment_id}"
            first = round((trace.tmin - segment.start_ns / 1e9) / trace.deltat)
            data = trace.get_ydata()
            assert trace.deltat == 1 / segment.sampling_rate_hz and first >= 0
            raw = segment.data[first:first + data.size]
            assert raw.shape == data.shape, (first, data.size, segment.npts, viewer.get_time_range())
            matches = data.dtype == raw.dtype and data.tobytes() == raw.tobytes()
            print(json.dumps({
                "state": label, "sample_span": [first, first + data.size],
                "raw_match": matches, "max_abs_difference": float(np.max(np.abs(data.astype(float) - raw))),
                "display_dtype": str(data.dtype), "source_dtype": str(raw.dtype),
                "time_range_s": list(viewer.get_time_range()),
                "lowpass_hz": viewer.lowpass, "highpass_hz": viewer.highpass,
                "fft_filtering": viewer.menuitem_fft_filtering.isChecked(),
                "demean": viewer.menuitem_demean.isChecked(),
                "degap": viewer.menuitem_degap.isChecked(),
                "gain": viewer.gain, "rotation_deg": viewer.rotate,
            }), flush=True)
            return data, raw, matches

        assert draw_inputs("startup_wiggle")[2]
        viewer.menuitem_demean.trigger()
        data, raw, matches = draw_inputs("demean_enabled")
        assert not matches
        np.testing.assert_allclose(data.mean(), 0.0, atol=1e-10)
        viewer.menuitem_demean.trigger()
        assert draw_inputs("demean_reset")[2]

        before_gain = viewer.grab().toImage()
        container.gain_control.edited(2.0)
        assert viewer.gain == 2.0 and draw_inputs("gain_projection_only")[2]
        assert viewer.grab().toImage() != before_gain
        container.gain_control.edited(1.0)

        lowpass, highpass = container.lowpass_control, container.highpass_control
        highpass.edited(8.0)
        assert viewer.highpass == 8.0 and not draw_inputs("highpass_enabled")[2]
        viewer.menuitem_fft_filtering.trigger()
        lowpass.edited(30.0)
        assert not draw_inputs("fft_highpass_and_lowpass")[2]
        highpass.edited(highpass.mi)
        lowpass.edited(lowpass.ma)
        viewer.menuitem_fft_filtering.trigger()
        assert viewer.highpass is None and viewer.lowpass is None and draw_inputs("filters_reset")[2]

        hook = ReversePolarity()
        hook.setup()
        viewer.add_snuffling(hook)
        hook.set_live_update(True)
        data, raw, matches = draw_inputs("local_post_hook_with_filter_controls_off")
        assert viewer.lowpass is None and viewer.highpass is None and not matches
        np.testing.assert_array_equal(data, -raw)
        hook.set_live_update(False)
        assert draw_inputs("post_hook_reset")[2]

        revision_id = workbench.history.current.revision_id
        def click(name):
            button = workbench.buttons[name]
            assert button.isEnabled(), name
            QTest.mouseClick(button, qc.Qt.LeftButton)

        click("later")
        assert workbench.history.draft is not None
        lowpass.edited(30.0)
        assert viewer.lowpass == 30.0 and not draw_inputs("lowpass_immediately_before_apply")[2]
        click("apply")
        applied = workbench.history.current
        assert applied.revision_id != revision_id and workbench.history.draft is None
        with TemporaryDirectory() as directory:
            path = Path(directory) / "probe-review.json"
            with patch.object(qw.QFileDialog, "getSaveFileName", return_value=(str(path), "")):
                workbench.save_action.trigger()
            assert b"raw-v1" not in path.read_bytes()
            previous_history = workbench.history
            with patch.object(qw.QFileDialog, "getOpenFileName", return_value=(str(path), "")):
                workbench.open_session_action.trigger()
            assert workbench.history is not previous_history, "Qt did not reopen the saved review"
            assert workbench.session_path == path
            assert workbench.history.current == applied
        workbench.select_blow("B01")
        assert viewer.lowpass == 30.0 and not draw_inputs("same_window_reopen_keeps_lowpass")[2]
        lowpass.edited(lowpass.ma)
        assert viewer.lowpass is None and draw_inputs("lowpass_reset_after_reopen")[2]
        assert segment.data.tobytes() == source_bytes
        print(json.dumps({"applied_revision": applied.revision_id, "saved_raw_v1": False}), flush=True)

    launch_workbench(load_example(), check=exercise)
    print("PASS: stock display can differ from raw; legacy Apply records no view evidence.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Run the Qt exercise and exit.")
    mode.add_argument("--raw-view-check", action="store_true", help="Run the raw-view audit and exit.")
    args = parser.parse_args()
    if args.raw_view_check:
        check_raw_view()
        return

    import pyrocko
    from pyrocko import config, pile
    from pyrocko.gui.qt_compat import qc, qw
    from pyrocko.gui.snuffler.marker import Marker
    from pyrocko.gui.snuffler.marker_editor import MarkerTableView
    from pyrocko.gui.snuffler.snuffling import Snuffling
    from PyQt5.QtTest import QTest

    fixture = Path(__file__).resolve().parents[1] / "fixtures" / "synthetic"
    manifest = json.loads((fixture / "expected.json").read_text(encoding="utf-8"))
    reference = manifest["snuffler_probe"]
    sources = manifest["sources"]
    paths = [fixture / source["path"] for source in sources.values()]

    def check_source_fingerprints():
        for path, source in zip(paths, sources.values(), strict=True):
            if sha256(path.read_bytes()).hexdigest() != source["sha256"]:
                raise ValueError(f"Unexpected fixture bytes in {path}")

    check_source_fingerprints()
    source_pile = pile.make_pile([str(path) for path in paths], fileformat="mseed")

    orig_time = fixture_time(reference["original_pick_utc"])
    calb_time = fixture_time(reference["calibration_pick_utc"])
    original = Marker([tuple(sources["original"]["trace_id"].split("."))], orig_time, orig_time)
    calibration = Marker([tuple(sources["calibration"]["trace_id"].split("."))], calb_time, calb_time)

    class Probe(Snuffling):
        def setup(self):
            self.set_name("SeisSlate sample probe")
            self.current = original
            self.last_move = None

        def call(self):
            self.last_move = None
            try:
                selected = self.get_selected_markers()
                if selected != [self.current]:
                    raise ValueError("Select the earlier Original marker in the Markers table.")
                trace, index = sample_for_marker(self.get_pile(), self.current)
                if index + 1 >= trace.data_len():
                    raise ValueError("The next sample would leave this source segment.")
            except ValueError as error:
                self.status.setText(str(error))
                return

            moved_time = trace.tmin + (index + 1) * trace.deltat
            replacement = Marker(self.current.get_nslc_ids(), moved_time, moved_time)
            self.cleanup()
            self.add_marker(replacement)
            self.current = replacement
            self.get_viewer().update()
            self.last_move = {
                "trace_id": ".".join(trace.nslc_id),
                "segment_start_utc": utc(trace.tmin),
                "before_index": index,
                "before_utc": utc(trace.tmin + index * trace.deltat),
                "after_index": index + 1,
                "after_utc": utc(moved_time),
                "partner_utc": utc(calibration.get_tmin()),
            }
            self.status.setText(f"Original sample {index} → {index + 1} at {utc(moved_time)}")
            print(json.dumps(self.last_move), flush=True)

    class PaintCounter(qc.QObject):
        count = 0

        def eventFilter(self, watched, event):
            if event.type() == qc.QEvent.Paint:
                self.count += 1
            return False

    errors = []

    def launch(window):
        viewer = window.get_view()
        probe = Probe()
        probe.setup()
        viewer.add_snuffling(probe)
        probe.add_marker(original)
        viewer.set_time_range(orig_time - 0.055, calb_time + 0.055)
        viewer.toggle_marker_editor()
        panel = qw.QWidget()
        layout = qw.QHBoxLayout(panel)
        button = qw.QPushButton("Nudge +1 sample")
        probe.status = qw.QLabel("Select the earlier Original marker in the Markers table.")
        button.clicked.connect(probe.call)
        layout.addWidget(button)
        layout.addWidget(probe.status, 1)
        window.add_panel(probe.get_name(), panel, visible=True)
        window.resize(1200, 740)
        if not args.check:
            return

        def check():
            try:
                table = window.findChild(MarkerTableView)
                assert table.isVisible(), "The native Markers table did not open."
                assert table.model().rowCount() == 2, "Expected two native markers."
                row = table.model().index(0, 0)
                QTest.mouseClick(table.viewport(), qc.Qt.LeftButton, pos=table.visualRect(row).center())
                assert probe.get_selected_markers() == [original], "The Original pick did not select."
                before = viewer.grab().toImage()
                counter = PaintCounter()
                viewer.installEventFilter(counter)
                QTest.mouseClick(button, qc.Qt.LeftButton)
                qw.QApplication.processEvents()
                assert counter.count > 0, "Qt did not repaint the waveform after nudge."
                viewer.removeEventFilter(counter)
                after = viewer.grab().toImage()
                left, right = viewer.get_time_range()
                pixels_per_second = before.width() / (right - left)
                x_old = round((orig_time - left) * pixels_per_second)
                x_new = round((fixture_time(reference["nudged_pick_utc"]) - left) * pixels_per_second)
                marker_band = qc.QRect(x_old - 2, 0, x_new - x_old + 5, before.height())
                assert before.copy(marker_band) != after.copy(marker_band), (
                    "Repaint did not change pixels around the original and nudged marker."
                )
                assert probe.last_move == {
                    "trace_id": sources["original"]["trace_id"],
                    "segment_start_utc": reference["original_segment_start_utc"],
                    "before_index": reference["original_sample_index"],
                    "before_utc": reference["original_pick_utc"],
                    "after_index": reference["nudged_sample_index"],
                    "after_utc": reference["nudged_pick_utc"],
                    "partner_utc": reference["calibration_pick_utc"],
                }, "Nudge did not match the fixture expectation."
                assert list(probe.get_markers()) == [probe.current, calibration]
                assert sample_for_marker(probe.get_pile(), probe.current)[1] == reference["nudged_sample_index"]
                print(f"PASS: native marker selection, exact sample nudge and {counter.count} repaint(s).")
            except Exception as error:
                errors.append(error)
            finally:
                window.close()

        qc.QTimer.singleShot(0, check)

    print(f"{platform.platform()} | Python {platform.python_version()} | "
          f"Pyrocko {pyrocko.__version__} | Qt {qc.qVersion()} | PyQt {qc.PYQT_VERSION_STR}")
    # Use an isolated PYROCKO_DIR for the probe so the first-start wizard stays out of the exercise.
    config.raw_config("snuffler").first_start = False
    pyrocko.snuffle(
        source_pile, markers=[calibration], ntracks=2, controls=False,
        opengl=False, instant_close=True, launch_hook=launch
    )
    check_source_fingerprints()
    if errors:
        raise SystemExit(f"FAIL: {errors[0]}")


if __name__ == "__main__":
    main()
