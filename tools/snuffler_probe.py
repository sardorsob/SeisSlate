"""Exercise exact source-sample nudging in Snuffler on the checked-in fixture.

Run with --check to drive the real marker table and nudge button through Qt.
Run without --check to repeat the interaction in the visible window.
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Run the Qt exercise and exit.")
    args = parser.parse_args()

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
