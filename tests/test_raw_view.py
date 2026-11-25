"""Exercise the standalone Qt raw inspector with original sample identities."""

from fractions import Fraction
from importlib.util import find_spec
from unittest import TestCase, skipUnless
from unittest.mock import patch

import numpy as np

from seisslate.sources import SampleSpan, Segment


START_NS = 946_684_800_123_456_789
HAS_DESKTOP = all(find_spec(name) is not None for name in ("PyQt5", "matplotlib"))


def segment(data, rate=Fraction(30, 7)):
    return Segment("source", "XX.RAW..EHZ", "segment", START_NS, rate, data)


@skipUnless(HAS_DESKTOP, "Install SeisSlate with the desktop extra")
class RawViewTest(TestCase):
    def setUp(self):
        from PyQt5.QtWidgets import QApplication

        from seisslate.raw_view import RawView

        self.app = QApplication.instance() or QApplication([])
        self.panel = RawView()
        self.panel.resize(650, 200)
        self.addCleanup(self.panel.close)
        self.panel.show()

    def test_rational_span_renders_every_source_value_and_keyboard_steps(self):
        from PyQt5.QtCore import QPoint, QRect, Qt
        from PyQt5.QtTest import QTest

        source = segment(np.array([100, 999, 31, -48], dtype=">i4"))
        span = SampleSpan(source.segment_id, 2, 4)
        point = source.ref(2)
        bounds = (source.ref(2), source.ref(3))
        self.panel.show_span(source, span, point=point, bounds=bounds)
        view = self.panel.view
        self.assertIsNotNone(view)
        self.assertEqual((view.source_hash, view.channel), (source.source_hash, source.channel))
        self.assertEqual((view.span, view.point, view.bounds), (span, point, bounds))
        self.assertEqual((view.display_start_ns, view.display_stop_ns), (START_NS + 466_666_667, START_NS + 933_333_334))
        line = self.panel.axes.lines[0]
        np.testing.assert_array_equal(line.get_xdata(), [0.0, 0.233333333])
        self.assertEqual(line.get_ydata().dtype.str, ">i4")
        self.assertEqual(line.get_ydata().tobytes(), source.data[2:4].tobytes())
        self.assertEqual(line.get_linestyle(), "None")
        self.assertEqual(self.panel.axes.get_yscale(), "linear")
        self.assertIn("units unknown", self.panel.readout.text())
        self.panel.resize(440, 170)
        self.app.processEvents()
        control = self.panel.sample_control
        control_rect = QRect(control.mapTo(self.panel, QPoint()), control.size())
        self.assertFalse(self.panel.canvas.geometry().intersects(control_rect))
        self.assertIn("Source sample 2", self.panel.readout.text())
        self.assertIn("2000-01-01T00:00:00.590123456Z", self.panel.readout.text())
        self.assertIn("Raw 31", self.panel.readout.text())
        self.assertIn("int32 (>i4)", self.panel.readout.text())

        self.panel.sample_control.setFocus()
        QTest.keyClick(self.panel.sample_control, Qt.Key_Up)
        self.assertIn("Source sample 3", self.panel.readout.text())
        self.assertIn("2000-01-01T00:00:00.823456789Z", self.panel.readout.text())
        self.assertIn("Raw -48", self.panel.readout.text())
        self.assertIs(self.panel.view, view)

    def test_one_sample_and_equal_inclusive_bounds(self):
        source = segment(np.array([9, -7, 11], dtype=np.int32), Fraction(3, 2))
        point = source.ref(1)
        self.panel.show_span(source, SampleSpan(source.segment_id, 1, 2), point=point, bounds=(point, point))

        self.assertEqual(self.panel.view.display_stop_ns, START_NS + 1_333_333_333)
        self.assertEqual(self.panel.sample_control.minimum(), self.panel.sample_control.maximum())
        np.testing.assert_array_equal(self.panel.axes.lines[0].get_ydata(), [-7])
        self.assertIn("zero width", self.panel.span_label.text())
        self.assertIn("2000-01-01T00:00:00.790123456Z", self.panel.readout.text())

    def test_bad_span_or_bounds_preserve_previous_view(self):
        from seisslate.raw_view import MAX_RAW_SAMPLES

        source = segment(np.arange(MAX_RAW_SAMPLES + 1, dtype=np.int32))
        span = SampleSpan(source.segment_id, 0, 1)
        self.panel.show_span(source, span)
        old_view, old_text = self.panel.view, self.panel.readout.text()
        with self.assertRaisesRegex(ValueError, "Narrow.*20,000"):
            self.panel.show_span(source, SampleSpan(source.segment_id, 0, source.npts))
        with self.assertRaisesRegex(ValueError, "source segment"):
            self.panel.show_span(source, SampleSpan("another-segment", 0, 1))
        with self.assertRaisesRegex(ValueError, "point and bounds"):
            self.panel.show_span(source, span, point=source.ref(0), bounds=(source.ref(0), source.ref(1)))
        self.assertIs(self.panel.view, old_view)
        self.assertEqual(self.panel.readout.text(), old_text)
        np.testing.assert_array_equal(self.panel.axes.lines[0].get_ydata(), [0])

    def test_displayed_bounds_are_immutable(self):
        source = segment(np.arange(4, dtype=np.int32))
        span = SampleSpan(source.segment_id, 0, 4)
        point = source.ref(1)
        bounds = [source.ref(0), source.ref(2)]
        self.panel.show_span(source, span, point=point, bounds=bounds)
        bounds[1] = source.ref(3)
        self.assertEqual(self.panel.view.bounds, (source.ref(0), source.ref(2)))

    def test_same_span_can_retry_after_a_plot_failure(self):
        source = segment(np.array([8, 0], dtype=np.int32))
        span = SampleSpan(source.segment_id, 0, 2)
        with patch.object(self.panel.canvas, "draw", side_effect=ValueError("Plot failed")):
            self.panel.show_span(source, span)
        self.assertIsNone(self.panel.view)
        self.panel.show_span(source, span)
        self.assertIsNotNone(self.panel.view)

    def test_exact_values_and_unavailable_plot(self):
        for data, expected, can_plot in (
            (np.array([np.iinfo(np.int64).min], dtype=np.int64), "-9223372036854775808", True),
            (np.array([np.iinfo(np.uint64).max], dtype=np.uint64), "18446744073709551615", True),
            (np.array([np.finfo(np.float32).max], dtype=np.float32), "3.4028234663852886e+38", True),
            (np.array([-0.0]), "-0.0", True),
            (np.array([np.finfo(np.float64).max]), "1.7976931348623157e+308", False),
            (np.array([np.inf]), "inf", False),
            (np.array([np.nan]), "nan", False),
        ):
            with self.subTest(dtype=data.dtype, expected=expected):
                source = segment(data)
                self.panel.show_span(source, SampleSpan(source.segment_id, 0, 1))
                self.assertIn(f"Raw {expected} ·", self.panel.readout.text())
                self.assertEqual(self.panel.canvas.isVisible(), can_plot)
                if can_plot:
                    np.testing.assert_array_equal(self.panel.axes.lines[0].get_ydata(), data)
                    self.assertIsNotNone(self.panel.view)
                else:
                    self.assertIsNone(self.panel.view)
                    self.assertIn("Plot and capture unavailable", self.panel.status.text())

        self.panel.clear()
        self.assertFalse(self.panel.sample_control.isEnabled())
        self.assertIsNone(self.panel.view)
        self.assertEqual(self.panel.readout.text(), "")
