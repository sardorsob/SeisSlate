import unittest
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.signal import iirfilter, sosfilt

from seisslate.landslide import Worksheet, build_rows, processing_definition
from seisslate.sources import NS_PER_SECOND, Segment, Source, SourceCatalog, utc_ns


CHANNEL = "XX.TEST..BHZ"
UNAVAILABLE = "XX.MISSING..BHZ"
START = utc_ns("2015-10-17T20:19:00.123456789Z")
HASH = "4" * 64


def make_segment(name, rate=20, start=START, values=None):
    data = np.zeros(10_000, dtype=np.int32) if values is None else np.asarray(values)
    return Segment(HASH, CHANNEL, name, start, rate, data)


def catalog(*segments):
    return SourceCatalog((Source(Path("fixture.mseed"), HASH, 0, segments),))


class LandslideRowsTest(unittest.TestCase):
    def test_explicit_channel_and_band_rows_include_unreviewed_and_unavailable(self):
        segment = make_segment("s", rate=10)
        rows = build_rows(catalog(segment), (CHANNEL, UNAVAILABLE))
        self.assertEqual([(row.channel, row.band, row.initial_status) for row in rows], [
            (CHANNEL, "HF", "insufficient_support"),
            (CHANNEL, "LP", "unreviewed"),
            (UNAVAILABLE, "HF", "unavailable"),
            (UNAVAILABLE, "LP", "unavailable"),
        ])
        self.assertEqual(rows[0].waveforms[0].support_reason, "nyquist")
        self.assertEqual(rows[0].segments, (segment,))
        self.assertEqual(Worksheet(rows).result(UNAVAILABLE, "LP").end.status, "unavailable")

    def test_cutoff_close_to_nyquist_cannot_silently_become_highpass(self):
        segment = make_segment("s", rate=10.000001)
        row, _ = build_rows(catalog(segment), (CHANNEL,))
        self.assertEqual(row.waveforms[0].support_reason, "nyquist")
        self.assertEqual(row.waveforms[0].values.size, 0)

    def test_frozen_filter_matches_independent_sos_and_preserves_raw_samples(self):
        for band_index, (bounds, guard_seconds, frequency) in enumerate((
            ((1, 5), 2, 2.5), ((0.01, 0.05), 200, 0.025),
        )):
            with self.subTest(band=bounds):
                rate = 20
                values = (200 * np.sin(2 * np.pi * frequency * np.arange(12_000) / rate)).astype(np.int32)
                values[0:200] += 50
                values[6000] += 700
                segment = make_segment("s", rate=rate, values=values)
                waveform = build_rows(catalog(segment), (CHANNEL,))[band_index].waveforms[0]
                sos = iirfilter(4, [2 * hz / rate for hz in bounds], btype="band", ftype="butter", output="sos")
                expected = sosfilt(sos, sosfilt(sos, values)[::-1])[::-1]
                np.testing.assert_array_equal(waveform.values, expected)
                np.testing.assert_array_equal(segment.data, values)
                self.assertFalse(waveform.values.flags.writeable)
                self.assertEqual((waveform.support.start, waveform.support.stop),
                                 (guard_seconds * rate, len(values) - guard_seconds * rate))

    def test_impulse_step_and_sinusoids_at_pilot_sample_rate(self):
        rate, count = 50, 100_000
        center = count // 2
        impulse = np.zeros(count)
        impulse[center] = 1
        step = np.ones(count)
        time = np.arange(count) / rate
        for band_index, (lo, hi) in enumerate(((1, 5), (0.01, 0.05))):
            with self.subTest(band=(lo, hi)):
                def process(values):
                    segment = make_segment("s", rate=rate, values=values)
                    return build_rows(catalog(segment), (CHANNEL,))[band_index].waveforms[0]

                pulse = process(impulse).values
                np.testing.assert_allclose(pulse[center - 1000:center][::-1],
                                           pulse[center + 1:center + 1001], rtol=0, atol=1e-10)
                step_response = process(step)
                start, stop = step_response.support.start, step_response.support.stop
                self.assertLess(np.max(np.abs(step_response.values[start:stop])), 0.01)

                in_band = process(np.sin(2 * np.pi * (lo * hi) ** 0.5 * time)).values[start:stop]
                below_band = process(np.sin(2 * np.pi * lo / 4 * time)).values[start:stop]
                self.assertAlmostEqual(np.sqrt(2 * np.mean(in_band ** 2)), 1.0, places=2)
                self.assertLess(np.sqrt(2 * np.mean(below_band ** 2)), 0.001)

    def test_review_window_is_half_open_and_does_not_change_filter_input(self):
        segment = make_segment("s", rate=37, values=np.arange(1500))
        data = catalog(segment)
        normal = build_rows(data, (CHANNEL,))[0].waveforms[0]
        clipped = build_rows(data, (CHANNEL,),
                             start_ns=segment.time_ns(87) + 1, stop_ns=segment.time_ns(100))[0].waveforms[0]
        self.assertEqual((clipped.support.start, clipped.support.stop), (88, 100))
        np.testing.assert_array_equal(clipped.values, normal.values)
        worksheet = Worksheet(build_rows(data, (CHANNEL,), start_ns=segment.time_ns(88),
                                         stop_ns=segment.time_ns(100)))
        for index in (87, 100):
            with self.assertRaisesRegex(ValueError, "support"):
                worksheet.preview(CHANNEL, "HF", "onset", "identifiable", "edge", segment.ref(index))

    def test_filter_does_not_read_across_nonfinite_gap(self):
        values = np.r_[np.zeros(100), np.nan, np.ones(101)]
        segment = make_segment("s", values=values)
        row = build_rows(catalog(segment), (CHANNEL,))[0]
        first, second = row.waveforms
        self.assertEqual((first.first_index, first.stop_index), (0, 100))
        self.assertEqual((second.first_index, second.stop_index), (101, 202))
        np.testing.assert_array_equal(first.values, np.zeros(100))
        for index in (39, 60, 100, 140, 162):
            with self.assertRaisesRegex(ValueError, "support"):
                Worksheet((row,)).preview(CHANNEL, "HF", "end", "identifiable", "edge", segment.ref(index))
        worksheet = Worksheet((row,))
        worksheet.apply(worksheet.preview(CHANNEL, "HF", "onset", "identifiable", "left", segment.ref(50)))
        result = worksheet.preview(CHANNEL, "HF", "end", "identifiable", "right", segment.ref(150)).result
        self.assertEqual(result.duration_status, "separate_support")
        self.assertIsNone(result.duration_seconds)

    def test_acquisition_gap_overlap_and_short_support(self):
        left = make_segment("left", values=np.arange(100))
        right = make_segment("right", start=START + 20 * NS_PER_SECOND, values=np.arange(100))
        worksheet = Worksheet(build_rows(catalog(left, right), (CHANNEL,)))
        worksheet.apply(worksheet.preview(CHANNEL, "HF", "onset", "identifiable", "left", left.ref(51)))
        result = worksheet.preview(CHANNEL, "HF", "end", "identifiable", "right", right.ref(51)).result
        self.assertEqual(result.duration_status, "separate_support")
        self.assertIsNone(result.duration_seconds)

        overlapping = make_segment("overlap", start=left.time_ns(90), values=np.arange(100))
        row = build_rows(catalog(left, overlapping), (CHANNEL,))[0]
        self.assertEqual(row.initial_status, "insufficient_support")
        self.assertTrue(all(w.support_reason == "overlap" for w in row.waveforms))
        short = make_segment("short", values=np.arange(80))
        self.assertEqual(build_rows(catalog(short), (CHANNEL,))[0].waveforms[0].support_reason, "edge_guard")

    def test_overlap_in_filter_input_outside_review_window_blocks_processing(self):
        left = make_segment("left", values=np.arange(1000))
        other = make_segment("other", start=left.time_ns(900), values=np.arange(200))
        row, _ = build_rows(catalog(left, other), (CHANNEL,), start_ns=left.time_ns(100), stop_ns=left.time_ns(700))
        self.assertEqual(row.initial_status, "insufficient_support")
        self.assertEqual(row.waveforms[0].support_reason, "overlap")


class WorksheetTest(unittest.TestCase):
    def setUp(self):
        self.segment = make_segment("s", rate=37)
        self.bounds = {"start_ns": self.segment.time_ns(80) - 1, "stop_ns": self.segment.time_ns(5000) + 1}
        self.rows = build_rows(catalog(self.segment), (CHANNEL, UNAVAILABLE), **self.bounds)
        self.worksheet = Worksheet(self.rows)

    def preview(self, boundary, status, index=None):
        sample = None if index is None else self.segment.ref(index)
        return self.worksheet.preview(CHANNEL, "HF", boundary, status, "analyst evidence", sample)

    def test_independent_boundaries_preview_apply_undo_and_duration(self):
        initial = self.worksheet.result(CHANNEL, "HF")
        end = self.preview("end", "identifiable", 116)
        self.assertEqual(self.worksheet.result(CHANNEL, "HF"), initial)
        self.worksheet.apply(end)
        onset = self.preview("onset", "ambiguous")
        self.worksheet.apply(onset)
        self.assertEqual(onset.result.onset.status, "ambiguous")
        self.assertEqual(onset.result.end.utc_ns, self.segment.time_ns(116))
        self.assertEqual(onset.result.status, "identifiable")
        self.assertEqual(onset.result.duration_status, "missing_boundary")

        for index in (116, 120):
            self.assertEqual(self.preview("onset", "identifiable", index).result.duration_status, "unordered")
        eligible = self.preview("onset", "identifiable", 101)
        self.assertEqual(eligible.result.duration_seconds, float(Fraction(15, 37)))
        self.worksheet.apply(eligible)
        self.assertTrue(self.worksheet.undo())
        self.assertEqual(self.worksheet.result(CHANNEL, "HF"), onset.result)
        with self.assertRaisesRegex(ValueError, "stale"):
            self.worksheet.apply(end)

    def test_reviewed_without_a_signal_requires_explicit_review_of_both_boundaries(self):
        onset = self.preview("onset", "no_identifiable_signal")
        self.worksheet.apply(onset)
        self.assertEqual(onset.result.status, "unreviewed")
        self.assertEqual(self.preview("end", "ambiguous").result.status, "ambiguous")
        self.assertEqual(self.preview("end", "insufficient_support").result.status, "insufficient_support")
        self.assertEqual(self.preview("end", "no_identifiable_signal").result.status, "no_identifiable_signal")

    def test_objective_row_availability_cannot_be_overridden(self):
        invalid = ((UNAVAILABLE, "HF", "unreviewed"), (CHANNEL, "HF", "unavailable"),
                   (CHANNEL, "LP", "unavailable"), (CHANNEL, "LP", "unreviewed"))
        for channel, band, status in invalid:
            with self.subTest(channel=channel, band=band, status=status):
                with self.assertRaisesRegex(ValueError, "availability"):
                    self.worksheet.preview(channel, band, "onset", status, "reason")
        self.assertEqual(self.preview("end", "insufficient_support").selection.status, "insufficient_support")

    def test_no_signal_requires_one_run_covering_the_full_explicit_window(self):
        segment = make_segment("s", values=np.r_[np.zeros(200), np.nan, np.zeros(250)])
        for start, stop in (
            (None, None),
            (segment.time_ns(30), segment.time_ns(140)),
            (segment.time_ns(100), segment.time_ns(175)),
            (segment.time_ns(100), segment.time_ns(350)),
        ):
            with self.subTest(start=start, stop=stop):
                row, _ = build_rows(catalog(segment), (CHANNEL,), start_ns=start, stop_ns=stop)
                worksheet = Worksheet((row,))
                with self.assertRaisesRegex(ValueError, "full review window"):
                    worksheet.preview(CHANNEL, "HF", "end", "no_identifiable_signal", "looked")
                self.assertEqual(worksheet.preview(CHANNEL, "HF", "end", "ambiguous", "unclear").selection.status,
                                 "ambiguous")
        row, _ = build_rows(catalog(segment), (CHANNEL,), start_ns=segment.time_ns(40) - 1,
                            stop_ns=segment.time_ns(160))
        self.assertTrue(row.full_review_support)

    def test_wrong_source_channel_time_support_and_status_do_not_become_picks(self):
        good = self.segment.ref(100)
        for bad in (replace(good, source_hash="bad"), replace(good, channel="OTHER"), self.segment.ref(0)):
            with self.assertRaisesRegex(ValueError, "support"):
                self.worksheet.preview(CHANNEL, "HF", "onset", "identifiable", "evidence", bad)
        with self.assertRaisesRegex(ValueError, "sample"):
            self.preview("onset", "ambiguous", 100)
        with self.assertRaisesRegex(ValueError, "sample"):
            self.preview("onset", "identifiable")
        with self.assertRaisesRegex(ValueError, "reason"):
            self.worksheet.preview(CHANNEL, "HF", "onset", "ambiguous", "  ")

    def test_portable_replay_checks_original_utc_processing_and_hashes(self):
        self.worksheet.apply(self.preview("onset", "identifiable", 101))
        self.worksheet.apply(self.preview("end", "identifiable", 116))
        self.worksheet.undo()
        payload = self.worksheet.to_dict()
        self.assertEqual(payload["processing"], processing_definition())
        self.assertEqual(Worksheet.from_dict(self.rows, payload).to_dict(), payload)
        self.assertEqual(payload["results"][0]["onset"]["utc_ns"], self.segment.time_ns(101))

        changed_raw = make_segment("s", rate=37, values=np.ones(10_000, dtype=np.int32))
        changed_rows = build_rows(catalog(changed_raw), (CHANNEL, UNAVAILABLE), **self.bounds)
        with self.assertRaisesRegex(ValueError, "identity"):
            Worksheet.from_dict(changed_rows, payload)
        payload["edits"][0]["selection"]["utc_ns"] += 1
        with self.assertRaisesRegex(ValueError, "replay"):
            Worksheet.from_dict(self.rows, payload)


if __name__ == "__main__":
    unittest.main()
