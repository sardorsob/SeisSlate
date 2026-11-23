import hashlib
from pathlib import Path
import shutil
from struct import pack_into, unpack_from
import tempfile
import unittest

import numpy as np
from obspy import Stream, Trace, UTCDateTime


FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "synthetic"
CHANNEL = "XX.ORIG..EHZ"
EPOCH = 946_684_800_000_000_000
MS = 1_000_000


def write_source(path, records):
    traces = [
        Trace(
            np.asarray(values, dtype=np.int32),
            header={
                "network": "XX",
                "station": "ORIG",
                "channel": "EHZ",
                "sampling_rate": rate,
                "starttime": UTCDateTime(ns=EPOCH + offset_ns),
            },
        )
        for offset_ns, rate, values in records
    ]
    Stream(traces).write(str(path), format="MSEED", encoding="INT32", reclen=256)


class SourceTests(unittest.TestCase):
    def setUp(self):
        from seisslate import sources

        self.api = sources
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def test_relocated_recording_keeps_sample_identity_and_immutable_data(self):
        path = FIXTURE / "original.mseed"
        copy = self.directory / "même enregistrement.mseed"
        shutil.copyfile(path, copy)
        source = self.api.load_source(path)
        relocated = self.api.load_source(copy)
        original, second = source.segments

        self.assertEqual(source.sha256, hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(source.size_bytes, path.stat().st_size)
        self.assertEqual(source.sha256, relocated.sha256)
        self.assertEqual(original.ref(360), relocated.segments[0].ref(360))
        self.assertEqual(original.time_ns(360), EPOCH + 360 * MS)
        self.assertEqual(second.time_ns(0), EPOCH + 2200 * MS)
        self.assertEqual(original.data[360], relocated.segments[0].data[360])
        with self.assertRaises(ValueError):
            original.data.flags.writeable = True
        self.assertEqual(original.data.shape, (2150,))
        self.assertEqual(path.read_bytes(), copy.read_bytes())

    def test_exact_off_grid_and_gap_keep_distinct_outcomes(self):
        catalog = self.api.SourceCatalog((self.api.load_source(FIXTURE / "original.mseed"),))
        exact = catalog.lookup(CHANNEL, EPOCH + 360 * MS)
        self.assertEqual(exact.status, "exact")
        self.assertEqual([ref.sample_index for ref in exact.candidates], [360])
        self.assertEqual(catalog.time_ns(exact.candidates[0]), EPOCH + 360 * MS)
        off_grid = catalog.lookup(CHANNEL, EPOCH + 360 * MS + 500)
        self.assertEqual(off_grid.status, "off_grid")
        self.assertEqual([ref.sample_index for ref in off_grid.candidates], [360, 361])
        gap = catalog.lookup(CHANNEL, EPOCH + 2165 * MS)
        self.assertEqual(gap.status, "gap")
        self.assertEqual(gap.candidates, ())

    def test_expected_identity_is_checked_before_decoding_replacement_bytes(self):
        path = FIXTURE / "original.mseed"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        size = path.stat().st_size
        with self.assertRaises(self.api.SourceIdentityError):
            self.api.load_source(path, expected_sha256=digest, expected_size_bytes=size + 1)
        copy = self.directory / "replaced.mseed"
        copy.write_bytes(b"?" * size)
        with self.assertRaises(self.api.SourceIdentityError):
            self.api.load_source(copy, expected_sha256=digest, expected_size_bytes=size)
        self.assertEqual(self.api.load_source(path, expected_sha256=digest, expected_size_bytes=size).sha256, digest)

    def test_shared_half_open_selection_spans_gap_and_unequal_grids(self):
        path = self.directory / "unequal.mseed"
        write_source(path, [(0, 500, [1, 2, 3, 4, 5, 6])])
        original = self.api.load_source(path)
        calibration = self.api.load_source(FIXTURE / "calibration.mseed")
        catalog = self.api.SourceCatalog((original, calibration))
        spans = catalog.spans((CHANNEL, "XX.CALB..EHZ"), EPOCH + MS, EPOCH + 5 * MS)
        by_id = {span.segment_id: (span.start, span.stop) for span in spans}
        self.assertEqual(by_id[original.segments[0].segment_id], (1, 3))
        self.assertEqual(by_id[calibration.segments[0].segment_id], (1, 5))

        gapped = self.api.load_source(FIXTURE / "original.mseed")
        catalog = self.api.SourceCatalog((gapped,))
        spans = catalog.spans((CHANNEL,), EPOCH + 2148 * MS, EPOCH + 2202 * MS)
        self.assertEqual([(span.start, span.stop) for span in spans], [(2148, 2150), (0, 2)])

    def test_duplicate_and_overlapping_segments_do_not_resolve_implicitly(self):
        path = self.directory / "overlap.mseed"
        write_source(path, [(0, 100, [1, 2, 3, 4])] * 2 + [(15 * MS, 100, [5, 6])])
        catalog = self.api.SourceCatalog((self.api.load_source(path),))
        result = catalog.lookup(CHANNEL, EPOCH + 20 * MS)
        self.assertEqual(result.status, "ambiguous")
        self.assertEqual(len({ref.segment_id for ref in result.candidates}), 3)
        self.assertEqual(len(catalog.overlaps), 3)
        self.assertEqual(catalog.lookup(CHANNEL, EPOCH + 40 * MS).status, "gap")

    def test_subsample_record_break_does_not_shift_original_timestamps(self):
        path = self.directory / "short-break.mseed"
        write_source(path, [(0, 100, [1, 2, 3]), (34 * MS, 100, [4, 5, 6])])
        source = self.api.load_source(path)
        catalog = self.api.SourceCatalog((source,))
        self.assertEqual([segment.start_ns for segment in source.segments], [EPOCH, EPOCH + 34 * MS])
        self.assertEqual(catalog.lookup(CHANNEL, EPOCH + 30 * MS).status, "gap")
        found = catalog.lookup(CHANNEL, EPOCH + 34 * MS)
        self.assertEqual(found.candidates, (source.segments[1].ref(0),))

    def test_equal_channel_names_can_be_scoped_to_different_files(self):
        first, second = (self.directory / filename for filename in ("first.mseed", "second.mseed"))
        write_source(first, [(0, 100, [1, 2, 3])])
        write_source(second, [(0, 100, [4, 5, 6])])
        left, right = self.api.load_source(first), self.api.load_source(second)
        catalog = self.api.SourceCatalog((left, right))
        self.assertEqual(catalog.lookup(CHANNEL, EPOCH + 10 * MS).status, "ambiguous")
        found = catalog.lookup(CHANNEL, EPOCH + 10 * MS, source_hash=right.sha256)
        self.assertEqual(found.candidates, (right.segments[0].ref(1),))

    def test_ns_derivation_does_not_accumulate_period_rounding(self):
        path = self.directory / "rate3.mseed"
        write_source(path, [(0, 3, [1, 2, 3, 4])])
        segment = self.api.load_source(path).segments[0]
        self.assertEqual(segment.time_ns(1), EPOCH + 333_333_333)
        self.assertEqual(segment.time_ns(2), EPOCH + 666_666_667)
        self.assertEqual(segment.time_ns(3), EPOCH + 1_000_000_000)
        self.assertEqual(self.api.utc_ns("2000-01-01T01:00:00.123456789+01:00"), EPOCH + 123_456_789)

    def test_fractional_header_rate_keeps_contiguous_records_on_one_grid(self):
        path = self.directory / "fractional-rate.mseed"
        write_source(path, [(0, 40 / 3, np.arange(500))])
        self.assertGreater(path.stat().st_size, 256)
        self.assertEqual(unpack_from(">hh", path.read_bytes(), 32), (40, -3))
        source = self.api.load_source(path)
        self.assertEqual(len(source.segments), 1)
        segment = source.segments[0]
        self.assertEqual(segment.npts, 500)
        catalog = self.api.SourceCatalog((source,))
        for index in (49, 50, 499):
            timestamp = EPOCH + index * 75 * MS
            result = catalog.lookup(CHANNEL, timestamp)
            self.assertEqual(result.status, "exact")
            self.assertEqual(result.candidates, (segment.ref(index),))
            self.assertEqual(catalog.time_ns(result.candidates[0]), timestamp)
        np.testing.assert_array_equal(segment.data, np.arange(500))

    def test_blockette_100_rate_overrides_nominal_fixed_header_rate(self):
        path = self.directory / "actual-rate.mseed"
        write_source(path, [(0, 10, [7, 9, 11])])
        record = bytearray(path.read_bytes())
        data_offset, first_blockette = unpack_from(">HH", record, 44)
        # Move the three INT32 samples to fit the 12-byte Blockette 100.
        record[data_offset + 12:data_offset + 24] = record[data_offset:data_offset + 12]
        pack_into(">HHf4x", record, data_offset, 100, 0, 12.5)
        pack_into(">H", record, first_blockette + 2, data_offset)
        pack_into(">H", record, 44, data_offset + 12)
        record[39] += 1
        path.write_bytes(record)
        segment = self.api.load_source(path).segments[0]
        self.assertEqual(segment.sampling_rate_hz, 12.5)
        self.assertEqual(segment.time_ns(2), EPOCH + 160 * MS)
        np.testing.assert_array_equal(segment.data, [7, 9, 11])

    def test_half_nanosecond_rounding_preserves_times_across_exact_grid_partitions(self):
        whole = self.api.Segment("hash", CHANNEL, "whole", EPOCH, 400_000_000, np.arange(5))
        latter = self.api.Segment("hash", CHANNEL, "latter", EPOCH + 5, 400_000_000, np.arange(2, 5))
        self.assertEqual(whole.time_ns(3), EPOCH + 8)
        self.assertEqual(latter.time_ns(1), EPOCH + 8)

    def test_out_of_bounds_or_wrong_channel_ref_is_rejected(self):
        catalog = self.api.SourceCatalog((self.api.load_source(FIXTURE / "original.mseed"),))
        segment = catalog.sources[0].segments[0]
        with self.assertRaises(ValueError):
            segment.ref(segment.npts)
        wrong = self.api.SampleRef(segment.source_hash, "WRONG", segment.segment_id, 360)
        with self.assertRaises(ValueError):
            catalog.time_ns(wrong)
        with self.assertRaises(ValueError):
            catalog.spans((CHANNEL,), EPOCH + MS, EPOCH)


if __name__ == "__main__":
    unittest.main()
