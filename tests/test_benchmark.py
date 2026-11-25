"""Check timing summaries without running the synthetic workloads."""

from unittest import TestCase

from tools.benchmark_review import compare_baseline, summarize


class BenchmarkSummaryTests(TestCase):
    def test_cold_and_warm_summary_use_nearest_rank_and_preserve_failures(self):
        def observation(phase, ms, peak, error=None):
            return {
                "build": "candidate", "workload": "hammer", "metric": "apply_refresh",
                "phase": phase, "duration_ms": ms, "peak_rss_bytes": peak, "error": error,
            }

        rows = [
            observation("cold", 91, 9_000),
            *(observation("warm", n, n * 1000) for n in range(20, 0, -1)),
            observation("warm", None, 99_000, "No repaint"),
        ]
        (result,) = summarize(rows)
        self.assertEqual(result["cold_ms"], 91)
        self.assertEqual(result["cold_peak_rss_bytes"], 9_000)
        self.assertEqual(result["warm_attempts"], 21)
        self.assertEqual(result["warm_successes"], 20)
        self.assertEqual(result["warm_failures"], 1)
        self.assertEqual(result["warm_median_ms"], 10)
        self.assertEqual(result["warm_p95_ms"], 19)
        self.assertEqual(result["warm_median_peak_rss_bytes"], 10_000)
        self.assertEqual(result["warm_p95_peak_rss_bytes"], 19_000)

    def test_baseline_compares_same_statistic_and_flags_over_twenty_percent(self):
        fields = {"workload": "hammer", "metric": "apply_refresh"}
        comparisons = compare_baseline([
            {**fields, "build": "candidate", "cold_ms": 80, "warm_p95_ms": 260},
            {**fields, "build": "baseline", "cold_ms": 100, "warm_p95_ms": 200},
        ])
        cold, warm = comparisons
        self.assertEqual((cold["statistic"], cold["baseline_ms"], cold["candidate_ms"]), ("cold", 100, 80))
        self.assertAlmostEqual(cold["change_percent"], -20)
        self.assertFalse(cold["investigate"])
        self.assertEqual((warm["statistic"], warm["baseline_ms"], warm["candidate_ms"]), ("warm_p95", 200, 260))
        self.assertAlmostEqual(warm["change_percent"], 30)
        self.assertTrue(warm["investigate"])

    def test_failed_cold_and_warm_comparisons_leave_missing_values(self):
        (result,) = summarize([
            {"build": "candidate", "workload": "hammer", "metric": "apply_refresh",
             "phase": phase, "duration_ms": None, "peak_rss_bytes": None, "error": "Worker exit"}
            for phase in ("cold", "warm")
        ])
        self.assertEqual(result["cold_error"], "Worker exit")
        self.assertEqual(result["warm_failures"], 1)
        self.assertIsNone(result["warm_median_ms"])
        self.assertIsNone(result["warm_p95_ms"])
        self.assertFalse(result["target"]["met"])
