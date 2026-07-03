from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from analysis_models import BenchmarkResult, BenchmarkComparison
from tokenizer_engine import TokenizerEngine
from tests.test_analysis_models import make_word_engine


class BenchmarkTests(unittest.TestCase):
    def test_benchmark_empty_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            engine = make_word_engine(Path(tmp) / "tok")
            res_dict = engine.benchmark("", iterations=5)

            self.assertEqual(res_dict["input_chars"], 0)
            self.assertEqual(res_dict["output_tokens"], 0)
            self.assertEqual(res_dict["total_seconds"], 0.0)

            # Constructing a BenchmarkResult from the dict
            res = BenchmarkResult(**res_dict)
            self.assertEqual(res.input_chars, 0)
            self.assertEqual(res.tokens_per_second, 0.0)

    def test_benchmark_non_empty_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            engine = make_word_engine(Path(tmp) / "tok")
            text = "hello world hello world alpha beta"
            res_dict = engine.benchmark(text, iterations=3)

            self.assertEqual(res_dict["input_chars"], len(text))
            self.assertGreater(res_dict["output_tokens"], 0)
            self.assertEqual(res_dict["iterations"], 3)
            self.assertGreater(res_dict["total_seconds"], 0.0)

            res = BenchmarkResult(**res_dict)
            self.assertEqual(res.input_chars, len(text))
            self.assertGreater(res.tokens_per_second, 0.0)

    def test_benchmark_comparison(self) -> None:
        primary_res = BenchmarkResult(
            source="primary",
            tokenizer_name="gpt2",
            input_chars=100,
            output_tokens=20,
            iterations=10,
            total_seconds=0.1,
            mean_seconds=0.01,
            tokens_per_second=200.0,
            chars_per_second=1000.0,
        )

        compare_res = BenchmarkResult(
            source="compare",
            tokenizer_name="llama",
            input_chars=100,
            output_tokens=15,
            iterations=10,
            total_seconds=0.2,
            mean_seconds=0.02,
            tokens_per_second=75.0,
            chars_per_second=500.0,
        )

        speedup = primary_res.tokens_per_second / compare_res.tokens_per_second if compare_res.tokens_per_second > 0 else None
        comparison = BenchmarkComparison(
            primary=primary_res,
            compare=compare_res,
            speedup=speedup,
        )

        self.assertEqual(comparison.primary.tokenizer_name, "gpt2")
        self.assertEqual(comparison.compare.tokenizer_name, "llama")
        self.assertAlmostEqual(comparison.speedup, 2.6666666666666665)


if __name__ == "__main__":
    unittest.main()
