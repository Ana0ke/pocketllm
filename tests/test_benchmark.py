"""Tests for benchmark reporter."""

from pocketllm.benchmark.reporter import BenchmarkResult


class TestBenchmarkResult:
    """Tests for BenchmarkResult formatting."""

    def test_format_markdown(self):
        result = BenchmarkResult(
            model="test-model.gguf",
            device="rpi5",
            avg_latency_ms=1500.0,
            avg_tokens=50.0,
            tokens_per_second=33.33,
            peak_memory_mb=2048.0,
            threads=4,
            runs=3,
        )
        md = result.format_markdown()
        assert "test-model.gguf" in md
        assert "33.33" in md

    def test_format_json(self):
        result = BenchmarkResult(
            model="test-model.gguf",
            device="rpi5",
            avg_latency_ms=1500.0,
            avg_tokens=50.0,
            tokens_per_second=33.33,
            peak_memory_mb=2048.0,
            threads=4,
            runs=3,
        )
        json_data = result.format_json()
        assert json_data["model"] == "test-model.gguf"
        assert json_data["tokens_per_second"] == 33.33
