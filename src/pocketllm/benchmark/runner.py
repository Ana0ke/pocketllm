"""Benchmark runner - Executes inference benchmarks on quantized models."""

import logging
import subprocess
import time
from pathlib import Path
from typing import Optional

from pocketllm.benchmark.reporter import BenchmarkResult

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Runs inference benchmarks on packaged models."""

    def run(
        self,
        package_path: Path,
        device=None,
        prompt: str = "Hello, how are you?",
        max_tokens: int = 128,
        warmup_runs: int = 1,
        benchmark_runs: int = 3,
    ) -> BenchmarkResult:
        """Run benchmark on a packaged model.

        Args:
            package_path: Path to the deployment package.
            device: Target device profile.
            prompt: Test prompt for inference.
            max_tokens: Maximum tokens to generate.
            warmup_runs: Number of warmup runs before measuring.
            benchmark_runs: Number of runs to average.

        Returns:
            BenchmarkResult with performance metrics.
        """
        model_file = self._find_model_file(package_path)
        if model_file is None:
            raise FileNotFoundError(f"No model file found in {package_path}")

        threads = device.cpu_cores if device and hasattr(device, "cpu_cores") else 4

        # Warmup
        for _ in range(warmup_runs):
            self._run_inference(model_file, prompt, max_tokens, threads)

        # Benchmark runs
        latencies = []
        tokens_generated = []

        for i in range(benchmark_runs):
            result = self._run_inference(model_file, prompt, max_tokens, threads)
            latencies.append(result["latency_ms"])
            tokens_generated.append(result["tokens"])

        avg_latency = sum(latencies) / len(latencies)
        avg_tokens = sum(tokens_generated) / len(tokens_generated)
        tokens_per_second = avg_tokens / (avg_latency / 1000) if avg_latency > 0 else 0

        # Measure memory usage
        memory_mb = self._measure_memory(model_file, prompt, max_tokens, threads)

        return BenchmarkResult(
            model=str(model_file),
            device=device.name if device and hasattr(device, "name") else "unknown",
            avg_latency_ms=avg_latency,
            avg_tokens=avg_tokens,
            tokens_per_second=tokens_per_second,
            peak_memory_mb=memory_mb,
            threads=threads,
            runs=benchmark_runs,
        )

    def _find_model_file(self, package_path: Path) -> Optional[Path]:
        """Find the model file in a package directory."""
        if package_path.is_file() and package_path.suffix == ".gguf":
            return package_path

        # Search in package directory
        for pattern in ["*.gguf", "model.gguf", "**/*.gguf"]:
            matches = list(package_path.glob(pattern))
            if matches:
                return matches[0]

        return None

    def _run_inference(
        self,
        model_file: Path,
        prompt: str,
        max_tokens: int,
        threads: int,
    ) -> dict:
        """Run a single inference and measure performance.

        Returns:
            Dict with 'latency_ms' and 'tokens' keys.
        """
        cmd = [
            "llama-cli",
            "-m", str(model_file),
            "-t", str(threads),
            "-c", "2048",
            "-n", str(max_tokens),
            "-p", prompt,
            "--log-disable",
        ]

        start_time = time.time()
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300
            )
        except subprocess.TimeoutExpired:
            logger.warning("Inference timed out")
            return {"latency_ms": 300000, "tokens": 0}
        except FileNotFoundError:
            logger.error("llama-cli not found. Install llama.cpp first.")
            return {"latency_ms": 0, "tokens": 0}

        elapsed_ms = (time.time() - start_time) * 1000

        # Estimate tokens from output (rough)
        output_text = result.stdout
        token_count = len(output_text.split()) if output_text else 0

        return {"latency_ms": elapsed_ms, "tokens": token_count}

    def _measure_memory(
        self, model_file: Path, prompt: str, max_tokens: int, threads: int
    ) -> float:
        """Estimate peak memory usage during inference.

        Returns:
            Peak memory in MB.
        """
        try:
            import psutil
        except ImportError:
            return 0.0

        # Run inference in a subprocess and track memory
        cmd = [
            "llama-cli",
            "-m", str(model_file),
            "-t", str(threads),
            "-c", "2048",
            "-n", str(max_tokens),
            "-p", prompt,
            "--log-disable",
        ]

        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            process = psutil.Process(proc.pid)
            peak_memory = 0.0

            while proc.poll() is None:
                try:
                    mem_info = process.memory_info()
                    peak_memory = max(peak_memory, mem_info.rss / 1024 / 1024)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    break
                time.sleep(0.1)

            proc.wait(timeout=60)
            return peak_memory
        except Exception as e:
            logger.error(f"Memory measurement failed: {e}")
            return 0.0
