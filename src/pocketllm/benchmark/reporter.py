"""Benchmark reporter - Formats and displays benchmark results."""

from dataclasses import dataclass

from rich.console import Console
from rich.table import Table


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""

    model: str
    device: str
    avg_latency_ms: float
    avg_tokens: float
    tokens_per_second: float
    peak_memory_mb: float
    threads: int
    runs: int

    def format_table(self) -> Table:
        """Format results as a Rich table."""
        table = Table(title="Benchmark Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Model", self.model)
        table.add_row("Device", self.device)
        table.add_row("Avg Latency", f"{self.avg_latency_ms:.1f} ms")
        table.add_row("Avg Tokens", f"{self.avg_tokens:.1f}")
        table.add_row("Tokens/sec", f"{self.tokens_per_second:.2f}")
        table.add_row("Peak Memory", f"{self.peak_memory_mb:.1f} MB")
        table.add_row("Threads", str(self.threads))
        table.add_row("Runs", str(self.runs))

        return table

    def format_markdown(self) -> str:
        """Format results as Markdown."""
        return f"""# Benchmark Results

| Metric | Value |
|--------|-------|
| Model | {self.model} |
| Device | {self.device} |
| Avg Latency | {self.avg_latency_ms:.1f} ms |
| Avg Tokens | {self.avg_tokens:.1f} |
| Tokens/sec | {self.tokens_per_second:.2f} |
| Peak Memory | {self.peak_memory_mb:.1f} MB |
| Threads | {self.threads} |
| Runs | {self.runs} |
"""

    def format_json(self) -> dict:
        """Format results as a JSON-serializable dict."""
        return {
            "model": self.model,
            "device": self.device,
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "avg_tokens": round(self.avg_tokens, 1),
            "tokens_per_second": round(self.tokens_per_second, 2),
            "peak_memory_mb": round(self.peak_memory_mb, 1),
            "threads": self.threads,
            "runs": self.runs,
        }
