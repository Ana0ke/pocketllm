"""PocketLLM CLI - Command-line interface."""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from pocketllm import __version__
from pocketllm.pipeline.quantizer import Quantizer
from pocketllm.pipeline.converter import Converter
from pocketllm.pipeline.packager import Packager
from pocketllm.devices.selector import DeviceSelector
from pocketllm.benchmark.runner import BenchmarkRunner

app = typer.Typer(
    name="pocketllm",
    help="🧊 Put any LLM in your pocket. Quantize and deploy to edge devices.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)
console = Console()


def version_callback(value: bool):
    if value:
        console.print(f"PocketLLM v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        None, "--version", "-v", callback=version_callback, is_eager=True,
        help="Show version and exit.",
    ),
):
    """🧊 Put any LLM in your pocket. Quantize and deploy to edge devices."""


@app.command()
def shrink(
    model_id: str = typer.Argument(
        ..., help="HuggingFace model ID (e.g. Qwen/Qwen2.5-1.5B-Instruct)"
    ),
    target: str = typer.Option(
        ..., "--target", "-t", help="Target device (e.g. rpi5, jetson-nano, android)"
    ),
    quant: Optional[str] = typer.Option(
        None, "--quant", "-q", help="Quantization method (int4, int8, gptq, awq)"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output directory (default: ./output/<model>-<target>)"
    ),
    benchmark: bool = typer.Option(
        False, "--benchmark", "-b", help="Run benchmark after packaging"
    ),
):
    """Quantize and package a model for edge deployment."""
    selector = DeviceSelector()

    # Validate target device
    device = selector.get_device(target)
    if device is None:
        console.print(f"[red]Error:[/red] Unknown device '{target}'")
        console.print(f"Run [bold]pocketllm devices[/bold] to see supported devices.")
        raise typer.Exit(1)

    # Auto-select quantization if not specified
    if quant is None:
        quant = device.recommended_quant
        console.print(f"[dim]Auto-selected quantization: {quant}[/dim]")

    # Determine output path
    model_name = model_id.split("/")[-1]
    if output is None:
        output = Path(f"./output/{model_name}-{target}-{quant}")
    output.mkdir(parents=True, exist_ok=True)

    console.print(f"\n[bold blue]PocketLLM[/bold blue] v{__version__}")
    console.print(f"  Model:   [green]{model_id}[/green]")
    console.print(f"  Device:  [green]{device.name}[/green]")
    console.print(f"  Quant:   [green]{quant}[/green]")
    console.print(f"  Output:  [green]{output}[/green]\n")

    # Step 1: Download model
    with console.status("[bold green]Downloading model..."):
        from pocketllm.utils.downloader import download_model
        model_path = download_model(model_id)
    console.print("  ✅ Model downloaded")

    # Step 2: Quantize
    with console.status(f"[bold green]Quantizing ({quant})..."):
        quantizer = Quantizer()
        quantized_path = quantizer.quantize(model_path, quant, device)
    console.print(f"  ✅ Quantized ({quant})")

    # Step 3: Package
    with console.status("[bold green]Packaging for target device..."):
        packager = Packager()
        package_path = packager.package(quantized_path, device, output)
    console.print("  ✅ Packaged")

    # Step 4: Benchmark (optional)
    if benchmark:
        with console.status("[bold green]Running benchmark..."):
            runner = BenchmarkRunner()
            results = runner.run(package_path, device)
            console.print(results.format_table())
    else:
        console.print("\n  💡 Run with [bold]--benchmark[/bold] to test performance.")

    console.print(f"\n[bold green]✨ Done![/bold green] Deployable package: {package_path}")


@app.command()
def devices():
    """List supported target devices."""
    selector = DeviceSelector()
    device_list = selector.list_devices()

    table = Table(title="Supported Devices")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("RAM", style="yellow")
    table.add_column("Arch", style="blue")
    table.add_column("Recommended", style="magenta")

    for d in device_list:
        table.add_row(d.id, d.name, f"{d.ram_gb}GB", d.arch, d.recommended_quant)

    console.print(table)


@app.command()
def quants():
    """List supported quantization methods."""
    table = Table(title="Supported Quantization Methods")
    table.add_column("Method", style="cyan")
    table.add_column("Bits", style="yellow")
    table.add_column("Backend", style="green")
    table.add_column("Description", style="white")

    quants_info = [
        ("int4", "4-bit", "llama.cpp (GGUF)", "Best compression, slight quality loss"),
        ("int8", "8-bit", "llama.cpp (GGUF)", "Good balance of size and quality"),
        ("f16", "16-bit", "llama.cpp (GGUF)", "No quality loss, larger size"),
        ("gptq", "4-bit", "auto-gptq", "Post-training quantization, GPU optimized"),
        ("awq", "4-bit", "autoawq", "Activation-aware, better quality at 4-bit"),
        ("mnn-int4", "4-bit", "MNN", "Mobile optimized, Android/iOS"),
        ("mnn-int8", "8-bit", "MNN", "Mobile optimized, better quality"),
    ]

    for q in quants_info:
        table.add_row(*q)

    console.print(table)


@app.command()
def benchmark(
    package_path: Path = typer.Argument(
        ..., help="Path to the packaged model directory"
    ),
    prompt: str = typer.Option(
        "Hello, how are you?", "--prompt", "-p", help="Test prompt"
    ),
    max_tokens: int = typer.Option(
        128, "--max-tokens", "-m", help="Max tokens to generate"
    ),
):
    """Benchmark a packaged model."""
    if not package_path.exists():
        console.print(f"[red]Error:[/red] Package not found: {package_path}")
        raise typer.Exit(1)

    runner = BenchmarkRunner()
    with console.status("[bold green]Running benchmark..."):
        results = runner.run(package_path, prompt=prompt, max_tokens=max_tokens)

    console.print(results.format_table())


# Allow `pocketllm <model_id>` as a shorthand for `pocketllm shrink`
@app.command(hidden=True)
def default(
    model_id: str = typer.Argument(...),
):
    """Default command - shorthand for shrink."""
    console.print(f"[dim]Tip: Use [bold]pocketllm shrink {model_id} --target <device>[/bold][/dim]")


if __name__ == "__main__":
    app()
