import json

import typer
from rich.console import Console
from rich.table import Table

from vramcalc.core.types import EstimateRequest
from vramcalc.runtimes.registry import get_estimator

app = typer.Typer(help="VRAM calculator for inference runtimes")
console = Console()


@app.command()
def estimate(
    runtime: str = typer.Option("vllm", help="Inference runtime"),
    model: str = typer.Option(..., help="Hugging Face model id"),
    revision: str | None = typer.Option(None, help="HF revision"),
    dtype: str = typer.Option("bf16", help="Weight/KV dtype"),
    context_length: int | None = typer.Option(None, help="Max context length (defaults to model config)"),
    concurrency: int = typer.Option(1, help="Concurrent sequences"),
    gpu_memory_gib: float | None = typer.Option(None, help="GPU memory in GiB for fit check"),
    as_json: bool = typer.Option(False, "--json", help="Output JSON"),
) -> None:
    req = EstimateRequest(
        runtime=runtime,
        model=model,
        revision=revision,
        dtype=dtype,
        context_length=context_length,
        concurrency=concurrency,
        gpu_memory_gib=gpu_memory_gib,
    )

    estimator = get_estimator(runtime)
    result = estimator.estimate(req)

    if as_json:
        console.print_json(data=json.loads(result.model_dump_json()))
        return

    table = Table(title=f"vramcalc: {model} ({runtime})")
    table.add_column("Component")
    table.add_column("GiB", justify="right")
    table.add_row("Assumed context length", str(result.assumed_context_length))
    table.add_row(
        "Inferred quantization",
        f"{result.inferred_quantization} ({result.quantization_source}, {result.quantization_confidence})",
    )
    table.add_row("Weights", f"{result.breakdown.weights_gib:.3f}")
    table.add_row("KV cache", f"{result.breakdown.kv_cache_gib:.3f}")
    table.add_row("Activations", f"{result.breakdown.activations_gib:.3f}")
    table.add_row("Runtime overhead", f"{result.breakdown.runtime_overhead_gib:.3f}")
    table.add_row("Total", f"{result.total_gib:.3f}")
    table.add_row("Fits", str(result.fits))
    console.print(table)

    console.print("Assumptions:")
    for a in result.assumptions:
        console.print(f"- {a}")


if __name__ == "__main__":
    app()
