from __future__ import annotations

import argparse
import csv
import datetime as dt
import shlex
import subprocess
import time
from pathlib import Path

from vramcalc.core.types import EstimateRequest
from vramcalc.hf.resolve import extract_arch_info, load_model_config
from vramcalc.runtimes.vllm.estimator import VllmEstimator


def _query_gpu_field(field: str, gpu_index: int) -> str:
    cmd = [
        "nvidia-smi",
        f"--query-gpu={field}",
        "--format=csv,noheader,nounits",
        "-i",
        str(gpu_index),
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    return out.splitlines()[0].strip()


def _sample_used_gib(gpu_index: int) -> float:
    mib = float(_query_gpu_field("memory.used", gpu_index))
    return mib / 1024.0


def _gpu_name(gpu_index: int) -> str:
    return _query_gpu_field("name", gpu_index)


def _run_and_measure_peak(workload_cmd: str, gpu_index: int, interval_s: float) -> tuple[float, float]:
    baseline = _sample_used_gib(gpu_index)
    proc = subprocess.Popen(shlex.split(workload_cmd))

    peak = baseline
    while proc.poll() is None:
        try:
            used = _sample_used_gib(gpu_index)
            peak = max(peak, used)
        except Exception:
            pass
        time.sleep(interval_s)

    # one final sample
    try:
        used = _sample_used_gib(gpu_index)
        peak = max(peak, used)
    except Exception:
        pass

    return baseline, peak


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect vLLM VRAM measurements and append CSV rows")
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--gpu-memory-gib", type=float, default=None)
    parser.add_argument("--workload-cmd", default=None, help="Command to run while sampling nvidia-smi")
    parser.add_argument("--measured-total-gib", type=float, default=None, help="Manual measured total GiB")
    parser.add_argument("--interval-s", type=float, default=0.5)
    parser.add_argument("--notes", default="")
    parser.add_argument(
        "--out",
        default="data/measurements/vllm_measurements.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    if args.workload_cmd is None and args.measured_total_gib is None:
        raise ValueError("Provide either --workload-cmd or --measured-total-gib")

    req = EstimateRequest(
        runtime="vllm",
        model=args.model,
        revision=args.revision,
        dtype=args.dtype,
        context_length=args.context_length,
        concurrency=args.concurrency,
        gpu_memory_gib=args.gpu_memory_gib,
    )

    cfg = load_model_config(args.model, args.revision)
    arch = extract_arch_info(cfg)
    est = VllmEstimator()._estimate_from_arch(req, arch)

    gpu_name = _gpu_name(args.gpu_index)

    baseline = None
    peak = None
    measured_total = args.measured_total_gib

    if args.workload_cmd is not None:
        baseline, peak = _run_and_measure_peak(args.workload_cmd, args.gpu_index, args.interval_s)
        measured_total = peak

    if measured_total is None:
        raise RuntimeError("Could not determine measured_total_gib")

    error_gib = est.total_gib - measured_total
    error_pct = (error_gib / measured_total) * 100.0 if measured_total > 0 else 0.0

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "timestamp": dt.datetime.utcnow().isoformat() + "Z",
        "model": args.model,
        "revision": args.revision or "",
        "dtype": args.dtype,
        "context_length": args.context_length,
        "concurrency": args.concurrency,
        "gpu_name": gpu_name,
        "gpu_index": args.gpu_index,
        "gpu_memory_gib": args.gpu_memory_gib if args.gpu_memory_gib is not None else "",
        "hidden_size": arch["hidden_size"],
        "num_hidden_layers": arch["num_hidden_layers"],
        "num_attention_heads": arch["num_attention_heads"],
        "num_key_value_heads": arch["num_key_value_heads"],
        "vocab_size": arch["vocab_size"],
        "intermediate_size": arch["intermediate_size"],
        "baseline_used_gib": "" if baseline is None else f"{baseline:.6f}",
        "peak_used_gib": "" if peak is None else f"{peak:.6f}",
        "measured_total_gib": f"{measured_total:.6f}",
        "estimated_total_gib": f"{est.total_gib:.6f}",
        "error_gib": f"{error_gib:.6f}",
        "error_pct": f"{error_pct:.3f}",
        "notes": args.notes,
    }

    write_header = not out_path.exists() or out_path.stat().st_size == 0
    with out_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)

    print(f"Saved measurement row to {out_path}")
    print(f"Measured={measured_total:.3f} GiB Estimated={est.total_gib:.3f} GiB Error={error_pct:.2f}%")


if __name__ == "__main__":
    main()
