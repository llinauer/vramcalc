import csv
from pathlib import Path

import pytest

from vramcalc.core.types import EstimateRequest
from vramcalc.runtimes.vllm.estimator import VllmEstimator

DATASET_PATH = Path("data/measurements/vllm_measurements.csv")
MAX_ABS_PCT_ERROR = 35.0


def _load_rows() -> list[dict[str, str]]:
    if not DATASET_PATH.exists():
        return []
    with DATASET_PATH.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if (r.get("measured_total_gib") or "").strip()]


def test_vllm_estimator_against_measurements() -> None:
    rows = _load_rows()
    if not rows:
        pytest.skip("No measured rows in data/measurements/vllm_measurements.csv yet")

    estimator = VllmEstimator()
    failures: list[str] = []

    for r in rows:
        arch = {
            "hidden_size": int(r["hidden_size"]),
            "num_hidden_layers": int(r["num_hidden_layers"]),
            "num_attention_heads": int(r["num_attention_heads"]),
            "num_key_value_heads": int(r["num_key_value_heads"]),
            "vocab_size": int(r["vocab_size"]),
            "intermediate_size": int(r["intermediate_size"]),
        }

        req = EstimateRequest(
            runtime="vllm",
            model=r["model"],
            revision=r["revision"] or None,
            dtype=r["dtype"],
            context_length=int(r["context_length"]),
            concurrency=int(r["concurrency"]),
            gpu_memory_gib=float(r["gpu_memory_gib"]) if (r["gpu_memory_gib"] or "").strip() else None,
        )

        est = estimator._estimate_from_arch(req, arch)
        measured = float(r["measured_total_gib"])
        if measured <= 0:
            continue

        abs_pct_error = abs(est.total_gib - measured) / measured * 100.0
        if abs_pct_error > MAX_ABS_PCT_ERROR:
            failures.append(
                f"{r['model']} ctx={r['context_length']} conc={r['concurrency']} "
                f"dtype={r['dtype']}: abs_pct_error={abs_pct_error:.2f}%"
            )

    assert not failures, "\n".join(failures)
