# vramcalc

Estimate inference VRAM usage from a Hugging Face model id.

## Quickstart (uv)

```bash
uv sync
uv run vramcalc estimate \
  --runtime vllm \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dtype bf16 \
  --context-length 4096 \
  --concurrency 1 \
  --gpu-memory-gib 24
```

## Notes

- MVP runtime: `vllm`
- Input is HF model id first; no manual param count required
- Current estimates are coarse and report assumptions explicitly

## Collect real vLLM measurements

Template dataset lives at:

- `data/measurements/vllm_measurements.csv`

Add a measured row (manual measurement):

```bash
uv run python scripts/collect_vllm_measurements.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dtype bf16 \
  --context-length 4096 \
  --concurrency 1 \
  --gpu-index 0 \
  --gpu-memory-gib 24 \
  --measured-total-gib 18.2
```

Or sample peak usage while running a workload command:

```bash
uv run python scripts/collect_vllm_measurements.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dtype bf16 \
  --context-length 4096 \
  --concurrency 1 \
  --gpu-index 0 \
  --workload-cmd "python your_vllm_workload.py"
```

Regression test:

```bash
uv run pytest tests/test_vllm_regression_dataset.py
```
