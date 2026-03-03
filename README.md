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
