from vramcalc.core.types import EstimateBreakdown, EstimateRequest, EstimateResult
from vramcalc.core.units import bytes_per_dtype, bytes_to_gib
from vramcalc.hf.resolve import extract_arch_info, load_model_config


class VllmEstimator:
    def estimate(self, req: EstimateRequest) -> EstimateResult:
        cfg = load_model_config(req.model, req.revision)
        arch = extract_arch_info(cfg)

        bpe = bytes_per_dtype(req.dtype)

        # Coarse param estimate for decoder-only transformer
        h = arch["hidden_size"]
        l = arch["num_hidden_layers"]
        v = arch["vocab_size"]
        ff = arch["intermediate_size"]

        params_per_layer = (4 * h * h) + (3 * h * ff)
        embedding_params = v * h
        total_params = (l * params_per_layer) + embedding_params

        weights_gib = bytes_to_gib(total_params * bpe)

        # KV cache estimate: 2 tensors (K,V) x layers x tokens x kv_heads x head_dim x dtype
        n_kv = arch["num_key_value_heads"]
        head_dim = h // arch["num_attention_heads"]
        tokens = req.context_length * req.concurrency
        kv_bytes = 2 * l * tokens * n_kv * head_dim * bpe
        kv_cache_gib = bytes_to_gib(kv_bytes)

        # Heuristic overheads
        activations_gib = 0.08 * (weights_gib + kv_cache_gib)
        runtime_overhead_gib = 1.0

        total = weights_gib + kv_cache_gib + activations_gib + runtime_overhead_gib
        fits = None if req.gpu_memory_gib is None else total <= req.gpu_memory_gib

        assumptions = [
            "Coarse decoder-only transformer parameter estimate",
            "KV cache based on context_length * concurrency",
            "Fixed runtime overhead heuristic for vLLM",
            "Does not yet model tensor/pipeline parallel partitioning",
        ]

        return EstimateResult(
            total_gib=total,
            fits=fits,
            breakdown=EstimateBreakdown(
                weights_gib=weights_gib,
                kv_cache_gib=kv_cache_gib,
                activations_gib=activations_gib,
                runtime_overhead_gib=runtime_overhead_gib,
            ),
            assumptions=assumptions,
        )
