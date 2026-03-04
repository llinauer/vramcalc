import pytest

from vramcalc.core.types import EstimateRequest
from vramcalc.core.units import bytes_to_gib
from vramcalc.runtimes.vllm.estimator import VllmEstimator


@pytest.fixture
def fake_arch() -> dict:
    return {
        "hidden_size": 1024,
        "num_hidden_layers": 12,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "vocab_size": 32000,
        "intermediate_size": 4096,
        "default_context_length": 8192,
    }


def _mock_hf(monkeypatch: pytest.MonkeyPatch, arch: dict) -> None:
    monkeypatch.setattr(
        "vramcalc.runtimes.vllm.estimator.load_model_config",
        lambda model, revision=None: {"_dummy": True},
    )
    monkeypatch.setattr(
        "vramcalc.runtimes.vllm.estimator.extract_arch_info",
        lambda cfg, model=None, revision=None: arch,
    )


def test_vllm_estimate_breakdown_matches_formula(monkeypatch: pytest.MonkeyPatch, fake_arch: dict) -> None:
    _mock_hf(monkeypatch, fake_arch)

    req = EstimateRequest(
        runtime="vllm",
        model="dummy/model",
        dtype="bf16",
        context_length=2048,
        concurrency=2,
        gpu_memory_gib=80,
    )

    result = VllmEstimator().estimate(req)

    h = fake_arch["hidden_size"]
    l = fake_arch["num_hidden_layers"]
    v = fake_arch["vocab_size"]
    ff = fake_arch["intermediate_size"]
    n_kv = fake_arch["num_key_value_heads"]
    head_dim = h // fake_arch["num_attention_heads"]
    bpe = 2

    params_per_layer = (4 * h * h) + (3 * h * ff)
    total_params = (l * params_per_layer) + (v * h)
    expected_weights = bytes_to_gib(total_params * bpe)

    tokens = req.context_length * req.concurrency
    expected_kv = bytes_to_gib(2 * l * tokens * n_kv * head_dim * bpe)
    expected_act = 0.08 * (expected_weights + expected_kv)
    expected_total = expected_weights + expected_kv + expected_act + 1.0

    assert result.assumed_context_length == 2048
    assert result.weight_bytes_per_param == 2
    assert result.kv_bytes_per_element == 2
    assert result.breakdown.weights_gib == pytest.approx(expected_weights)
    assert result.breakdown.kv_cache_gib == pytest.approx(expected_kv)
    assert result.breakdown.activations_gib == pytest.approx(expected_act)
    assert result.total_gib == pytest.approx(expected_total)


def test_vllm_fit_flag(monkeypatch: pytest.MonkeyPatch, fake_arch: dict) -> None:
    _mock_hf(monkeypatch, fake_arch)

    base_req = EstimateRequest(runtime="vllm", model="dummy/model", dtype="bf16")
    result_no_limit = VllmEstimator().estimate(base_req)
    assert result_no_limit.fits is None

    low_mem_req = base_req.model_copy(update={"gpu_memory_gib": 1.0})
    high_mem_req = base_req.model_copy(update={"gpu_memory_gib": 1000.0})

    assert VllmEstimator().estimate(low_mem_req).fits is False
    assert VllmEstimator().estimate(high_mem_req).fits is True


def test_kv_cache_scales_with_concurrency(monkeypatch: pytest.MonkeyPatch, fake_arch: dict) -> None:
    _mock_hf(monkeypatch, fake_arch)

    req1 = EstimateRequest(runtime="vllm", model="dummy/model", context_length=1024, concurrency=1)
    req2 = EstimateRequest(runtime="vllm", model="dummy/model", context_length=1024, concurrency=4)

    r1 = VllmEstimator().estimate(req1)
    r2 = VllmEstimator().estimate(req2)

    assert r2.breakdown.kv_cache_gib == pytest.approx(r1.breakdown.kv_cache_gib * 4)


def test_uses_model_default_context_length_when_not_provided(
    monkeypatch: pytest.MonkeyPatch, fake_arch: dict
) -> None:
    _mock_hf(monkeypatch, fake_arch)

    req = EstimateRequest(runtime="vllm", model="dummy/model", context_length=None, concurrency=1)
    result = VllmEstimator().estimate(req)

    assert result.assumed_context_length == 8192


def test_quantization_changes_weight_precision_only(monkeypatch: pytest.MonkeyPatch, fake_arch: dict) -> None:
    quant_arch = dict(fake_arch)
    quant_arch.update(
        {
            "quantization": "awq-4bit",
            "quantization_source": "config",
            "quantization_confidence": "high",
        }
    )
    _mock_hf(monkeypatch, quant_arch)

    req = EstimateRequest(runtime="vllm", model="dummy/model", dtype="bf16", context_length=1024)
    result = VllmEstimator().estimate(req)

    assert result.weight_bytes_per_param == 0.5
    assert result.kv_bytes_per_element == 2
