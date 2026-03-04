from vramcalc.hf.resolve import extract_arch_info


def test_extract_arch_info_infers_default_context_length_from_nested_config() -> None:
    cfg = {
        "model": {
            "hidden_size": 1024,
            "num_hidden_layers": 8,
            "num_attention_heads": 16,
            "vocab_size": 32000,
            "intermediate_size": 4096,
            "max_position_embeddings": 8192,
        }
    }

    arch = extract_arch_info(cfg)
    assert arch["default_context_length"] == 8192


def test_extract_arch_info_infers_quantization_from_config() -> None:
    cfg = {
        "hidden_size": 1024,
        "num_hidden_layers": 8,
        "num_attention_heads": 16,
        "vocab_size": 32000,
        "intermediate_size": 4096,
        "quantization_config": {
            "quant_method": "awq",
            "bits": 4,
        },
    }

    arch = extract_arch_info(cfg)
    assert arch["quantization"] == "awq-4bit"
    assert arch["quantization_source"] == "config"
    assert arch["quantization_confidence"] == "high"


def test_extract_arch_info_infers_quantization_from_model_name_when_no_config(monkeypatch):
    monkeypatch.setattr("vramcalc.hf.resolve.list_repo_files", lambda repo_id, revision=None: [])

    cfg = {
        "hidden_size": 1024,
        "num_hidden_layers": 8,
        "num_attention_heads": 16,
        "vocab_size": 32000,
        "intermediate_size": 4096,
    }

    arch = extract_arch_info(cfg, model="some-org/cool-model-GPTQ")
    assert arch["quantization"] == "gptq"
    assert arch["quantization_source"] == "name"
    assert arch["quantization_confidence"] == "low"
