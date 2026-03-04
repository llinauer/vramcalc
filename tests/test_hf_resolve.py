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
