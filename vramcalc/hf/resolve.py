import json
from huggingface_hub import hf_hub_download


REQUIRED_FIELDS = ["hidden_size", "num_hidden_layers", "num_attention_heads", "vocab_size"]


def load_model_config(model: str, revision: str | None = None) -> dict:
    config_path = hf_hub_download(repo_id=model, filename="config.json", revision=revision)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg


def extract_arch_info(cfg: dict) -> dict:
    missing = [k for k in REQUIRED_FIELDS if k not in cfg]
    if missing:
        raise ValueError(f"Missing required config fields: {missing}")

    n_kv_heads = cfg.get("num_key_value_heads", cfg["num_attention_heads"])
    return {
        "hidden_size": int(cfg["hidden_size"]),
        "num_hidden_layers": int(cfg["num_hidden_layers"]),
        "num_attention_heads": int(cfg["num_attention_heads"]),
        "num_key_value_heads": int(n_kv_heads),
        "vocab_size": int(cfg["vocab_size"]),
        "intermediate_size": int(cfg.get("intermediate_size", 4 * int(cfg["hidden_size"]))),
        "model_type": str(cfg.get("model_type", "unknown")),
    }
