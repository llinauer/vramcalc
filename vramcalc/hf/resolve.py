import json
from collections.abc import Mapping

from huggingface_hub import hf_hub_download


REQUIRED_FIELDS = ["hidden_size", "num_hidden_layers", "num_attention_heads", "vocab_size"]


def _flatten_dict(d: Mapping, parent_key: str = "", sep: str = ".") -> dict[str, object]:
    flat: dict[str, object] = {}
    for k, v in d.items():
        key = f"{parent_key}{sep}{k}" if parent_key else str(k)
        if isinstance(v, Mapping):
            flat.update(_flatten_dict(v, key, sep=sep))
        else:
            flat[key] = v
    return flat


def _get_value(flat_cfg: dict[str, object], key: str, default: object = None) -> object:
    if key in flat_cfg:
        return flat_cfg[key]

    # fallback: match by leaf key, e.g. "model.hidden_size" -> "hidden_size"
    suffix = f".{key}"
    for k, v in flat_cfg.items():
        if k.endswith(suffix):
            return v

    return default


def load_model_config(model: str, revision: str | None = None) -> dict:
    config_path = hf_hub_download(repo_id=model, filename="config.json", revision=revision)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg


def extract_arch_info(cfg: dict) -> dict:
    flat_cfg = _flatten_dict(cfg)

    missing = [k for k in REQUIRED_FIELDS if _get_value(flat_cfg, k) is None]
    if missing:
        raise ValueError(f"Missing required config fields: {missing}")

    hidden_size = int(_get_value(flat_cfg, "hidden_size"))
    num_attention_heads = int(_get_value(flat_cfg, "num_attention_heads"))

    n_kv_heads_raw = _get_value(flat_cfg, "num_key_value_heads", num_attention_heads)
    intermediate_size_raw = _get_value(flat_cfg, "intermediate_size", 4 * hidden_size)

    return {
        "hidden_size": hidden_size,
        "num_hidden_layers": int(_get_value(flat_cfg, "num_hidden_layers")),
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": int(n_kv_heads_raw),
        "vocab_size": int(_get_value(flat_cfg, "vocab_size")),
        "intermediate_size": int(intermediate_size_raw),
        "model_type": str(_get_value(flat_cfg, "model_type", "unknown")),
    }
