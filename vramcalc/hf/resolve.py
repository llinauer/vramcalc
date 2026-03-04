import json
from collections.abc import Mapping

from huggingface_hub import hf_hub_download, list_repo_files


REQUIRED_FIELDS = ["hidden_size", "num_hidden_layers", "num_attention_heads", "vocab_size"]
CONTEXT_LENGTH_KEYS = [
    "max_position_embeddings",
    "max_sequence_length",
    "n_positions",
    "seq_length",
    "model_max_length",
]


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


def _infer_quantization_from_config(flat_cfg: dict[str, object]) -> tuple[str | None, str | None, str | None]:
    qcfg = _get_value(flat_cfg, "quantization_config")
    if isinstance(qcfg, Mapping):
        q_method = str(qcfg.get("quant_method", "")).lower()
        bits = qcfg.get("bits")
        if q_method:
            if bits is not None:
                return f"{q_method}-{bits}bit", "config", "high"
            return q_method, "config", "high"

        if qcfg.get("load_in_4bit"):
            return "bnb-4bit", "config", "high"
        if qcfg.get("load_in_8bit"):
            return "bnb-8bit", "config", "high"

    return None, None, None


def _infer_quantization_from_strings(values: list[str], source: str) -> tuple[str | None, str | None, str | None]:
    s = " ".join(values).lower()
    rules = [
        ("awq", "awq"),
        ("gptq", "gptq"),
        ("gguf", "gguf"),
        ("bnb-4bit", "bnb-4bit"),
        ("4bit", "4bit"),
        ("int4", "int4"),
        ("q4", "q4"),
        ("8bit", "8bit"),
        ("int8", "int8"),
        ("q8", "q8"),
        ("fp8", "fp8"),
    ]
    for needle, label in rules:
        if needle in s:
            confidence = "medium" if source == "files" else "low"
            return label, source, confidence
    return None, None, None


def infer_quantization(model: str | None, revision: str | None, flat_cfg: dict[str, object]) -> dict[str, str]:
    q, src, conf = _infer_quantization_from_config(flat_cfg)
    if q:
        return {"quantization": q, "quantization_source": src, "quantization_confidence": conf}

    if model:
        try:
            repo_files = list_repo_files(repo_id=model, revision=revision)
            q, src, conf = _infer_quantization_from_strings(repo_files, "files")
            if q:
                return {"quantization": q, "quantization_source": src, "quantization_confidence": conf}
        except Exception:
            pass

        q, src, conf = _infer_quantization_from_strings([model], "name")
        if q:
            return {"quantization": q, "quantization_source": src, "quantization_confidence": conf}

    return {"quantization": "unknown", "quantization_source": "unknown", "quantization_confidence": "low"}


def load_model_config(model: str, revision: str | None = None) -> dict:
    config_path = hf_hub_download(repo_id=model, filename="config.json", revision=revision)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg


def extract_arch_info(cfg: dict, model: str | None = None, revision: str | None = None) -> dict:
    flat_cfg = _flatten_dict(cfg)

    missing = [k for k in REQUIRED_FIELDS if _get_value(flat_cfg, k) is None]
    if missing:
        raise ValueError(f"Missing required config fields: {missing}")

    hidden_size = int(_get_value(flat_cfg, "hidden_size"))
    num_attention_heads = int(_get_value(flat_cfg, "num_attention_heads"))

    n_kv_heads_raw = _get_value(flat_cfg, "num_key_value_heads", num_attention_heads)
    intermediate_size_raw = _get_value(flat_cfg, "intermediate_size", 4 * hidden_size)

    default_context_length = None
    for key in CONTEXT_LENGTH_KEYS:
        value = _get_value(flat_cfg, key)
        if value is not None:
            try:
                default_context_length = int(value)
                break
            except (TypeError, ValueError):
                continue
    if default_context_length is None:
        default_context_length = 4096

    q_info = infer_quantization(model=model, revision=revision, flat_cfg=flat_cfg)

    return {
        "hidden_size": hidden_size,
        "num_hidden_layers": int(_get_value(flat_cfg, "num_hidden_layers")),
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": int(n_kv_heads_raw),
        "vocab_size": int(_get_value(flat_cfg, "vocab_size")),
        "intermediate_size": int(intermediate_size_raw),
        "model_type": str(_get_value(flat_cfg, "model_type", "unknown")),
        "default_context_length": default_context_length,
        **q_info,
    }
