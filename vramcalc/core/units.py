BYTES_PER_GIB = 1024**3


def bytes_per_dtype(dtype: str) -> float:
    d = dtype.lower()
    table = {
        "fp32": 4,
        "float32": 4,
        "fp16": 2,
        "float16": 2,
        "bf16": 2,
        "fp8": 1,
        "int8": 1,
        "int4": 0.5,
    }
    if d not in table:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return table[d]


def bytes_to_gib(n_bytes: float) -> float:
    return n_bytes / BYTES_PER_GIB
