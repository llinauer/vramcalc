from pydantic import BaseModel, Field


class EstimateRequest(BaseModel):
    runtime: str = Field(default="vllm")
    model: str
    revision: str | None = None
    dtype: str = Field(default="bf16")
    context_length: int | None = Field(default=None, ge=1)
    concurrency: int = Field(default=1, ge=1)
    gpu_memory_gib: float | None = Field(default=None, gt=0)


class EstimateBreakdown(BaseModel):
    weights_gib: float
    kv_cache_gib: float
    activations_gib: float
    runtime_overhead_gib: float


class EstimateResult(BaseModel):
    total_gib: float
    fits: bool | None
    assumed_context_length: int
    inferred_quantization: str
    quantization_source: str
    quantization_confidence: str
    breakdown: EstimateBreakdown
    assumptions: list[str]
