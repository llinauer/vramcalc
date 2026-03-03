from vramcalc.runtimes.vllm.estimator import VllmEstimator


def get_estimator(runtime: str):
    r = runtime.lower()
    if r == "vllm":
        return VllmEstimator()
    raise ValueError(f"Unsupported runtime: {runtime}")
