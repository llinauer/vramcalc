from typing import Protocol

from vramcalc.core.types import EstimateRequest, EstimateResult


class RuntimeEstimator(Protocol):
    def estimate(self, req: EstimateRequest) -> EstimateResult:
        ...
