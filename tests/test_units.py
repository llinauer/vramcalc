import pytest

from vramcalc.core.units import bytes_per_dtype


def test_bytes_per_dtype_known() -> None:
    assert bytes_per_dtype("bf16") == 2
    assert bytes_per_dtype("fp8") == 1
    assert bytes_per_dtype("int4") == 0.5


def test_bytes_per_dtype_unknown() -> None:
    with pytest.raises(ValueError):
        bytes_per_dtype("weird")
