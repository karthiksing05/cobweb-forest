"""Assert every executable check of the theory (verification/checks.py).  Slow (~1-3 min)."""
import pytest

from verification import checks as C


@pytest.mark.parametrize("check", C.ALL, ids=[c.__name__ for c in C.ALL])
def test_statement(check):
    r = check()
    assert r["passed"], r
