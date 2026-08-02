"""Tests for utils.py."""

import pytest

from utils import divide, factorial, is_even


def test_factorial():
    assert factorial(0) == 1
    assert factorial(5) == 120


def test_divide():
    assert divide(10, 2) == 5
    assert divide(1, 4) == 0.25


def test_divide_by_zero():
    with pytest.raises(ValueError):
        divide(1, 0)


def test_is_even():
    assert is_even(4) is True
    assert is_even(5) is False
