"""Tests for stats.py."""

from stats import running_mean


def test_running_mean_first_call():
    assert running_mean([1, 2, 3]) == [1.0, 1.5, 2.0]


def test_running_mean_second_call():
    # The function should compute the running mean from scratch each call.
    assert running_mean([10, 20]) == [10.0, 15.0]


def test_running_mean_empty():
    assert running_mean([]) == []
