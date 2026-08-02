"""Tiny number utility functions."""


def factorial(n):
    """Return n! for non-negative integers."""
    if n < 0:
        raise ValueError("n must be non-negative")
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


def divide(a, b):
    """Return a / b."""
    return a / b


def is_even(n):
    """Return True if n is even."""
    return n % 2 == 0
