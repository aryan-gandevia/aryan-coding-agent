"""Tests for greeter.py."""

from greeter import hello


def test_hello():
    assert hello("World") == "Hello, World!"
    assert hello("Aryan") == "Hello, Aryan!"
