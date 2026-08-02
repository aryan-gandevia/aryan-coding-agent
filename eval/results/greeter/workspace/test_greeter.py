"""Tests for greeter.py."""

from greeter import goodbye, hello


def test_hello():
    assert hello("World") == "Hello, World!"
    assert hello("Aryan") == "Hello, Aryan!"


def test_goodbye():
    assert goodbye("World") == "Goodbye, World!"
    assert goodbye("Aryan") == "Goodbye, Aryan!"
