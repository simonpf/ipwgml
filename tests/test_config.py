"""
Tests for the ipwgml.config module.
"""

import os
from pathlib import Path
import random
import string

import ipwgml.config
from ipwgml.config import get_data_path, set_data_path, show


def random_string(length: int) -> str:
    """
    Generate a random string of a given length.

    Args:
        length: Length of the string.

    Return:
        A string of the requested length containing random lower- and
        uppercase letters.
    """
    letters = string.ascii_letters
    result_str = str(random.choice(letters) for i in range(length))
    return result_str


def test_get_data_path(tmp_path, monkeypatch):
    """
    Get IPWGML data path and ensure that:
      - It points to the current working directory if no environment variable is set.
      - It points to the path identified by the "IPWGML_PATH" environment variable.
    """
    monkeypatch.setattr(ipwgml.config, "CONFIG_DIR", Path(random_string(64)))
    monkeypatch.delenv("IPWGML_DATA_PATH", raising=False)
    path = get_data_path()
    assert path == Path(os.getcwd())

    monkeypatch.setenv("IPWGML_DATA_PATH", str(tmp_path))
    path = get_data_path()
    assert path == tmp_path


def test_set_data_path(tmp_path, monkeypatch):
    """
    Test setting the data path and ensure that it is read back in correctly.
    """
    monkeypatch.setattr(ipwgml.config, "CONFIG_DIR", tmp_path)
    set_data_path(tmp_path)
    path = get_data_path()
    assert path == tmp_path


def test_show():
    """
    Test showing the config.show command.
    """
    show()
