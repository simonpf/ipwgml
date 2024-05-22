"""
Tests for the ipwgml.config module.
"""
from pathlib import Path
import os


from ipwgml.config import get_data_path


def test_get_data_path(tmp_path, monkeypatch):
    """
    Get IPWGML data path and ensure that:
      - It points to the current working directory if no environment variable is set.
      - It points to the path identified by the "IPWGML_PATH" environment variable.
    """
    monkeypatch.delenv("IPWGML_PATH", raising=False)
    path = get_data_path()
    assert path == Path(os.getcwd())

    monkeypatch.setenv("IPWGML_PATH", str(tmp_path))
    path = get_data_path()
    assert path == tmp_path
