"""
ipwgml.config
=============

Provides functionality to manage the local configuration of the ipwgml package.
"""
from pathlib import Path
import os


def get_data_path() -> Path:
    """
    Get the root of the IPWGML data path.

    Determine the IPWGML data path by checking whether the "IPWGML_PATH" environment variable
    is set. Otherwise returns the current working directory.
    """
    path = os.environ.get("IPWGML_PATH", None)
    if path is None:
        path = Path(os.getcwd())
    else:
        path = Path(path)
    return path
