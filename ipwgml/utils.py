"""
ipwgml.utils
============

Defines helper functions used throught the ipwgml package.
"""
from pathlib import Path
from contextlib import contextmanager

import xarray as xr


@contextmanager
def open_if_required(path_or_dataset: str | Path | xr.Dataset) -> xr.Dataset:
    """
    Open and close an xarray.Dataset or do nothing if data is already loaded.

    Args:
         path_or_dataset: A Path pointing to a NetCDF4 to open of an already
             loaded dataset.

    Return:
         An xarray.Dataset providing access to the loaded data.
    """
    if isinstance(path_or_dataset, (str, Path)):
        handle = xr.open_dataset(path_or_dataset)
        path_or_dataset = handle
    else:
        handle = None

    try:
        yield path_or_dataset
    finally:
        if handle is not None:
            handle.close()
