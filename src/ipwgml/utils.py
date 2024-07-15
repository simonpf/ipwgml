"""
ipwgml.utils
============

Defines helper functions used throught the ipwgml package.
"""

from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

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
        handle = xr.open_dataset(path_or_dataset, engine="h5netcdf")
        path_or_dataset = handle
    else:
        handle = None

    try:
        yield path_or_dataset
    finally:
        if handle is not None:
            handle.close()


def get_median_time(path: Path) -> datetime:
    """
    Extract median time from filename.
    """
    date = datetime.strptime(path.name.split("_")[-1][:-3], "%Y%m%d%H%M%S")
    return date


def cleanup_files(path: Path, no_action: bool = False) -> None:
    """
    Removes all files that do not have matching files in all input and target files.

    Args:
        path: A Path object pointing to the folder containing the
            IPWGML training scenes.
        no_action: Just print filename, don't remove any files.
    """
    path = Path(path)

    all_times = None
    for fldr in ["gmi", "target", "ancillary", "geo_ir", "geo"]:
        if not (path / "on_swath" / fldr).exists():
            continue
        files = sorted(list((path / "on_swath" / fldr).glob("*.nc")))
        times = set(map(get_median_time, files))
        if all_times is None:
            all_times = times
        else:
            all_times = all_times.intersection(times)
        if not (path / "gridded" / fldr).exists():
            continue
        files = sorted(list((path / "gridded" / fldr).glob("*.nc")))
        times = set(map(get_median_time, files))
        all_times = all_times.intersection(times)

    for fldr in ["target", "gmi", "ancillary", "geo_ir", "geo"]:
        if not (path / "on_swath" / fldr).exists():
            continue
        files = sorted(list((path / "on_swath" / fldr).glob("*.nc")))
        for fle in files:
            if get_median_time(fle) not in all_times:
                print("Extra file: ", fle)
                if not no_action:
                    fle.unlink()

        files = sorted(list((path / "on_gridded" / fldr).glob("*.nc")))
        for fle in files:
            if get_median_time(fle) not in all_times:
                print("Extra file: ", fle)
                if not no_action:
                    fle.unlink()
