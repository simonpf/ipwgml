"""
Tests for the ipwgml.utils module.
"""
import numpy as np
import xarray as xr


from ipwgml.utils import open_if_required


def test_open_if_required(tmp_path):
    """
    Test the 'open_if_required' context manager for flexible handling of paths pointing to
    NetCDF files and already-loaded data in the form of xr.Datasets.
    """

    test_data = xr.Dataset({
        "surface_precip": (("y", "x"), np.random.rand(64, 64))
    })
    test_data.to_netcdf(tmp_path / "test.nc")

    with open_if_required(tmp_path / "test.nc") as data:
        data_loaded = data.surface_precip

    assert np.all(data_loaded.data == test_data.surface_precip.data)

    with open_if_required(test_data) as data:
        data_not_loaded = data.surface_precip

    assert np.all(data_not_loaded.data == test_data.surface_precip.data)
