"""
Tests for the ipwgml.target module.
"""
import numpy as np
import xarray as xr

from ipwgml.target import TargetConfig


def test_load_load_data_spatial(spr_gmi_gridded_spatial_train):
    """
    Test loading of target data.
    """
    target_path = spr_gmi_gridded_spatial_train / "spr" / "gmi" / "gridded" / "spatial" / "train" / "target"
    target_files = sorted(list(target_path.glob("*.nc")))

    target_data = xr.load_dataset(target_files[0])

    target_config = TargetConfig(
        target="surface_precip",
        min_rqi=1.0,
        no_snow=False,
        no_hail=False,
        min_gcf=None,
        max_gcf=None
    )
    precip_data = target_config.load_data(target_files[0])
    valid = np.isfinite(precip_data)
    assert target_data["radar_quality_index"].data[valid].min() >= 1.0
    assert (target_data["snow_fraction"].data > 0.0).any()

    # Test varying minimum RQI requirement
    target_config = TargetConfig(
        target="surface_precip",
        min_rqi=0.0,
        no_snow=False,
        no_hail=False,
        min_gcf=None,
        max_gcf=None
    )
    precip_data = target_config.load_data(target_files[0])
    valid = np.isfinite(precip_data)
    assert target_data["radar_quality_index"].data[valid].min() < 1.0

    # Test no snow requirement.
    target_config = TargetConfig(
        target="surface_precip",
        min_rqi=1.0,
        no_snow=True,
        no_hail=False,
        min_gcf=None,
        max_gcf=None
    )
    precip_data = target_config.load_data(target_files[0])
    valid = np.isfinite(precip_data)
    assert target_data["radar_quality_index"].data[valid].min() >= 1.0
    assert (target_data["snow_fraction"].data[valid] == 0.0).all()


def test_load_load_data_tabular(spr_gmi_on_swath_tabular_train):
    """
    Test loading of target data.
    """
    target_path = spr_gmi_on_swath_tabular_train / "spr" / "gmi" / "on_swath" / "tabular" / "train" / "target"
    target_files = sorted(list(target_path.glob("*.nc")))

    target_data = xr.load_dataset(target_files[0])

    target_config = TargetConfig(
        target="surface_precip",
        min_rqi=1.0,
        no_snow=False,
        no_hail=False,
        min_gcf=None,
        max_gcf=None
    )
    precip_data = target_config.load_data(target_files[0])
    valid = np.isfinite(precip_data)
    assert target_data["radar_quality_index"].data[valid].min() >= 1.0
    assert (target_data["snow_fraction"].data > 0.0).any()

    # Test varying minimum RQI requirement
    target_config = TargetConfig(
        target="surface_precip",
        min_rqi=0.0,
        no_snow=False,
        no_hail=False,
        min_gcf=None,
        max_gcf=None
    )
    precip_data = target_config.load_data(target_files[0])
    valid = np.isfinite(precip_data)
    assert target_data["radar_quality_index"].data[valid].min() < 1.0

    # Test no snow requirement.
    target_config = TargetConfig(
        target="surface_precip",
        min_rqi=1.0,
        no_snow=True,
        no_hail=False,
        min_gcf=None,
        max_gcf=None
    )
    precip_data = target_config.load_data(target_files[0])
    valid = np.isfinite(precip_data)
    assert target_data["radar_quality_index"].data[valid].min() >= 1.0
    assert (target_data["snow_fraction"].data[valid] == 0.0).all()
