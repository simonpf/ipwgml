"""
Tests for the ipwgml.target module.
"""
import numpy as np
import xarray as xr

from ipwgml.target import TargetConfig


def test_load_load_reference_precip_spatial(spr_gmi_gridded_spatial_train):
    """
    Test loading of target data.
    """
    target_path = spr_gmi_gridded_spatial_train / "spr" / "gmi" / "training" / "gridded" / "spatial" / "target"
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
    precip_data = target_config.load_reference_precip(target_files[0])
    valid = np.isfinite(precip_data)
    assert np.isclose(target_data["radar_quality_index"].data[valid].min(), 1.0, rtol=1e-3)
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
    precip_data = target_config.load_reference_precip(target_files[0])
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
    precip_data = target_config.load_reference_precip(target_files[0])
    valid = np.isfinite(precip_data)
    assert np.isclose(target_data["radar_quality_index"].data[valid].min(), 1.0, rtol=1e-3)
    assert (target_data["snow_fraction"].data[valid] == 0.0).all()


def test_load_load_precip_mask_spatial(spr_gmi_gridded_spatial_train):
    """
    Test loading of target data.
    """
    target_path = spr_gmi_gridded_spatial_train / "spr" / "gmi" / "training" / "gridded" / "spatial" / "target"
    target_files = sorted(list(target_path.glob("*.nc")))

    target_data = xr.load_dataset(target_files[0])

    target_config = TargetConfig(
        precip_threshold=1.0
    )
    precip_data = target_config.load_reference_precip(target_data)
    precip_mask = target_config.load_precip_mask(target_data)
    mask = target_config.get_mask(target_data)
    assert (precip_data[~mask][precip_mask[~mask]] >= 1.0).all()


def test_load_load_heavy_precip_mask_spatial(spr_gmi_gridded_spatial_train):
    """
    Test loading of target data.
    """
    target_path = spr_gmi_gridded_spatial_train / "spr" / "gmi" / "training" / "gridded" / "spatial" / "target"
    target_files = sorted(list(target_path.glob("*.nc")))

    target_data = xr.load_dataset(target_files[0])

    target_config = TargetConfig(heavy_precip_threshold=11.0)
    precip_data = target_config.load_reference_precip(target_data)
    precip_mask = target_config.load_heavy_precip_mask(target_data)
    mask = target_config.get_mask(target_data)
    assert (precip_data[~mask][precip_mask[~mask]] >= 11.0).all()


def test_load_load_data_tabular(spr_gmi_on_swath_tabular_train):
    """
    Test loading of target data.
    """
    target_path = spr_gmi_on_swath_tabular_train / "spr" / "gmi" / "training" / "on_swath" / "tabular" / "target"
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
    precip_data = target_config.load_reference_precip(target_files[0])
    valid = np.isfinite(precip_data)
    assert np.isclose(target_data["radar_quality_index"].data[valid].min(), 1.0, rtol=1e-3)
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
    precip_data = target_config.load_reference_precip(target_files[0])
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
    precip_data = target_config.load_reference_precip(target_files[0])
    valid = np.isfinite(precip_data)
    assert np.isclose(target_data["radar_quality_index"].data[valid].min(), 1.0, rtol=1e-3)
    assert (target_data["snow_fraction"].data[valid] == 0.0).all()


def test_load_load_precip_mask_tabular(spr_gmi_on_swath_tabular_train):
    """
    Test loading of target data.
    """
    target_path = spr_gmi_on_swath_tabular_train / "spr" / "gmi" / "training" / "on_swath" / "tabular" / "target"
    target_files = sorted(list(target_path.glob("*.nc")))

    target_data = xr.load_dataset(target_files[0])

    target_config = TargetConfig(
        precip_threshold=1.0
    )
    precip_data = target_config.load_reference_precip(target_data)
    precip_mask = target_config.load_precip_mask(target_data)
    mask = target_config.get_mask(target_data)
    assert (precip_data[~mask][precip_mask[~mask]] >= 1.0).all()


def test_load_load_heavy_precip_mask_tabular(spr_gmi_on_swath_tabular_train):
    """
    Test loading of target data.
    """
    target_path = spr_gmi_on_swath_tabular_train / "spr" / "gmi" / "training" / "on_swath" / "tabular" / "target"
    target_files = sorted(list(target_path.glob("*.nc")))

    target_data = xr.load_dataset(target_files[0])

    target_config = TargetConfig(heavy_precip_threshold=11.0)
    precip_data = target_config.load_reference_precip(target_data)
    precip_mask = target_config.load_heavy_precip_mask(target_data)
    mask = target_config.get_mask(target_data)
    assert (precip_data[~mask][precip_mask[~mask]] >= 11.0).all()
