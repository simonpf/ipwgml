"""
Tests for the ipwgml.input module.
"""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from ipwgml.input import (
    normalize,
    InputConfig,
    GMI,
    Ancillary,
    calculate_input_features,
)


def test_normalize():
    """
    Test normalization of input data.
    """
    data = np.random.rand(128, 128)
    stats = xr.Dataset({"min": 0, "max": 1, "mean": 0.5, "std_dev": 1.0})

    data_n = normalize(data, stats, "standardize")
    assert (0.0 <= data).all()
    assert (data_n < 0.0).any()
    assert data.mean() > 0.0
    assert np.isclose(data_n.mean(), 0.0, atol=1e-2)

    data = np.random.rand(128, 128)
    stats = xr.Dataset({"min": 0, "max": 1, "mean": 0.5, "std_dev": 1.0})
    data_n = normalize(data, stats, "minmax")
    assert (0.0 <= data).all()
    assert (data_n < 0.0).any()
    assert data.mean() > 0.0
    assert np.isclose(data_n.mean(), 0.0, atol=1e-2)

    data = np.random.rand(128, 128)
    data[data > 0.5] = np.nan
    data_n = normalize(data, stats, "minmax", nan=-1.5)
    assert np.isclose(data_n.min(), -1.5)


def test_parsing():
    """
    Test parsing of input data configs.
    """
    inpt = "gmi"
    cfg = InputConfig.parse(inpt)
    assert isinstance(cfg, GMI)

    inpt = {"name": "GMI", "channels": [0, 1]}
    cfg = InputConfig.parse(inpt)
    assert isinstance(cfg, GMI)

    cfg = GMI(channels=[0, 1])
    assert isinstance(cfg, GMI)

    inpt = "ancillary"
    cfg = InputConfig.parse(inpt)
    assert isinstance(cfg, Ancillary)

    inpt = {"name": "ancillary", "variables": ["two_meter_temperature"]}
    cfg = InputConfig.parse(inpt)
    assert isinstance(cfg, Ancillary)

    cfg = Ancillary(variables=["two_meter_temperature"])
    assert isinstance(cfg, Ancillary)


def test_gmi_input(spr_gmi_gridded_spatial_train):
    """
    Test loading of GMI input data.
    """
    gmi_path = (
        spr_gmi_gridded_spatial_train
        / "spr"
        / "gmi"
        / "training"
        / "gridded"
        / "spatial"
        / "gmi"
    )
    gmi_files = sorted(list(gmi_path.glob("*.nc")))
    target_path = (
        spr_gmi_gridded_spatial_train
        / "spr"
        / "gmi"
        / "training"
        / "gridded"
        / "spatial"
        / "target"
    )
    target_files = sorted(list(target_path.glob("*.nc")))
    inpt = {"name": "gmi", "channels": [0, 1]}
    cfg = InputConfig.parse(inpt)
    target_data = xr.load_dataset(target_files[0])
    inpt_data = cfg.load_data(gmi_files[0], target_time=target_data.time)

    assert inpt_data["obs_gmi"].shape[0] == cfg.features["obs_gmi"]
    assert inpt_data["eia_gmi"].shape[0] == cfg.features["obs_gmi"]

    assert "obs_gmi" in inpt_data
    assert inpt_data["obs_gmi"].shape[0] == 2
    assert "eia_gmi" in inpt_data

    assert cfg.stats is not None

    obs = inpt_data["obs_gmi"]
    assert np.isnan(obs).any()
    valid = np.isfinite(obs)
    assert np.all(obs[valid] > 0.0)

    # Test replacement of NAN value
    inpt = {"name": "gmi", "channels": [0, 1], "normalize": "minmax", "nan": -1.5}
    cfg = InputConfig.parse(inpt)
    target_data = xr.load_dataset(target_files[0])
    inpt_data = cfg.load_data(gmi_files[0], target_time=target_data.time)

    obs = inpt_data["obs_gmi"]
    assert np.isfinite(obs).all()
    valid = np.isfinite(obs)
    assert not np.all(obs[valid] > 0.0)


def test_ancillary_input(spr_gmi_gridded_spatial_train):
    """
    Test loading of ancillary input data.
    """
    anc_path = (
        spr_gmi_gridded_spatial_train
        / "spr"
        / "gmi"
        / "training"
        / "gridded"
        / "spatial"
        / "ancillary"
    )
    anc_files = sorted(list(anc_path.glob("*.nc")))
    target_path = (
        spr_gmi_gridded_spatial_train
        / "spr"
        / "gmi"
        / "training"
        / "gridded"
        / "spatial"
        / "target"
    )
    target_files = sorted(list(target_path.glob("*.nc")))
    inpt = {"name": "ancillary", "variables": ["total_column_water_vapor"]}
    cfg = InputConfig.parse(inpt)

    target_data = xr.load_dataset(target_files[0])
    inpt_data = cfg.load_data(anc_files[0], target_time=target_data.time)

    assert "ancillary" in inpt_data
    assert inpt_data["ancillary"].shape[0] == 1
    assert inpt_data["ancillary"].shape[0] == cfg.features["ancillary"]


def test_geo_ir_input(spr_gmi_gridded_spatial_train):
    """
    Test loading of GEO-IR input data.
    """
    geo_ir_path = (
        spr_gmi_gridded_spatial_train
        / "spr"
        / "gmi"
        / "training"
        / "gridded"
        / "spatial"
        / "geo_ir"
    )
    geo_ir_files = sorted(list(geo_ir_path.glob("*.nc")))
    target_path = (
        spr_gmi_gridded_spatial_train
        / "spr"
        / "gmi"
        / "training"
        / "gridded"
        / "spatial"
        / "target"
    )
    target_files = sorted(list(target_path.glob("*.nc")))

    inpt = {"name": "geo_ir", "time_steps": [0, 1, 2, 3]}
    cfg = InputConfig.parse(inpt)
    target_data = xr.load_dataset(target_files[0])
    inpt_data = cfg.load_data(geo_ir_files[0], target_time=target_data.time)
    assert "obs_geo_ir" in inpt_data
    assert inpt_data["obs_geo_ir"].shape[0] == len(cfg.time_steps)
    assert inpt_data["obs_geo_ir"].shape[0] == cfg.features["obs_geo_ir"]

    inpt = {"name": "geo_ir", "nearest": True, "normalize": "minmax", "nan": -1.5}
    cfg = InputConfig.parse(inpt)
    target_data = xr.load_dataset(target_files[0])
    inpt_data = cfg.load_data(geo_ir_files[0], target_time=target_data.time)
    assert "obs_geo_ir" in inpt_data
    assert inpt_data["obs_geo_ir"].shape[0] == 1
    assert inpt_data["obs_geo_ir"].shape[0] == cfg.features["obs_geo_ir"]

    assert cfg.stats is not None


def test_geo_input_spatial(spr_gmi_gridded_spatial_train):
    """
    Test loading of GEO input data.
    """
    ipwgml_path = spr_gmi_gridded_spatial_train
    geo_path = ipwgml_path / "spr" / "gmi" / "training" / "gridded" / "spatial" / "geo"
    geo_files = sorted(list(geo_path.glob("*.nc")))
    target_path = (
        ipwgml_path / "spr" / "gmi" / "training" / "gridded" / "spatial" / "target"
    )
    target_files = sorted(list(target_path.glob("*.nc")))

    inpt = {"name": "geo", "time_steps": [1, 2], "channels": [0, 3, 9]}
    cfg = InputConfig.parse(inpt)
    target_data = xr.load_dataset(target_files[0])
    inpt_data = cfg.load_data(geo_files[0], target_time=target_data.time)
    assert "obs_geo" in inpt_data
    assert inpt_data["obs_geo"].shape[0] == len(cfg.time_steps) * len(cfg.channels)
    assert inpt_data["obs_geo"].shape[0] == cfg.features["obs_geo"]

    inpt = {"name": "geo", "nearest": True}
    cfg = InputConfig.parse(inpt)
    target_data = xr.load_dataset(target_files[0])
    inpt_data = cfg.load_data(geo_files[0], target_time=target_data.time)
    assert "obs_geo" in inpt_data
    assert inpt_data["obs_geo"].shape[0] == len(cfg.channels)
    assert inpt_data["obs_geo"].shape[0] == cfg.features["obs_geo"]


def test_geo_input_tabular(spr_gmi_on_swath_tabular_train):
    """
    Test loading of GEO input data.
    """
    ipwgml_path = spr_gmi_on_swath_tabular_train
    geo_path = ipwgml_path / "spr" / "gmi" / "training" / "on_swath" / "tabular" / "geo"
    geo_files = sorted(list(geo_path.glob("*.nc")))
    target_path = (
        ipwgml_path / "spr" / "gmi" / "training" / "on_swath" / "tabular" / "target"
    )
    target_files = sorted(list(target_path.glob("*.nc")))

    inpt = {"name": "geo", "time_steps": [1, 2], "channels": [0, 3, 9]}
    cfg = InputConfig.parse(inpt)
    target_data = xr.load_dataset(target_files[0])
    inpt_data = cfg.load_data(geo_files[0], target_time=target_data.time)
    assert "obs_geo" in inpt_data
    assert inpt_data["obs_geo"].shape[0] == len(cfg.time_steps) * len(cfg.channels)
    assert inpt_data["obs_geo"].shape[0] == cfg.features["obs_geo"]

    inpt = {"name": "geo", "nearest": True}
    cfg = InputConfig.parse(inpt)
    target_data = xr.load_dataset(target_files[0])
    inpt_data = cfg.load_data(geo_files[0], target_time=target_data.time)
    assert "obs_geo" in inpt_data
    assert inpt_data["obs_geo"].shape[0] == len(cfg.channels)
    assert inpt_data["obs_geo"].shape[0] == cfg.features["obs_geo"]


def test_calculate_input_features():
    """
    Test calculation of input features.
    """
    inputs = [
        {"name": "gmi", "include_angles": True, "channels": [0, 3, 5]},
        {"name": "ancillary", "variables": ["two_meter_temperature", "land_fraction"]},
        {"name": "geo_ir", "time_steps": [8, 9, 10, 11]},
        {"name": "geo", "time_steps": [1, 2], "channels": [0, 1, 2]},
    ]

    features = calculate_input_features(inputs, stack=False)
    assert features["obs_gmi"] == 3
    assert features["eia_gmi"] == 3
    assert features["ancillary"] == 2
    assert features["obs_geo_ir"] == 4
    assert features["obs_geo"] == 6

    features = calculate_input_features(inputs, stack=True)
    assert features == 18
