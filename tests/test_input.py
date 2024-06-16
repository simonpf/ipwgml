"""
Tests for the ipwgml.input module.
"""
import pytest
import xarray as xr

from ipwgml.input import (
    InputConfig,
    PMW,
    Ancillary
)

def test_parsing():
    """
    Test parsing of input data records.
    """
    inpt = "pmw"
    rec = InputConfig.parse(inpt)
    assert isinstance(rec, PMW)

    inpt = {"name": "pmw", "channels": [0, 1]}
    rec = InputConfig.parse(inpt)
    assert isinstance(rec, PMW)

    inpt = PMW(channels=[0, 1])
    assert isinstance(rec, PMW)

    inpt = "ancillary"
    rec = InputConfig.parse(inpt)
    assert isinstance(rec, Ancillary)

    inpt = {"name": "ancillary", "variables": ["two_meter_temperature"]}
    rec = InputConfig.parse(inpt)
    assert isinstance(rec, Ancillary)

    inpt = Ancillary(variables=["two_meter_temperature"])
    assert isinstance(rec, Ancillary)


def test_pmw_input(spr_gmi_gridded_spatial_train):
    """
    Test loading of PMW input data.
    """
    pmw_path = spr_gmi_gridded_spatial_train / "spr" / "gmi" / "training" / "gridded" / "spatial" / "pmw"
    pmw_files = sorted(list(pmw_path.glob("*.nc")))
    target_path = spr_gmi_gridded_spatial_train / "spr" / "gmi" / "training" / "gridded" / "spatial" / "target"
    target_files = sorted(list(target_path.glob("*.nc")))
    inpt = {"name": "pmw", "channels": [0, 1]}
    rec = InputConfig.parse(inpt)
    target_data = xr.load_dataset(target_files[0])
    inpt_data = rec.load_data(pmw_files[0], target_time=target_data.time)

    assert "obs_pmw" in inpt_data
    assert inpt_data["obs_pmw"].shape[0] == 2
    assert "eia_pmw" in inpt_data


def test_ancillary_input(spr_gmi_gridded_spatial_train):
    """
    Test loading of ancillary input data.
    """
    anc_path = spr_gmi_gridded_spatial_train / "spr" / "gmi" / "training" / "gridded" / "spatial" / "ancillary"
    anc_files = sorted(list(anc_path.glob("*.nc")))
    target_path = spr_gmi_gridded_spatial_train / "spr" / "gmi" / "training" / "gridded" / "spatial" / "target"
    target_files = sorted(list(target_path.glob("*.nc")))
    inpt = {"name": "ancillary", "variables": ["total_column_water_vapor"]}
    rec = InputConfig.parse(inpt)

    target_data = xr.load_dataset(target_files[0])
    inpt_data = rec.load_data(anc_files[0], target_time=target_data.time)

    assert "ancillary" in inpt_data
    assert inpt_data["ancillary"].shape[0] == 1


def test_geo_ir_input(spr_gmi_gridded_spatial_train):
    """
    Test loading of GEO-IR input data.
    """
    geo_ir_path = spr_gmi_gridded_spatial_train / "spr" / "gmi" / "training" / "gridded" / "spatial" / "geo_ir"
    geo_ir_files = sorted(list(geo_ir_path.glob("*.nc")))
    target_path = spr_gmi_gridded_spatial_train / "spr" / "gmi" / "training" / "gridded" / "spatial" / "target"
    target_files = sorted(list(target_path.glob("*.nc")))

    inpt = {"name": "geo_ir", "time_steps": [0, 1, 2, 3]}
    rec = InputConfig.parse(inpt)
    target_data = xr.load_dataset(target_files[0])
    inpt_data = rec.load_data(geo_ir_files[0], target_time=target_data.time)
    assert "obs_geo_ir" in inpt_data
    assert inpt_data["obs_geo_ir"].shape[0] == len(rec.time_steps)

    inpt = {"name": "geo_ir", "nearest": True}
    rec = InputConfig.parse(inpt)
    target_data = xr.load_dataset(target_files[0])
    inpt_data = rec.load_data(geo_ir_files[0], target_time=target_data.time)
    assert "obs_geo_ir" in inpt_data
    assert inpt_data["obs_geo_ir"].shape[0] == 1


@pytest.mark.skip()
def test_geo_input(spr_gmi_gridded_spatial_train):
    """
    Test loading of GEO input data.
    """
    geo_path = spr_gmi_gridded_spatial_train / "spr" / "gmi" / "training" / "gridded" / "spatial" /  "geo"
    geo_files = sorted(list(geo_path.glob("*.nc")))
    target_path = spr_gmi_gridded_spatial_train / "spr" / "gmi" / "training" / "gridded" / "spatial" /  "target"
    target_files = sorted(list(target_path.glob("*.nc")))

    inpt = {"name": "geo", "time_steps": [1, 2]}
    rec = InputConfig.parse(inpt)
    target_data = xr.load_dataset(target_files[0])
    inpt_data = rec.load_data(geo_files[0], target_time=target_data.time)
    assert "obs_geo" in inpt_data
    assert inpt_data["obs_geo"].shape[0] == len(rec.time_steps)

    inpt = {"name": "geo", "nearest": True}
    rec = InputConfig.parse(inpt)
    target_data = xr.load_dataset(target_files[0])
    inpt_data = rec.load_data(geo_files[0], target_time=target_data.time)
    assert "obs_geo" in inpt_data
    assert inpt_data["obs_geo"].shape[0] == 1
