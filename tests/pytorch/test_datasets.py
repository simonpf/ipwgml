"""
Tests for the ipwgml.pytorch.data module.
"""
import torch

from ipwgml.pytorch.datasets import SPRTabular, SPRSpatial


def test_dataset_spr_tabular(spr_gmi_on_swath_tabular_train):
    """
    Test loading of tabular data from the SPR dataset.
    """
    data_path = spr_gmi_on_swath_tabular_train
    dataset = SPRTabular(
        sensor="gmi",
        geometry="on_swath",
        split="training",
        retrieval_input=["gmi", "geo", "geo_ir", "ancillary"],
        ipwgml_path=data_path,
        download=False
    )

    assert len(dataset) > 0
    x, y = next(iter(dataset))
    assert "obs_gmi" in x
    assert x["obs_gmi"].shape == (13,)
    assert "obs_geo_ir" in x
    assert x["obs_geo_ir"].shape == (8,)
    assert "obs_geo" in x
    assert x["obs_geo"].shape == (64,)
    assert "ancillary" in x
    assert y.numel() == 1


def test_dataset_spr_tabular_stacked(spr_gmi_on_swath_tabular_train):
    """
    Test loading of tabular data from the SPR dataset.
    """
    data_path = spr_gmi_on_swath_tabular_train
    dataset = SPRTabular(
        sensor="gmi",
        geometry="on_swath",
        split="training",
        retrieval_input=["gmi", "geo_ir", "ancillary"],
        ipwgml_path=data_path,
        stack=True,
        download=False
    )

    assert len(dataset) > 0
    x, y = next(iter(dataset))
    assert isinstance(x, torch.Tensor)
    assert y.numel() == 1


def test_dataset_spr_tabular_batched(spr_gmi_on_swath_tabular_train):
    """
    Test loading of tabular data from the SPR dataset with batching.
    """
    batch_size = 1024
    data_path = spr_gmi_on_swath_tabular_train
    dataset = SPRTabular(
        sensor="gmi",
        geometry="on_swath",
        split="training",
        retrieval_input=["gmi", "geo", "geo_ir", "ancillary"],
        ipwgml_path=data_path,
        download=False,
        batch_size=batch_size
    )

    assert len(dataset) > 0
    for ind, (x, y) in enumerate(dataset):
        if ind < len(dataset) - 1:
            assert "obs_gmi" in x
            assert x["obs_gmi"].shape == (batch_size, 13)
            assert x["obs_geo_ir"].shape == (batch_size, 8)
            assert x["obs_geo"].shape == (batch_size, 64)
            assert "ancillary" in x
            assert y.numel() == batch_size

    dataset = SPRTabular(
        sensor="gmi",
        geometry="on_swath",
        split="training",
        retrieval_input=[
            {"name": "gmi", "channels": [0, -2, -1]},
            {"name": "geo", "nearest": True},
            {"name": "geo_ir", "nearest": True},
            {"name": "ancillary", "variables": ["two_meter_temperature"]}
        ],
        ipwgml_path=data_path,
        download=False,
        batch_size=batch_size
    )

    assert len(dataset) > 0
    for ind, (x, y) in enumerate(dataset):
        if ind < len(dataset) - 1:
            assert "obs_gmi" in x
            assert x["obs_gmi"].shape == (batch_size, 3)
            assert x["obs_geo_ir"].shape == (batch_size, 1)
            #assert x["obs_geo"].shape == (batch_size, 16)
            assert "ancillary" in x
            assert x["ancillary"].shape == (batch_size, 1)
            assert y.numel() == batch_size


def test_dataset_spr_spatial(spr_gmi_gridded_spatial_train):
    """
    Test loading of tabular data from the SPR dataset.
    """
    data_path = spr_gmi_gridded_spatial_train
    dataset = SPRSpatial(
        sensor="gmi",
        geometry="gridded",
        split="training",
        retrieval_input=["gmi", "ancillary", "geo_ir"],
        ipwgml_path=data_path,
        download=False
    )

    assert len(dataset) > 0
    x, y = next(iter(dataset))
    assert "obs_gmi" in x
    assert x["obs_gmi"].shape == (13, 256, 256)
    assert "ancillary" in x
    assert y.shape == (256, 256)


def test_dataset_spr_spatial_stacked(spr_gmi_gridded_spatial_train):
    """
    Test loading of tabular data from the SPR dataset.
    """
    data_path = spr_gmi_gridded_spatial_train
    dataset = SPRSpatial(
        sensor="gmi",
        geometry="gridded",
        split="training",
        retrieval_input=["gmi", "ancillary", "geo_ir"],
        stack=True,
        ipwgml_path=data_path,
        download=False
    )

    assert len(dataset) > 0
    x, y = next(iter(dataset))
    assert isinstance(x, torch.Tensor)
