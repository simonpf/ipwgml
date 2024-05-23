"""
Tests for the ipwgml.pytorch.data module.
"""
from ipwgml.pytorch.datasets import SPRTabular, SPRSpatial


def test_dataset_spr_tabular(spr_gmi_native_tabular_train):
    """
    Test loading of tabular data from the SPR dataset.
    """
    data_path = spr_gmi_native_tabular_train
    dataset = SPRTabular(
        sensor="gmi",
        geometry="native",
        split="train",
        retrieval_input=["pmw", "ancillary"],
        ipwgml_path=data_path,
        download=False
    )

    assert len(dataset) > 0
    x, y = next(iter(dataset))
    assert "obs_pmw" in x
    assert x["obs_pmw"].shape == (13,)
    assert "ancillary" in x
    assert y.numel() == 1


def test_dataset_spr_spatial(spr_gmi_gridded_spatial_train):
    """
    Test loading of tabular data from the SPR dataset.
    """
    data_path = spr_gmi_gridded_spatial_train
    dataset = SPRSpatial(
        sensor="gmi",
        geometry="gridded",
        split="train",
        retrieval_input=["pmw", "ancillary"],
        ipwgml_path=data_path,
        download=False
    )

    assert len(dataset) > 0
    x, y = next(iter(dataset))
    assert "obs_pmw" in x
    assert x["obs_pmw"].shape == (13, 256, 256)
    assert "ancillary" in x
    assert y.shape == (256, 256)
