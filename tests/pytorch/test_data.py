"""
Tests for the ipwgml.pytorch.data module.
"""
from ipwgml.pytorch.data import SPRTabular


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
