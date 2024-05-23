"""
Tests for the ipwgml.data module.
"""
from ipwgml.data import list_files


def test_list_files():
    """
    Tests finding files from SPR dataset and ensure that more than on file is found.
    """
    files = list_files("spr/gmi/gridded/spatial/train/pmw")
    assert len(files) > 0
    parts = files[0].split(".")
    assert len(parts) == 2
    assert parts[-1] == "nc"


def test_download_files_spr_gmi_gridded_spatial_train(spr_gmi_gridded_spatial_train):
    """
    Ensure that fixture successfully downloaded files.
    """
    ds_path = spr_gmi_gridded_spatial_train / "spr" / "gmi" / "gridded" / "spatial" / "train"
    pmw_files = list((ds_path / "pmw").glob("*.nc"))
    assert len(pmw_files) == 4
    ancillary_files = list((ds_path / "ancillary").glob("*.nc"))
    assert len(ancillary_files) == 4
    target_files = list((ds_path / "target").glob("*.nc"))
    assert len(target_files) == 4

def test_download_files_spr_gmi_native_tabular_train(spr_gmi_native_tabular_train):
    """
    Ensure that fixture successfully downloaded files.
    """
    ds_path = spr_gmi_native_tabular_train / "spr" / "gmi" / "native" / "tabular" / "train"
    files = list((ds_path / "pmw").glob("*.nc"))
    assert len(files) == 1
