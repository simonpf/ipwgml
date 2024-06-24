"""
Tests for the ipwgml.data module.
"""

from conftest import TEST_URL

from ipwgml.data import list_files


def test_list_files():
    """
    Tests finding files from SPR dataset and ensure that more than on file is found.
    """
    files = list_files("spr/gmi/training/gridded/spatial/gmi", base_url=TEST_URL)
    assert len(files) > 0
    parts = files[0].split(".")
    assert len(parts) == 2
    assert parts[-1] == "nc"


def test_download_files_spr_gmi_gridded_spatial_train(spr_gmi_gridded_spatial_train):
    """
    Ensure that fixture successfully downloaded files.
    """
    ds_path = (
        spr_gmi_gridded_spatial_train
        / "spr"
        / "gmi"
        / "training"
        / "gridded"
        / "spatial"
    )
    gmi_files = list((ds_path / "gmi").glob("*.nc"))
    assert len(gmi_files) == 2
    ancillary_files = list((ds_path / "ancillary").glob("*.nc"))
    assert len(ancillary_files) == 2
    ancillary_files = list((ds_path / "geo_ir").glob("*.nc"))
    assert len(ancillary_files) == 2
    target_files = list((ds_path / "target").glob("*.nc"))
    assert len(target_files) == 2


def test_download_files_spr_gmi_on_swath_tabular_train(spr_gmi_on_swath_tabular_train):
    """
    Ensure that fixture successfully downloaded files.
    """
    ds_path = (
        spr_gmi_on_swath_tabular_train
        / "spr"
        / "gmi"
        / "training"
        / "on_swath"
        / "tabular"
    )
    files = list((ds_path / "gmi").glob("*.nc"))
    assert len(files) == 1
    files = list((ds_path / "ancillary").glob("*.nc"))
    assert len(files) == 1
    files = list((ds_path / "geo_ir").glob("*.nc"))
    assert len(files) == 1
    files = list((ds_path / "target").glob("*.nc"))
    assert len(files) == 1


def test_download_files_spr_gmi_evaluation(spr_gmi_evaluation):
    """
    Ensure that fixture successfully downloaded files.
    """
    ds_path = spr_gmi_evaluation / "spr" / "gmi" / "evaluation" / "gridded"
    files = list((ds_path / "gmi").glob("*.nc"))
    assert len(files) == 1
    files = list((ds_path / "ancillary").glob("*.nc"))
    assert len(files) == 1
    # files = list((ds_path / "geo_ir").glob("*.nc"))
    # assert len(files) == 1
    files = list((ds_path / "target").glob("*.nc"))
    assert len(files) == 1

    ds_path = spr_gmi_evaluation / "spr" / "gmi" / "evaluation" / "on_swath"
    files = list((ds_path / "gmi").glob("*.nc"))
    assert len(files) == 1
    files = list((ds_path / "ancillary").glob("*.nc"))
    assert len(files) == 1
    # files = list((ds_path / "geo_ir").glob("*.nc"))
    # assert len(files) == 1
    files = list((ds_path / "target").glob("*.nc"))
    assert len(files) == 1
