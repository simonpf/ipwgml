import pytest

from ipwgml.data import (
    download_missing
)


TEST_URL = "https://rain.atmos.colostate.edu/gprof_nn/ipwgml/.test"



@pytest.fixture(scope="session")
def spr_gmi_gridded_spatial_train(tmp_path_factory):
    """
    Fixture to download satellite-precipitation retrieval benchmark data for GMI with
    gridded geometry.
    """
    dest = tmp_path_factory.mktemp("ipwgml")
    download_missing("spr/gmi/gridded/spatial/train/pmw", dest, base_url=TEST_URL)
    download_missing("spr/gmi/gridded/spatial/train/ancillary", dest, base_url=TEST_URL)
    download_missing("spr/gmi/gridded/spatial/train/target", dest, base_url=TEST_URL)
    return dest / "spr" / "gmi" / "gridded" / "spatial" / "train"


@pytest.fixture(scope="session")
def spr_gmi_gridded_tabular_train(tmp_path_factory):
    """
    Fixture to download satellite-precipitation retrieval benchmark data for GMI with
    gridded geometry.
    """
    dest = tmp_path_factory.mktemp("ipwgml")
    download_missing("spr/gmi/gridded/tabular/train/", dest, base_url=TEST_URL)
    return dest / "spr" / "gmi" / "gridded" / "tabular" / "train"
