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
    return dest


@pytest.fixture(scope="session")
def spr_gmi_native_tabular_train(tmp_path_factory):
    """
    Fixture to download satellite-precipitation retrieval benchmark data for GMI with
    native geometry.
    """
    dest = tmp_path_factory.mktemp("ipwgml")
    download_missing("spr/gmi/native/tabular/train/pmw", dest, base_url=TEST_URL)
    download_missing("spr/gmi/native/tabular/train/ancillary", dest, base_url=TEST_URL)
    download_missing("spr/gmi/native/tabular/train/target", dest, base_url=TEST_URL)
    return dest
