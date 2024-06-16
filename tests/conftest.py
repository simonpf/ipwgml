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
    download_missing("spr/gmi/training/gridded/spatial/pmw", dest, base_url=TEST_URL)
    download_missing("spr/gmi/training/gridded/spatial/ancillary", dest, base_url=TEST_URL)
    download_missing("spr/gmi/training/gridded/spatial/target", dest, base_url=TEST_URL)
    download_missing("spr/gmi/training/gridded/spatial/geo_ir", dest, base_url=TEST_URL)
    download_missing("spr/gmi/training/gridded/spatial/geo", dest, base_url=TEST_URL)
    return dest


@pytest.fixture(scope="session")
def spr_gmi_on_swath_tabular_train(tmp_path_factory):
    """
    Fixture to download satellite-precipitation retrieval benchmark data for GMI with
    on_swath geometry.
    """
    dest = tmp_path_factory.mktemp("ipwgml")
    download_missing("spr/gmi/training/on_swath/tabular/pmw", dest, base_url=TEST_URL)
    download_missing("spr/gmi/training/on_swath/tabular/ancillary", dest, base_url=TEST_URL)
    download_missing("spr/gmi/training/on_swath/tabular/target", dest, base_url=TEST_URL)
    download_missing("spr/gmi/training/on_swath/tabular/geo_ir", dest, base_url=TEST_URL)
    download_missing("spr/gmi/training/on_swath/tabular/geo", dest, base_url=TEST_URL)
    return dest


@pytest.fixture(scope="session")
def spr_gmi_evaluation(tmp_path_factory):
    """
    Fixture to download satellite-precipitation retrieval evaluation data for GMI.
    """
    dest = tmp_path_factory.mktemp("ipwgml")
    download_missing("spr/gmi/evaluation/gridded/pmw", dest, base_url=TEST_URL)
    download_missing("spr/gmi/evaluation/gridded/ancillary", dest, base_url=TEST_URL)
    download_missing("spr/gmi/evaluation/gridded/target", dest, base_url=TEST_URL)
    download_missing("spr/gmi/evaluation/on_swath/pmw", dest, base_url=TEST_URL)
    download_missing("spr/gmi/evaluation/on_swath/ancillary", dest, base_url=TEST_URL)
    download_missing("spr/gmi/evaluation/on_swath/target", dest, base_url=TEST_URL)
    return dest
