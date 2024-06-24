"""
Tests for the ipwgml.tiling module.
"""

import numpy as np
import xarray as xr

from ipwgml.tiling import get_starts_and_clips, DatasetTiler


def test_get_starts_and_clips():
    """
    Test calculation of starts and clips for all tiles and ensure that using them to
    tile an array produces tiles of the expected size and that assembly reproduces the
    original array.
    """
    tile_size = 64
    overlap = 32
    starts, clips = get_starts_and_clips(1024, tile_size, overlap)
    assert len(starts) == 2 * (1024 // tile_size) - 1
    assert len(clips) == len(starts) - 1

    arr = np.arange(1024)
    tiles = []
    for start in starts:
        tiles.append(arr[start : start + tile_size])

    ass = []
    for ind, tile in enumerate(tiles):
        assert tile.size == tile_size
        clip_l = 0 if ind == 0 else clips[ind - 1]
        clip_r = 0 if ind == len(clips) else clips[ind]
        ass.append(tile[clip_l : tile_size - clip_r])

    ass = np.concatenate(ass)
    assert np.all(np.isclose(ass, arr))


def test_dataset_tiler(spr_gmi_evaluation):
    """
    Ensure that tiling of an xarray.Dataset produces tiles of the expected size.
    """
    ds_path = spr_gmi_evaluation / "spr" / "gmi" / "evaluation" / "gridded"
    files = list((ds_path / "target").glob("*.nc"))
    dataset = xr.load_dataset(files[0])

    tiler = DatasetTiler(dataset, 256, overlap=64)
    for row_ind in range(tiler.n_rows_tiled):
        for col_ind in range(tiler.n_cols_tiled):
            tile = tiler.get_tile(row_ind, col_ind)
            assert tile.latitude.size == 256
            assert tile.longitude.size == 256


def test_dataset_tiler_calculate_weights(spr_gmi_evaluation):
    """
    Ensure that tiling an xarray.Dataset and reassembling the tiles reproduces the input data.
    """
    ds_path = spr_gmi_evaluation / "spr" / "gmi" / "evaluation" / "on_swath"
    files = list((ds_path / "target").glob("*.nc"))
    dataset = xr.load_dataset(files[0])
    results = dataset.copy(deep=True)
    results.surface_precip.data[:] = 0.0
    results["weights"] = (("scan", "pixel"), np.zeros_like(results.surface_precip.data))

    tiler = DatasetTiler(dataset, 64, overlap=16, spatial_dims=("scan", "pixel"))
    result_tiler = DatasetTiler(results, 64, overlap=16, spatial_dims=("scan", "pixel"))

    valid = np.isfinite(dataset.surface_precip.data)
    assert not np.all(
        np.isclose(
            results.surface_precip.data[valid], dataset.surface_precip.data[valid]
        )
    )

    for row_ind in range(tiler.n_rows_tiled):
        for col_ind in range(tiler.n_cols_tiled):
            weights = tiler.get_weights(row_ind, col_ind)
            assert weights.min() >= 0.0
            assert weights.max() <= 1.0
            tile = tiler.get_tile(row_ind, col_ind)
            result_tile = result_tiler.get_tile(row_ind, col_ind)
            result_tile.surface_precip.data += weights * tile.surface_precip.data
            result_tile.weights.data += weights

    assert np.all(
        np.isclose(
            results.surface_precip.data[valid], dataset.surface_precip.data[valid]
        )
    )
    assert np.all(np.isclose(results.weights.data, 1.0))


def test_tiler_trivial(spr_gmi_evaluation):
    """
    Ensure that tiler works even when tile size is None, i.e., the tile extends over the full
    spatial extent of the input data.
    """
    ds_path = spr_gmi_evaluation / "spr" / "gmi" / "evaluation" / "gridded"
    files = list((ds_path / "target").glob("*.nc"))
    dataset = xr.load_dataset(files[0])
    tiler = DatasetTiler(dataset, tile_size=None, overlap=64)

    assert tiler.n_rows_tiled == 1
    assert tiler.n_cols_tiled == 1

    for row_ind in range(tiler.n_rows_tiled):
        for col_ind in range(tiler.n_cols_tiled):
            weights = tiler.get_weights(row_ind, col_ind)
            assert np.all(weights == 1.0)
