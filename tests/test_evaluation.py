"""
Tests for the ipwgml.evaluation module.
"""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from ipwgml.evaluation import (
    InputFiles,
    Evaluator,
    load_retrieval_input_data,
    process_scene_spatial,
    process_scene_tabular,
)
from ipwgml.input import InputConfig, GMI, Ancillary
from ipwgml.metrics import Metric, Bias, MSE, CorrelationCoef


def test_find_files(spr_gmi_evaluation):
    """
    Ensure that the evaluator find evaluation files.
    """
    evaluator = Evaluator(
        "gmi",
        "on_swath",
        ["gmi", "ancillary"],
        ipwgml_path=spr_gmi_evaluation,
        download=False,
    )

    assert len(evaluator) == 1
    assert evaluator.gmi_on_swath is not None
    assert evaluator.ancillary_on_swath is not None
    assert evaluator.target_gridded is not None
    assert evaluator.target_on_swath is not None


def test_load_input_data(spr_gmi_evaluation):
    """
    Test loading of input data for retrieval evaluation.
    """
    target_file_gridded = (
        spr_gmi_evaluation
        / "spr"
        / "gmi"
        / "evaluation"
        / "gridded"
        / "target"
        / "target_20230701195312.nc"
    )
    target_file_on_swath = (
        spr_gmi_evaluation
        / "spr"
        / "gmi"
        / "evaluation"
        / "on_swath"
        / "target"
        / "target_20230701195312.nc"
    )
    gmi_file_gridded = (
        spr_gmi_evaluation
        / "spr"
        / "gmi"
        / "evaluation"
        / "gridded"
        / "gmi"
        / "gmi_20230701195312.nc"
    )
    gmi_file_on_swath = (
        spr_gmi_evaluation
        / "spr"
        / "gmi"
        / "evaluation"
        / "on_swath"
        / "gmi"
        / "gmi_20230701195312.nc"
    )
    ancillary_file_gridded = (
        spr_gmi_evaluation
        / "spr"
        / "gmi"
        / "evaluation"
        / "gridded"
        / "ancillary"
        / "ancillary_20230701195312.nc"
    )
    ancillary_file_on_swath = (
        spr_gmi_evaluation
        / "spr"
        / "gmi"
        / "evaluation"
        / "on_swath"
        / "ancillary"
        / "ancillary_20230701195312.nc"
    )
    target_file = (
        spr_gmi_evaluation
        / "spr"
        / "gmi"
        / "evaluation"
        / "gridded"
        / "target"
        / "target_20230701195312.nc"
    )

    input_files = InputFiles(
        target_file_gridded,
        target_file_on_swath,
        gmi_file_gridded,
        gmi_file_on_swath,
        None,
        None,
        ancillary_file_gridded,
        ancillary_file_on_swath,
        None,
        None,
        None,
        None,
    )

    input_data = load_retrieval_input_data(
        input_files, retrieval_input=[GMI(), Ancillary()], geometry="on_swath"
    )
    assert "scan" in input_data.dims
    assert "pixel" in input_data.dims
    assert "obs_gmi" in input_data
    assert "eia_gmi" in input_data
    assert "latitude" in input_data
    assert "longitude" in input_data
    assert "time" in input_data

    assert "gpm_input_file" in input_data.attrs
    assert "scan_start" in input_data.attrs
    assert "scan_end" in input_data.attrs

    input_data = load_retrieval_input_data(
        input_files, retrieval_input=[GMI(), Ancillary()], geometry="gridded"
    )
    assert "latitude" in input_data.dims
    assert "longitude" in input_data.dims
    assert "obs_gmi" in input_data
    assert "eia_gmi" in input_data
    assert "time" in input_data


@pytest.fixture
def input_data_gridded(spr_gmi_evaluation):
    target_file_gridded = (
        spr_gmi_evaluation
        / "spr"
        / "gmi"
        / "evaluation"
        / "gridded"
        / "target"
        / "target_20230701195312.nc"
    )
    target_file_on_swath = (
        spr_gmi_evaluation
        / "spr"
        / "gmi"
        / "evaluation"
        / "on_swath"
        / "target"
        / "target_20230701195312.nc"
    )
    gmi_file_gridded = (
        spr_gmi_evaluation
        / "spr"
        / "gmi"
        / "evaluation"
        / "gridded"
        / "gmi"
        / "gmi_20230701195312.nc"
    )
    gmi_file_on_swath = (
        spr_gmi_evaluation
        / "spr"
        / "gmi"
        / "evaluation"
        / "on_swath"
        / "gmi"
        / "gmi_20230701195312.nc"
    )
    ancillary_file_gridded = (
        spr_gmi_evaluation
        / "spr"
        / "gmi"
        / "evaluation"
        / "gridded"
        / "ancillary"
        / "ancillary_20230701195312.nc"
    )
    ancillary_file_on_swath = (
        spr_gmi_evaluation
        / "spr"
        / "gmi"
        / "evaluation"
        / "on_swath"
        / "ancillary"
        / "ancillary_20230701195312.nc"
    )
    target_file = (
        spr_gmi_evaluation
        / "spr"
        / "gmi"
        / "evaluation"
        / "gridded"
        / "target"
        / "target_20230701195312.nc"
    )

    input_files = InputFiles(
        target_file_gridded,
        target_file_on_swath,
        gmi_file_gridded,
        gmi_file_on_swath,
        None,
        None,
        ancillary_file_gridded,
        ancillary_file_on_swath,
        None,
        None,
        None,
        None,
    )

    input_data = load_retrieval_input_data(
        input_files, retrieval_input=[GMI(), Ancillary()], geometry="gridded"
    )
    return input_data


@pytest.fixture
def input_data_on_swath(spr_gmi_evaluation):
    target_file_gridded = (
        spr_gmi_evaluation
        / "spr"
        / "gmi"
        / "evaluation"
        / "gridded"
        / "target"
        / "target_20230701195312.nc"
    )
    target_file_on_swath = (
        spr_gmi_evaluation
        / "spr"
        / "gmi"
        / "evaluation"
        / "on_swath"
        / "target"
        / "target_20230701195312.nc"
    )
    gmi_file_gridded = (
        spr_gmi_evaluation
        / "spr"
        / "gmi"
        / "evaluation"
        / "gridded"
        / "gmi"
        / "gmi_20230701195312.nc"
    )
    gmi_file_on_swath = (
        spr_gmi_evaluation
        / "spr"
        / "gmi"
        / "evaluation"
        / "on_swath"
        / "gmi"
        / "gmi_20230701195312.nc"
    )
    ancillary_file_gridded = (
        spr_gmi_evaluation
        / "spr"
        / "gmi"
        / "evaluation"
        / "gridded"
        / "ancillary"
        / "ancillary_20230701195312.nc"
    )
    ancillary_file_on_swath = (
        spr_gmi_evaluation
        / "spr"
        / "gmi"
        / "evaluation"
        / "on_swath"
        / "ancillary"
        / "ancillary_20230701195312.nc"
    )
    target_file = (
        spr_gmi_evaluation
        / "spr"
        / "gmi"
        / "evaluation"
        / "gridded"
        / "target"
        / "target_20230701195312.nc"
    )

    input_files = InputFiles(
        target_file_gridded,
        target_file_on_swath,
        gmi_file_gridded,
        gmi_file_on_swath,
        None,
        None,
        ancillary_file_gridded,
        ancillary_file_on_swath,
        None,
        None,
        None,
        None,
    )

    input_data = load_retrieval_input_data(
        input_files, retrieval_input=[GMI(), Ancillary()], geometry="on_swath"
    )
    return input_data


@pytest.mark.parametrize(
    "input_data_fixture", ["input_data_gridded", "input_data_on_swath"]
)
def test_process_tiled(input_data_fixture, request):
    """
    Test tiled non-batch processing of evaluation scenes. Ensure that
    - Input tiles have the requested shape
    - Results are assembled correctly by returning longitude and latitude coordinates
      in the surface_precip and probability_of_precip fields.
    """

    input_data = request.getfixturevalue(input_data_fixture)
    if "latitude" in input_data.dims:
        spatial_dims = ["latitude", "longitude"]
    else:
        spatial_dims = ["scan", "pixel"]

    inputs = []

    def retrieval_fn(input_data):
        """
        Dummy retrieval function that stores all input in the 'inputs' variable.
        """
        inputs.append(input_data)
        results = input_data[["obs_gmi"]].copy(deep=True)
        lons = input_data["longitude"].data
        lats = input_data["latitude"].data
        if lons.ndim == 1:
            lons, lats = np.meshgrid(lons, lats)

        lons_rounded = (lons - np.round(lons)) > 0
        lats_rounded = (lats - np.round(lats)) > 0

        results = xr.Dataset(
            {
                "surface_precip": (spatial_dims, lons),
                "probability_of_precip": (spatial_dims, lats),
                "precip_flag": (spatial_dims, lons_rounded),
                "heavy_precip_flag": (spatial_dims, lats_rounded),
            }
        )
        return results

    results = process_scene_spatial(
        input_data,
        tile_size=(64, 64),
        overlap=16,
        batch_size=None,
        retrieval_fn=retrieval_fn,
    )

    for inpt in inputs:
        if "scan" in inpt.dims:
            assert inpt.scan.size == 64
        if "pixel" in inpt.dims:
            assert inpt.pixel.size == 64
        if "latitude" in inpt.dims:
            assert inpt.latitude.size == 64
        if "longitude" in inpt.dims:
            assert inpt.sizes["longitude"] == 64
        assert "batch" not in inpt.dims

    if "scan" in input_data.dims:
        assert input_data.scan.size == results.scan.size
    if "pixel" in input_data.dims:
        assert input_data.pixel.size == results.pixel.size
    if "latitude" in input_data.dims:
        assert input_data.latitude.size == results.latitude.size
    if "longitude" in input_data.dims:
        assert input_data.longitude.size == results.longitude.size

    lons = results.surface_precip.data
    lats = results.probability_of_precip.data
    lons_ref = input_data.longitude.data
    lats_ref = input_data.latitude.data
    if lons_ref.ndim == 1:
        lons_ref, lats_ref = np.meshgrid(lons_ref, lats_ref)

    assert np.isclose(lons, lons_ref, rtol=1e-3).all()
    assert np.isclose(lats, lats_ref, rtol=1e-3).all()

    lons_rounded = (lons_ref - np.round(lons_ref)) > 0
    lats_rounded = (lats_ref - np.round(lats_ref)) > 0

    assert (lons_rounded == results["precip_flag"]).all()
    assert (lats_rounded == results["heavy_precip_flag"]).all()


@pytest.mark.parametrize(
    "input_data_fixture", ["input_data_gridded", "input_data_on_swath"]
)
def test_process_untiled(input_data_fixture, request):
    """
    Test evaluation with a given tile size and ensure that the full input
    is passed to retrieval fn.
    """
    input_data = request.getfixturevalue(input_data_fixture)
    if "latitude" in input_data.dims:
        spatial_dims = ["latitude", "longitude"]
    else:
        spatial_dims = ["scan", "pixel"]

    inputs = []

    def retrieval_fn(input_data):
        """
        Dummy retrieval function that stores all input in the 'inputs' variable.
        """
        inputs.append(input_data)
        results = input_data[["obs_gmi"]].copy(deep=True)
        lons = input_data["longitude"].data
        lats = input_data["latitude"].data
        if lons.ndim == 1:
            lons, lats = np.meshgrid(lons, lats)

        lons_rounded = (lons - np.round(lons)) > 0
        lats_rounded = (lats - np.round(lats)) > 0

        results = xr.Dataset(
            {
                "surface_precip": (spatial_dims, lons),
                "probability_of_precip": (spatial_dims, lats),
                "precip_flag": (spatial_dims, lons_rounded),
                "heavy_precip_flag": (spatial_dims, lats_rounded),
            }
        )
        return results

    results = process_scene_spatial(
        input_data,
        tile_size=None,
        overlap=16,
        batch_size=None,
        retrieval_fn=retrieval_fn,
    )

    assert len(inputs) == 1

    for inpt in inputs:
        if "scan" in inpt.dims:
            assert inpt.scan.size == input_data.scan.size
        if "pixel" in inpt.dims:
            assert inpt.pixel.size == input_data.pixel.size
        if "latitude" in inpt.dims:
            assert inpt.latitude.size == input_data.latitude.size
        if "longitude" in inpt.dims:
            assert inpt.longitude.size == input_data.longitude.size
        assert "batch" not in inpt.dims

    if "scan" in input_data.dims:
        assert input_data.scan.size == results.scan.size
    if "pixel" in input_data.dims:
        assert input_data.pixel.size == results.pixel.size
    if "latitude" in input_data.dims:
        assert input_data.latitude.size == results.latitude.size
    if "longitude" in input_data.dims:
        assert input_data.longitude.size == results.longitude.size

    lons = results.surface_precip.data
    lats = results.probability_of_precip.data
    lons_ref = input_data.longitude.data
    lats_ref = input_data.latitude.data
    if lons_ref.ndim == 1:
        lons_ref, lats_ref = np.meshgrid(lons_ref, lats_ref)

    assert np.isclose(lons, lons_ref, rtol=1e-3).all()
    assert np.isclose(lats, lats_ref, rtol=1e-3).all()

    lons_rounded = (lons_ref - np.round(lons_ref)) > 0
    lats_rounded = (lats_ref - np.round(lats_ref)) > 0

    assert (lons_rounded == results["precip_flag"]).all()
    assert (lats_rounded == results["heavy_precip_flag"]).all()


@pytest.mark.parametrize(
    "input_data_fixture", ["input_data_gridded", "input_data_on_swath"]
)
def test_process_tiled_batched(input_data_fixture, request):
    """
    Test evaluation with a given tile size and ensure that input data
    passed to retrieval_fn has the expected tile size and that is is
    batched.
    """
    input_data = request.getfixturevalue(input_data_fixture)
    if "latitude" in input_data.dims:
        spatial_dims = ("latitude", "longitude")
    else:
        spatial_dims = ("scan", "pixel")

    inputs = []

    def retrieval_fn(input_data):
        """
        Dummy retrieval function that stores all input in the 'inputs' variable.
        """
        inputs.append(input_data)
        results = input_data[["obs_gmi"]].copy(deep=True)
        obs = input_data["obs_gmi"].data[:, 0]
        lons = input_data["longitude"].data
        lats = input_data["latitude"].data
        if lons.ndim == 2:
            lons = np.broadcast_to(lons[..., None, :], obs.shape)
            lats = np.broadcast_to(lats[..., None], obs.shape)

        lons_rounded = (lons - np.round(lons)) > 0
        lats_rounded = (lats - np.round(lats)) > 0

        results = xr.Dataset(
            {
                "surface_precip": (("batch",) + spatial_dims, lons),
                "probability_of_precip": (("batch",) + spatial_dims, lats),
                "precip_flag": (("batch",) + spatial_dims, lons_rounded),
                "heavy_precip_flag": (("batch",) + spatial_dims, lats_rounded),
            }
        )
        return results

    results = process_scene_spatial(
        input_data,
        tile_size=(64, 64),
        overlap=16,
        batch_size=8,
        retrieval_fn=retrieval_fn,
    )

    for inpt in inputs:
        if "scan" in inpt.dims:
            assert inpt.sizes["scan"] == 64
        if "pixel" in inpt.dims:
            assert inpt.sizes["pixel"] == 64
        if "latitude" in inpt.dims:
            assert inpt.sizes["latitude"] == 64
        if "longitude" in inpt.dims:
            assert inpt.sizes["longitude"] == 64
        assert "batch" in inpt.dims

    if "scan" in input_data.dims:
        assert input_data.scan.size == results.scan.size
    if "pixel" in input_data.dims:
        assert input_data.pixel.size == results.pixel.size
    if "latitude" in input_data.dims:
        assert input_data.latitude.size == results.latitude.size
    if "longitude" in input_data.dims:
        assert input_data.longitude.size == results.longitude.size

    lons = results.surface_precip.data
    lats = results.probability_of_precip.data
    lons_ref = input_data.longitude.data
    lats_ref = input_data.latitude.data
    if lons_ref.ndim == 1:
        lons_ref, lats_ref = np.meshgrid(lons_ref, lats_ref)

    assert np.isclose(lons, lons_ref, rtol=1e-3).all()
    assert np.isclose(lats, lats_ref, rtol=1e-3).all()

    lons_rounded = (lons_ref - np.round(lons_ref)) > 0
    lats_rounded = (lats_ref - np.round(lats_ref)) > 0

    assert (lons_rounded == results["precip_flag"]).all()
    assert (lats_rounded == results["heavy_precip_flag"]).all()


@pytest.mark.parametrize(
    "input_data_fixture", ["input_data_gridded", "input_data_on_swath"]
)
def test_process_tabular(input_data_fixture, request):
    """
    Test processing of granules with tabular input data.
    """
    input_data = request.getfixturevalue(input_data_fixture)

    inputs = []

    def retrieval_fn(input_data):
        """
        Dummy retrieval function that stores all input in the 'inputs' variable.
        """
        inputs.append(input_data)
        results = input_data[["obs_gmi"]].copy(deep=True)
        results = results[{"channels_gmi": 0}].rename(obs_gmi="surface_precip")
        return results

    results = process_scene_tabular(
        input_data, batch_size=256, retrieval_fn=retrieval_fn
    )

    for inpt in inputs[:-1]:
        assert inpt.batch.size == 256

    if "scan" in input_data.dims:
        assert input_data.scan.size == results.scan.size
    if "pixel" in input_data.dims:
        assert input_data.pixel.size == results.pixel.size
    if "latitude" in input_data.dims:
        assert input_data.latitude.size == results.latitude.size
    if "longitude" in input_data.dims:
        assert input_data.longitude.size == results.longitude.size


@pytest.mark.parametrize("geometry", ["gridded"])
def test_evaluate_scene(geometry, spr_gmi_evaluation, tmp_path):
    """
    Test running evaluation on a single scene with a retrieval returning the reference
    precipitation and ensure that the resulting correlation coefficient is 1.
    """
    evaluator = Evaluator(
        "gmi",
        geometry,
        ["gmi", "ancillary"],
        ipwgml_path=spr_gmi_evaluation,
        download=False,
    )

    target_data = xr.load_dataset(evaluator.target_gridded[0])[["surface_precip"]]
    if geometry == "on_swath":
        input_data = xr.load_dataset(evaluator.ancillary_on_swath[0])
        target_data = target_data.interp(
            latitude=input_data.latitude,
            longitude=input_data.longitude,
            method="nearest",
        )

    def retrieval_fn(*args):
        return target_data

    metrics = [
        Bias(),
        MSE(),
        CorrelationCoef(),
    ]
    evaluator.precip_quantification_metrics = metrics

    evaluator.evaluate_scene(
        0, None, None, None, retrieval_fn, "spatial", track=True, output_path=tmp_path
    )

    files = list(tmp_path.glob("results_*.nc"))
    assert len(files) > 0

    bias = metrics[0].compute()
    assert np.isclose(bias.bias.data, 0.0, atol=1e-3)

    mse = metrics[1].compute()
    assert np.isclose(mse.mse.data, 0.0)

    correlation_coef = metrics[2].compute()
    assert np.isclose(correlation_coef.correlation_coef.data, 1.0)


@pytest.mark.parametrize("geometry", ["gridded"])
def test_quantification_metrics(geometry, spr_gmi_evaluation, tmp_path):
    """
    Test accessing and setting of quantification metrics.
    """
    evaluator = Evaluator(
        "gmi",
        geometry,
        ["gmi", "ancillary"],
        ipwgml_path=spr_gmi_evaluation,
        download=False,
    )

    metrics = evaluator.precip_quantification_metrics
    for metric in metrics:
        assert isinstance(metric, Metric)

    evaluator.precip_quantification_metrics = ["Bias"]
    metrics = evaluator.precip_quantification_metrics
    assert len(metrics) == 1
    assert isinstance(metrics[0], Bias)

    metric = Bias()
    evaluator.precip_quantification_metrics = [metric]
    metrics = evaluator.precip_quantification_metrics
    assert metric is metrics[0]


class RetrievalFn:
    def __init__(self, target_file: Path):
        self.target_data = xr.load_dataset(target_file)

    def __call__(self, *args):
        return self.target_data


@pytest.mark.parametrize("geometry", ["gridded"])
def test_evaluate(geometry, spr_gmi_evaluation, tmp_path):
    """
    Test evaluation over all files.
    """
    evaluator = Evaluator(
        "gmi",
        geometry,
        ["gmi", "ancillary"],
        ipwgml_path=spr_gmi_evaluation,
        download=False,
    )

    retrieval_fn = RetrievalFn(evaluator.target_gridded[0])

    evaluator.evaluate(
        retrieval_fn=retrieval_fn,
        input_data_format="spatial",
        n_processes=2,
        output_path=tmp_path,
    )

    files = list(tmp_path.glob("results_*.nc"))
    assert len(files) > 0

    results = evaluator.get_results()
