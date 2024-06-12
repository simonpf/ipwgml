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
from ipwgml.input import (
    InputConfig,
    PMW,
    Ancillary
)
from ipwgml.metrics import (
    Metric,
    Bias,
    MSE,
    CorrelationCoef
)


def test_find_files(spr_gmi_evaluation):
    """
    Ensure that the evaluator find evaluation files.
    """
    evaluator = Evaluator(
        "gmi",
        "on_swath",
        ["pmw", "ancillary"],
        ipwgml_path=spr_gmi_evaluation,
        download=False
    )

    assert len(evaluator) == 1
    assert evaluator.pmw_on_swath is not None
    assert evaluator.ancillary_on_swath is not None
    assert evaluator.target_gridded is not None
    assert evaluator.target_on_swath is not None


def test_load_input_data(spr_gmi_evaluation):
    """
    Test loading of input data for retrieval evaluation.
    """
    target_file_gridded = (
        spr_gmi_evaluation / "spr" / "gmi" / "evaluation"
        / "gridded" / "target" / "target_20230701212646.nc"
    )
    target_file_on_swath = (
        spr_gmi_evaluation / "spr" / "gmi" / "evaluation"
        / "on_swath" / "target" / "target_20230701212646.nc"
    )
    pmw_file_gridded = (
        spr_gmi_evaluation / "spr" / "gmi" / "evaluation"
        / "gridded" / "pmw" / "pmw_20230701212646.nc"
    )
    pmw_file_on_swath = (
        spr_gmi_evaluation / "spr" / "gmi" / "evaluation"
        / "on_swath" / "pmw" / "pmw_20230701212646.nc"
    )
    ancillary_file_gridded = (
        spr_gmi_evaluation / "spr" / "gmi" / "evaluation"
        / "gridded" / "ancillary" / "ancillary_20230701212646.nc"
    )
    ancillary_file_on_swath = (
        spr_gmi_evaluation / "spr" / "gmi" / "evaluation"
        / "on_swath" / "ancillary" / "ancillary_20230701212646.nc"
    )
    target_file = (
        spr_gmi_evaluation / "spr" / "gmi" / "evaluation"
        / "gridded" / "target" / "target_20230701212646.nc"
    )


    input_files = InputFiles(
        target_file_gridded,
        target_file_on_swath,
        pmw_file_gridded,
        pmw_file_on_swath,
        ancillary_file_gridded,
        ancillary_file_on_swath,
        None,
        None,
        None,
        None,
    )

    input_data = load_retrieval_input_data(
        input_files,
        retrieval_input=[PMW(), Ancillary()],
        geometry="on_swath"
    )
    assert "scans" in input_data.dims
    assert "pixels" in input_data.dims
    assert "obs_pmw" in input_data
    assert "eia_pmw" in input_data
    assert "latitude" in input_data
    assert "longitude" in input_data
    assert "time" in input_data

    assert "pmw_input_file" in input_data.attrs
    assert "scan_start" in input_data.attrs
    assert "scan_end" in input_data.attrs

    input_data = load_retrieval_input_data(
        input_files,
        retrieval_input=[PMW(), Ancillary()],
        geometry="gridded"
    )
    assert "latitude" in input_data.dims
    assert "longitude" in input_data.dims
    assert "obs_pmw" in input_data
    assert "eia_pmw" in input_data
    assert "time" in input_data


@pytest.fixture
def input_data_gridded(spr_gmi_evaluation):
    target_file_gridded = (
        spr_gmi_evaluation / "spr" / "gmi" / "evaluation"
        / "gridded" / "target" / "target_20230701212646.nc"
    )
    target_file_on_swath = (
        spr_gmi_evaluation / "spr" / "gmi" / "evaluation"
        / "on_swath" / "target" / "target_20230701212646.nc"
    )
    pmw_file_gridded = (
        spr_gmi_evaluation / "spr" / "gmi" / "evaluation"
        / "gridded" / "pmw" / "pmw_20230701212646.nc"
    )
    pmw_file_on_swath = (
        spr_gmi_evaluation / "spr" / "gmi" / "evaluation"
        / "on_swath" / "pmw" / "pmw_20230701212646.nc"
    )
    ancillary_file_gridded = (
        spr_gmi_evaluation / "spr" / "gmi" / "evaluation"
        / "gridded" / "ancillary" / "ancillary_20230701212646.nc"
    )
    ancillary_file_on_swath = (
        spr_gmi_evaluation / "spr" / "gmi" / "evaluation"
        / "on_swath" / "ancillary" / "ancillary_20230701212646.nc"
    )
    target_file = (
        spr_gmi_evaluation / "spr" / "gmi" / "evaluation"
        / "gridded" / "target" / "target_20230701212646.nc"
    )

    input_files = InputFiles(
        target_file_gridded,
        target_file_on_swath,
        pmw_file_gridded,
        pmw_file_on_swath,
        ancillary_file_gridded,
        ancillary_file_on_swath,
        None,
        None,
        None,
        None,
    )

    input_data = load_retrieval_input_data(
        input_files,
        retrieval_input=[PMW(), Ancillary()],
        geometry="gridded"
    )
    return input_data


@pytest.fixture
def input_data_on_swath(spr_gmi_evaluation):
    target_file_gridded = (
        spr_gmi_evaluation / "spr" / "gmi" / "evaluation"
        / "gridded" / "target" / "target_20230701212646.nc"
    )
    target_file_on_swath = (
        spr_gmi_evaluation / "spr" / "gmi" / "evaluation"
        / "on_swath" / "target" / "target_20230701212646.nc"
    )
    pmw_file_gridded = (
        spr_gmi_evaluation / "spr" / "gmi" / "evaluation"
        / "gridded" / "pmw" / "pmw_20230701212646.nc"
    )
    pmw_file_on_swath = (
        spr_gmi_evaluation / "spr" / "gmi" / "evaluation"
        / "on_swath" / "pmw" / "pmw_20230701212646.nc"
    )
    ancillary_file_gridded = (
        spr_gmi_evaluation / "spr" / "gmi" / "evaluation"
        / "gridded" / "ancillary" / "ancillary_20230701212646.nc"
    )
    ancillary_file_on_swath = (
        spr_gmi_evaluation / "spr" / "gmi" / "evaluation"
        / "on_swath" / "ancillary" / "ancillary_20230701212646.nc"
    )
    target_file = (
        spr_gmi_evaluation / "spr" / "gmi" / "evaluation"
        / "gridded" / "target" / "target_20230701212646.nc"
    )

    input_files = InputFiles(
        target_file_gridded,
        target_file_on_swath,
        pmw_file_gridded,
        pmw_file_on_swath,
        ancillary_file_gridded,
        ancillary_file_on_swath,
        None,
        None,
        None,
        None,
    )

    input_data = load_retrieval_input_data(
        input_files,
        retrieval_input=[PMW(), Ancillary()],
        geometry="on_swath"
    )
    return input_data


@pytest.mark.parametrize(
    "input_data_fixture",
    ["input_data_gridded", "input_data_on_swath"]
)
def test_process_tiled(input_data_fixture, request):
    """
    Test evaluation with a given tile size and ensure that input data
    passed to retrieval_fn has the expected tile size.
    """

    input_data = request.getfixturevalue(input_data_fixture)

    inputs = []
    def retrieval_fn(input_data):
        """
        Dummy retrieval function that stores all input in the 'inputs' variable.
        """
        inputs.append(input_data)
        results = input_data[["obs_pmw"]].copy(deep=True)
        results = results[{"channels_pmw": 0}].rename(
            obs_pmw="surface_precip"
        )
        return results

    results = process_scene_spatial(
        input_data,
        tile_size=(64, 64),
        overlap=16,
        batch_size=None,
        retrieval_fn=retrieval_fn
    )

    for inpt in inputs:
        if "scans" in inpt.dims:
            assert inpt.scans.size == 64
        if "pixels" in inpt.dims:
            assert inpt.pixels.size == 64
        if "latitude" in inpt.dims:
            assert inpt.latitude.size == 64
        if "longitude" in inpt.dims:
            assert inpt.sizes["longitude"] == 64
        assert "batch" not in inpt.dims

    if "scans" in input_data.dims:
        assert input_data.scans.size == results.scans.size
    if "pixels" in input_data.dims:
        assert input_data.pixels.size == results.pixels.size
    if "latitude" in input_data.dims:
        assert input_data.latitude.size == results.latitude.size
    if "longitude" in input_data.dims:
        assert input_data.longitude.size == results.longitude.size


@pytest.mark.parametrize(
    "input_data_fixture",
    ["input_data_gridded", "input_data_on_swath"]
)
def test_process_untiled(input_data_fixture, request):
    """
    Test evaluation with a given tile size and ensure that the full input
    is passed to retrieval fn.
    """
    input_data = request.getfixturevalue(input_data_fixture)

    inputs = []
    def retrieval_fn(input_data):
        """
        Dummy retrieval function that stores all input in the 'inputs' variable.
        """
        inputs.append(input_data)
        results = input_data[["obs_pmw"]].copy(deep=True)
        results = results[{"channels_pmw": 0}].rename(
            obs_pmw="surface_precip"
        )
        return results

    results = process_scene_spatial(
        input_data,
        tile_size=None,
        overlap=16,
        batch_size=None,
        retrieval_fn=retrieval_fn
    )

    assert len(inputs) == 1

    for inpt in inputs:
        if "scans" in inpt.dims:
            assert inpt.scans.size == input_data.scans.size
        if "pixels" in inpt.dims:
            assert inpt.pixels.size == input_data.pixels.size
        if "latitude" in inpt.dims:
            assert inpt.latitude.size == input_data.latitude.size
        if "longitude" in inpt.dims:
            assert inpt.longitude.size == input_data.longitude.size
        assert "batch" not in inpt.dims

    if "scans" in input_data.dims:
        assert input_data.scans.size == results.scans.size
    if "pixels" in input_data.dims:
        assert input_data.pixels.size == results.pixels.size
    if "latitude" in input_data.dims:
        assert input_data.latitude.size == results.latitude.size
    if "longitude" in input_data.dims:
        assert input_data.longitude.size == results.longitude.size


@pytest.mark.parametrize(
    "input_data_fixture",
    ["input_data_gridded", "input_data_on_swath"]
)
def test_process_tiled_batched(input_data_fixture, request):
    """
    Test evaluation with a given tile size and ensure that input data
    passed to retrieval_fn has the expected tile size and that is is
    batched.
    """

    input_data = request.getfixturevalue(input_data_fixture)

    inputs = []
    def retrieval_fn(input_data):
        """
        Dummy retrieval function that stores all input in the 'inputs' variable.
        """
        inputs.append(input_data)
        results = input_data[["obs_pmw"]].copy(deep=True)
        results = results[{"channels_pmw": 0}].rename(
            obs_pmw="surface_precip"
        )
        return results

    results = process_scene_spatial(
        input_data,
        tile_size=(64, 64),
        overlap=16,
        batch_size=8,
        retrieval_fn=retrieval_fn
    )

    for inpt in inputs:
        if "scans" in inpt.dims:
            assert inpt.scans.size == 64
        if "pixels" in inpt.dims:
            assert inpt.pixels.size == 64
        if "latitude" in inpt.dims:
            assert inpt.latitude.size == 64
        if "longitude" in inpt.dims:
            assert inpt.longitude.size == 64
        assert "batch" in inpt.dims

    if "scans" in input_data.dims:
        assert input_data.scans.size == results.scans.size
    if "pixels" in input_data.dims:
        assert input_data.pixels.size == results.pixels.size
    if "latitude" in input_data.dims:
        assert input_data.latitude.size == results.latitude.size
    if "longitude" in input_data.dims:
        assert input_data.longitude.size == results.longitude.size


@pytest.mark.parametrize(
    "input_data_fixture",
    ["input_data_gridded", "input_data_on_swath"]
)
def test_process_tiled_batched(input_data_fixture, request):
    """
    Test evaluation with a given tile size and ensure that input data
    passed to retrieval_fn has the expected tile size and that is is
    batched.
    """

    input_data = request.getfixturevalue(input_data_fixture)

    inputs = []
    def retrieval_fn(input_data):
        """
        Dummy retrieval function that stores all input in the 'inputs' variable.
        """
        inputs.append(input_data)
        results = input_data[["obs_pmw"]].copy(deep=True)
        results = results[{"channels_pmw": 0}].rename(
            obs_pmw="surface_precip"
        )
        return results

    results = process_scene_spatial(
        input_data,
        tile_size=(64, 64),
        overlap=16,
        batch_size=8,
        retrieval_fn=retrieval_fn
    )

    for inpt in inputs:
        if "scans" in inpt.dims:
            assert inpt.scans.size == 64
        if "pixels" in inpt.dims:
            assert inpt.pixels.size == 64
        if "latitude" in inpt.dims:
            assert inpt.sizes["latitude"] == 64
        if "longitude" in inpt.dims:
            assert inpt.sizes["longitude"] == 64
        assert "batch" in inpt.dims

    if "scans" in input_data.dims:
        assert input_data.scans.size == results.scans.size
    if "pixels" in input_data.dims:
        assert input_data.pixels.size == results.pixels.size
    if "latitude" in input_data.dims:
        assert input_data.latitude.size == results.latitude.size
    if "longitude" in input_data.dims:
        assert input_data.longitude.size == results.longitude.size


@pytest.mark.parametrize(
    "input_data_fixture",
    ["input_data_gridded", "input_data_on_swath"]
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
        results = input_data[["obs_pmw"]].copy(deep=True)
        results = results[{"channels_pmw": 0}].rename(
            obs_pmw="surface_precip"
        )
        return results

    results = process_scene_tabular(
        input_data,
        batch_size=256,
        retrieval_fn=retrieval_fn
    )

    for inpt in inputs[:-1]:
        assert inpt.samples.size == 256

    if "scans" in input_data.dims:
        assert input_data.scans.size == results.scans.size
    if "pixels" in input_data.dims:
        assert input_data.pixels.size == results.pixels.size
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
        ["pmw", "ancillary"],
        ipwgml_path=spr_gmi_evaluation,
        download=False
    )

    target_data = xr.load_dataset(evaluator.target_gridded[0])[["surface_precip"]]
    if geometry == "on_swath":
        input_data = xr.load_dataset(evaluator.ancillary_on_swath[0])
        target_data = target_data.interp(
            latitude=input_data.latitude,
            longitude=input_data.longitude,
            method="nearest"
        )

    def retrieval_fn(*args):
        return target_data

    metrics = [
        Bias(),
        MSE(),
        CorrelationCoef(),
    ]

    evaluator.evaluate_scene(
        0, None, None, None, retrieval_fn, "spatial",
        quantification_metrics=metrics,
        detection_metrics=[],
        probabilistic_detection_metrics=[],
        output_path=tmp_path
    )

    files = list(tmp_path.glob("results_*.nc"))
    assert len(files) > 0

    bias = metrics[0].compute()
    assert bias.bias.data == 0.0

    mse = metrics[1].compute()
    assert mse.mse.data == 0.0

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
        ["pmw", "ancillary"],
        ipwgml_path=spr_gmi_evaluation,
        download=False
    )

    metrics = evaluator.quantification_metrics
    for metric in metrics:
        assert isinstance(metric, Metric)

    evaluator.quantification_metrics = ["Bias"]
    metrics = evaluator.quantification_metrics
    assert len(metrics) == 1
    assert isinstance(metrics[0], Bias)

    metric = Bias()
    evaluator.quantification_metrics = [metric]
    metrics = evaluator.quantification_metrics
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
        ["pmw", "ancillary"],
        ipwgml_path=spr_gmi_evaluation,
        download=False
    )

    retrieval_fn = RetrievalFn(evaluator.target_gridded[0])


    evaluator.evaluate(
        None, None, None, retrieval_fn, "spatial",
        n_processes=2,
        output_path=tmp_path
    )

    files = list(tmp_path.glob("results_*.nc"))
    assert len(files) > 0
