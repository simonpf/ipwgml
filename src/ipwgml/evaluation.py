"""
ipwgml.evaluation
=================

Evaluation functionality for the ipwgml SPR dataset.

This module provides the ``Evaluator`` class that implements a generic
retrieval evaluator based on the test data split of the IPWG SPR dataset.

The evaluator takes care of downloading the data and loading it in the
format required by the retrieval. The interface to the retrieval has
to be provided in the form of a retrieval callback function. The
evaluator calls this function with an ``xarray.Dataset`` containing
the retrieval input data in the format requested by the user and
expects the retrieval callback function to return the corresponding
retrieval results. The evaluator than assess the results against
the reference estimates using various metrics.


Members
-------
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import copy
from dataclasses import dataclass
from datetime import datetime
import logging
from math import trunc, ceil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rich.progress import Progress, track
import xarray as xr

from ipwgml import baselines
from ipwgml import config
from ipwgml.data import download_missing
import ipwgml.logging
import ipwgml.metrics
from ipwgml.plotting import cmap_precip
from ipwgml.metrics import Metric
from ipwgml.tiling import DatasetTiler
from ipwgml.input import InputConfig, parse_retrieval_inputs
from ipwgml.target import TargetConfig


LOGGER = logging.getLogger(__name__)


def get_expected_dims(input_data: xr.Dataset) -> Tuple[str]:
    """
    Given an xarray.Dataset containing the retrieval input data, calculate
    the expected dimensions in the retrieval results.
    """
    if "latitude" in input_data.dims:
        spatial_dims = ("latitude", "longitude")
    elif "scan" in input_data.dims:
        spatial_dims = ("scan", "pixel")
    else:
        spatial_dims = ()

    if "batch" in input_data.dims:
        dims = ("batch",) + spatial_dims
    else:
        dims = spatial_dims

    return dims


def _check_retrieval_results(
    input_data: xr.Dataset,
    retrieved: xr.Dataset,
    expected_dims: List[str],
    verbose: bool = False,
) -> None:
    """
    Check retrieval results returned from 'retrieval_fn'.

    Args:
        input_data: An xarray.Dataset containing the retrieval input data.
        results: The retrieval results returned from the 'retrieval_fn'
        expected_dims: A list containing the expected dimensions of the retrieval
            results.
        verbose: If 'True' will warn about missing or extra retrieval results.
    """
    if set(expected_dims) != set(retrieved.dims):
        msg = (
            "Results returned from 'retrieval_fn' should have the same "
            f"dimenions as the input data ({expected_dims}) but have "
            f"dimensions {list(retrieved.dims)}."
        )
        raise RuntimeError(msg)

    for dim in expected_dims:
        if retrieved.sizes[dim] != input_data.sizes[dim]:
            msg = (
                f"The extent ({retrieved.sizes[dim]}) of retrieval results "
                f" along dimensions '{dim}' is inconsistent with the input "
                f"data ({input_data.sizes[dim]})."
            )
            raise RuntimeError(msg)

    expected = [
        "surface_precip",
        "probability_of_precip",
        "precip_flag",
        "probability_of_heavy_precipiation" "heavy_precip_flag",
    ]
    missing = []
    for var in expected:
        if var not in retrieved:
            missing.append(var)

    if len(missing) > 0:
        msg = (
            f"The retrieval results returned by 'callback_fn' are missing the expected results "
            f"for {missing}."
        )
        if verbose:
            LOGGER.warning(msg)

    extra = []
    for var in retrieved.variables:
        if var not in expected:
            extra.append(var)
    if len(extra) > 0:
        msg = (
            f"The retrieval results returned by 'callback_fn' contain unsupported "
            f"variables {extra}. They will be ignored."
        )
        if verbose:
            LOGGER.warning(msg)


def process(
    retrieval_fn: Callable[[xr.Dataset], xr.Dataset],
    input_data: xr.Dataset,
    coords: Tuple[int, int],
    result_tiler: DatasetTiler,
) -> List[str]:
    """
    Performs the retrieval on a single tile adds the retrieval
    results to the corresponding result tile.

    Args:
        retrieval_fn: The retrieval callback function.
        input_data: An xarray.Dataset containing the tiled retrieval
            input data.
        coords: A tuple containing the row- and column-index of
            the tile that is being processed.
        result_tiler: The tiler providing access to the result
            dataset.

    Return:
        A list containing the names of the retrieval variables that
        were present in the output from the retrieval callback
        function.
    """
    retrieved = retrieval_fn(input_data)
    expected_dims = get_expected_dims(input_data)
    _check_retrieval_results(input_data, retrieved, expected_dims)
    retrieved = retrieved.transpose(*expected_dims, ...)

    results_t = result_tiler.get_tile(*coords)
    weights = result_tiler.get_weights(*coords)
    slcs = result_tiler.get_slices(*coords)

    vars_retrieved = []
    for var in [
        "surface_precip",
        "probability_of_precip",
        "probability_of_heavy_precip",
    ]:
        if var in retrieved:
            results_t[var].data += weights * retrieved[var]
            vars_retrieved.append(var)

    for var in ["precip_flag", "heavy_precip_flag"]:
        if var in retrieved:
            results_t[var][slcs].data[:] = retrieved[var][slcs].data
            vars_retrieved.append(var)

    return vars_retrieved


def process_batched(
    retrieval_fn: Callable[[xr.Dataset], xr.Dataset],
    input_data: List[xr.Dataset],
    spatial_dims: List[str],
    coords: List[Tuple[int, int]],
    result_tiler: DatasetTiler,
) -> List[str]:
    """
    Performs the retrieval on a batch of input data tiles and
    adds the retrieval results to the corresponding result tiles.

    Args:
        retrieval_fn: The retrieval callback function.
        input_data: An xarray.Dataset containing the a batch of input
            data tiles.
        coords: A tuple containing the row- and column-index of
            the tile that is being processed.
        result_tiler: The tiler providing access to the result
            dataset.

    Return:
        A list containing the names of the retrieval variables that
        were present in the output from the retrieval callback
        function.
    """
    batch_size = len(input_data)
    if any([dim in input_data[0].coords for dim in spatial_dims]):
        input_data = [inpt.reset_index(spatial_dims) for inpt in input_data]
    input_data = xr.concat(input_data, dim="batch")
    retrieved_batched = retrieval_fn(input_data)
    expected_dims = get_expected_dims(input_data)
    _check_retrieval_results(input_data, retrieved_batched, expected_dims)
    retrieved_batched = retrieved_batched.transpose(*expected_dims, ...)

    for batch_ind in range(batch_size):

        vars_retrieved = []

        retrieved = retrieved_batched[{"batch": batch_ind}]
        results_t = result_tiler.get_tile(*coords[batch_ind])
        weights = result_tiler.get_weights(*coords[batch_ind])
        slcs = result_tiler.get_slices(*coords[batch_ind])

        for var in [
            "surface_precip",
            "probability_of_precip",
            "probability_of_heavy_precip",
        ]:
            if var in retrieved:
                results_t[var].data += weights * retrieved[var]
                vars_retrieved.append(var)

        for var in ["precip_flag", "heavy_precip_flag"]:
            if var in retrieved:
                results_t[var][slcs].data[:] = retrieved[var][slcs].data
                vars_retrieved.append(var)
    return vars_retrieved


def load_retrieval_input_data(
    input_files: "InputFiles",
    retrieval_input: List[InputConfig],
    geometry: str,
) -> xr.Dataset:
    """
    Load retrieval input data.

    Args:
        input_files: A InputFiles dataclass object specifying the input files for the
            given collocation.
        retrieval_input: List of the retrieval inputs.
        geometry: The type of data to load: "on_swath" or "gridded".

    Return:
        An xarray.Dataset containing the input data from the sources
        specified in 'retrieval_input' as well as the latitude and
        longitude coordinates and meaurements times of the reference
        precipitation estimates.
    """
    if geometry == "on_swath":
        spatial_dims = ("scan", "pixel")
    else:
        spatial_dims = ("latitude", "longitude")

    input_data = xr.Dataset()

    # Load time from target file.
    target_file = input_files.get_path("target", geometry)
    with xr.open_dataset(target_file) as target_data:
        input_data["time"] = (spatial_dims, target_data.time.data)

    for inpt in retrieval_input:
        path = input_files.get_path(inpt.name, geometry)
        if path is not None:
            if inpt.name == "ancillary":
                dims = (f"ancillary_features",) + spatial_dims
            else:
                dims = (f"channels_{inpt.name}",) + spatial_dims
            data = inpt.load_data(path, target_time=input_data.time)
            for name, arr in data.items():
                input_data[name] = dims, arr

    anc_file = input_files.get_path("ancillary", geometry)
    with xr.open_dataset(anc_file) as anc_data:
        for name, attr in anc_data.attrs.items():
            if name == "pmw_input_file":
                input_data.attrs["gpm_input_file"] = attr
            else:
                input_data.attrs[name] = attr

    ancillary_file = input_files.get_path("ancillary", geometry)
    ancillary_data = xr.load_dataset(ancillary_file)
    if "latitude" not in ancillary_data.dims:
        input_data["latitude"] = (spatial_dims, ancillary_data.latitude.data)
        input_data["longitude"] = (spatial_dims, ancillary_data.longitude.data)
    else:
        input_data["latitude"] = ancillary_data.latitude
        input_data["longitude"] = ancillary_data.longitude

    return input_data


def process_scene_spatial(
    input_data: xr.Dataset,
    tile_size: int | Tuple[int, int] | None,
    overlap: int | None,
    batch_size: int | None,
    retrieval_fn: Callable[[xr.Dataset], xr.Dataset],
) -> xr.Dataset:
    """
    Process an overpass scene using a given retrieval callback function
    for an image-based retrieval.

    This function takes care of tiling and potentially batching of the
    input scenes.

    Args:
        input_data: An xarray.Dataset containing all required input data for
            the scene.
        tile_size: The tile size expected by the retrieval function. Set to
            'None' provide full scene as input data.
        overlap: The overlap between neighboring tiles.
        batch_size: The batch size expected by the retrieval function.
        retrieval_fn: The retrieval callback function to use to evaluate
            the retrieval on the input data.

    Return:
        An xarray.Dataset containing the assembled retrieval results
        for the given input scene.
    """
    spatial_dims = ["latitude", "longitude", "scan", "pixel"]
    spatial_dims = [dim for dim in spatial_dims if dim in input_data.dims]
    shape = tuple([input_data[dim].size for dim in spatial_dims])

    if isinstance(tile_size, int):
        tile_size = (tile_size,) * 2
    if overlap is None:
        if tile_size is None:
            overlap = 0
        else:
            overlap = min(tile_size) // 4

    input_data_tiler = DatasetTiler(
        input_data, tile_size=tile_size, overlap=overlap, spatial_dims=spatial_dims
    )

    if batch_size is None:
        batched = False
        batch_size = 1
    else:
        batched = True

    # Intialize container for results.
    results = xr.Dataset(
        {
            spatial_dims[0]: (spatial_dims[0], input_data[spatial_dims[0]].data),
            spatial_dims[1]: (spatial_dims[1], input_data[spatial_dims[1]].data),
            "surface_precip": (spatial_dims, np.zeros(shape, dtype=np.float32)),
            "probability_of_precip": (spatial_dims, np.zeros(shape, dtype=np.float32)),
            "probability_of_heavy_precip": (
                spatial_dims,
                np.zeros(shape, dtype=np.float32),
            ),
            "precip_flag": (spatial_dims, np.zeros(shape, dtype=bool)),
            "heavy_precip_flag": (spatial_dims, np.zeros(shape, dtype=bool)),
        }
    )

    result_tiler = DatasetTiler(
        results, tile_size=tile_size, overlap=overlap, spatial_dims=spatial_dims
    )

    batch_stack = []
    coord_stack = []

    for row_ind in range(input_data_tiler.n_rows_tiled):
        for col_ind in range(input_data_tiler.n_cols_tiled):
            input_tile = input_data_tiler.get_tile(row_ind, col_ind)
            batch_stack.append(input_tile)
            coord_stack.append((row_ind, col_ind))

            while len(batch_stack) >= batch_size:
                batch = batch_stack[:batch_size]
                batch_stack = batch_stack[batch_size:]
                coords = coord_stack[:batch_size]
                coord_stack = coord_stack[batch_size:]
                if batched:
                    assert len(batch) == batch_size
                    assert len(coords) == batch_size
                    vars_retrieved = process_batched(
                        retrieval_fn, batch, spatial_dims, coords, result_tiler
                    )
                else:
                    assert len(batch) == 1
                    assert len(coords) == 1
                    vars_retrieved = process(
                        retrieval_fn, batch[0], coords[0], result_tiler
                    )

    # Process remaining tiles.
    if len(batch_stack) > 0:
        vars_retrieved = process_batched(
            retrieval_fn, batch_stack, spatial_dims, coord_stack, result_tiler
        )

    return results[vars_retrieved]


def process_scene_tabular(
    input_data: xr.Dataset,
    batch_size: int | None,
    retrieval_fn: Callable[[xr.Dataset], xr.Dataset],
) -> xr.Dataset:
    """
    Process a collocation scene with input data in tabular format.

    Args:
        input_data: An xarary.Dataset containing the retrieval input data.
        batch_size: The batch size to use for processing.
        retrieval_fn: The retrieval callback function.

    Return:
        An xarray.Dataset containing the retrieval results reshaped
        into their original 2D structure.
    """
    spatial_dims = ["latitude", "longitude", "scan", "pixel"]
    spatial_dims = [dim for dim in spatial_dims if dim in input_data.dims]
    shape = tuple([input_data[dim].size for dim in spatial_dims])

    input_data_flat = input_data.stack({"batch": spatial_dims}).copy(deep=True)
    n_samples = input_data_flat.batch.size
    if batch_size is None:
        batch_size = n_samples

    input_data_flat["surface_precip"] = (
        ("batch",),
        np.zeros(n_samples, dtype=np.float32),
    )
    input_data_flat["probability_of_precip"] = (
        ("batch",),
        np.zeros(n_samples, dtype=np.float32),
    )
    input_data_flat["probability_of_heavy_precip"] = (
        ("batch",),
        np.zeros(n_samples, dtype=np.float32),
    )
    input_data_flat["precip_flag"] = (("batch",), np.zeros(n_samples, dtype=bool))
    input_data_flat["heavy_precip_flag"] = (
        ("batch",),
        np.zeros(n_samples, dtype=bool),
    )

    batch_start = 0
    vars_retrieved = []
    while batch_start < n_samples:
        inds = {"batch": slice(batch_start, batch_start + batch_size)}
        batch = input_data_flat[inds]
        retrieved = retrieval_fn(batch)
        for var in [
            "surface_precip",
            "probability_of_precip",
            "probability_of_heavy_precip",
            "precip_flag",
            "heavy_precip_flag",
        ]:
            if var in retrieved:
                batch[var].data[:] = retrieved[var].data
                vars_retrieved.append(var)

        batch_start += batch_size

    results = input_data_flat[vars_retrieved].unstack()
    return results


@dataclass
class InputFiles:
    """
    Helper class that holds the input files required for evaluation.

    """

    target_file_gridded: Path
    target_file_on_swath: Path
    gmi_file_gridded: Path
    gmi_file_on_swath: Path
    atms_file_gridded: Path
    atms_file_on_swath: Path
    ancillary_file_gridded: Path
    ancillary_file_on_swath: Path
    geo_file_gridded: Optional[Path]
    geo_file_on_swath: Optional[Path]
    geo_ir_file_gridded: Optional[Path]
    geo_ir_file_on_swath: Optional[Path]

    def get_path(self, name: str, geometry: str) -> Path | None:
        """
        Get path to input data file for given input and geometry.

        Args:
            name: The name of the input.
            geometry: A string specifying the geometry: 'on_swath' or 'gridded'.

        Return:
            A Path object pointing to the input file to load or None.
        """
        if name not in ["target", "gmi", "atms", "ancillary", "geo_ir", "geo"]:
            raise ValueError(
                "'name' must be one of the supported input datasets ('target', "
                "'gmi', 'atms', 'ancillary', 'geo_ir', 'geo')"
            )
        if geometry not in ["on_swath", "gridded"]:
            raise ValueError(
                "'geometry' must be one of the supported geometries ('on_swath', "
                "'gridded')."
            )
        return getattr(self, f"{name}_file_{geometry}")


def evaluate_scene(
    input_files: InputFiles,
    retrieval_input: List[InputConfig],
    target_config: TargetConfig,
    geometry: str,
    tile_size: int | Tuple[int, int] | None,
    overlap: int | None,
    batch_size: int | None,
    retrieval_fn: Callable[[xr.Dataset], xr.Dataset],
    input_data_format: str,
    precip_quantification_metrics: List[Metric],
    precip_detection_metrics: List[Metric],
    prob_precip_detection_metrics: List[Metric],
    heavy_precip_detection_metrics: List[Metric],
    prob_heavy_precip_detection_metrics: List[Metric],
    output_path: Optional[Path] = None,
) -> xr.Dataset:
    """
    Evaluate retrieval on a single collocation file.

    Args:
        input_files: An input files record containing the paths to all retrieval
            input files.
        retrieval_input: A list defining the retrieval inputs to load.
        target_config: An optional TargetConfig specifying quality requirements for the retrieval
                target data to load.
        geometry: A string defining the geometry of the retrieval: 'on_swath' or
            'gridded'.
        tile_size: The tile size to use for the retrieval or 'None' if no tiling
            should be applied.
        overlap: The overlap to apply for the tiling.
        batch_size: If not 'None', inputs to 'retrieval_fn' will be batched
            using the given batch size. This only has an effect for
            tabular and spatial retrievals with tiling. Batches may include
            less samples than the batch size.
        retrieval_fn: A callback function that runs the retrieval on the
            input data.
        input_data_format: A string specifying whether the retrieval expects input data in
            spatial or tabular format.
        precip_quantification_metrics: A list containing the metrics to use
            to evaluate quantitative precipitation estimates.
        precip_detection_metrics: A list containing the metrics to use to evaluate
            the precipitation detection.
        prob_precip_detection_metrics: A list containing the metrics to use
            to evaluate the probabilistic precipitation detection.
        heavy_precip_detection_metrics: A list containing the metrics to use
            to evaluate the heavy precipitation detection.
        prob_heavy_precip_detection_metrics: A list containing the metrics
            to use to evaluate the probabilistic heavy precipitation detection.
        output_path: If given the retrieval results from the scene will be written
            to this path.
    """
    input_data = load_retrieval_input_data(
        input_files=input_files, retrieval_input=retrieval_input, geometry=geometry
    )

    if input_data_format == "spatial":
        results = process_scene_spatial(
            input_data=input_data,
            tile_size=tile_size,
            overlap=overlap,
            batch_size=batch_size,
            retrieval_fn=retrieval_fn,
        )
    else:
        results = process_scene_tabular(
            input_data=input_data, batch_size=batch_size, retrieval_fn=retrieval_fn
        )

    with xr.open_dataset(input_files.target_file_gridded) as target_data:

        scan_inds = target_data.scan_index
        pixel_inds = target_data.pixel_index

        if geometry == "on_swath":
            if "latitude" in results:
                results = results.drop_vars(["latitude", "longitude"])
            results = results[{"scan": scan_inds, "pixel": pixel_inds}]
            invalid = pixel_inds.data < 0
            for var in [
                "surface_precip",
                "probability_of_precip",
                "probability_of_heavy_precip",
            ]:
                if var in results:
                    results[var].data[invalid] = np.nan

        surface_precip_ref = target_config.load_reference_precip(target_data)
        invalid_mask = target_config.get_mask(target_data)
        valid_mask = (
            (pixel_inds.data >= 0)
            * np.isfinite(results.surface_precip.data)
            * ~invalid_mask
        )
        surface_precip_ref = target_data.surface_precip
        surface_precip_ref.data[~valid_mask] = np.nan

        for metric in precip_quantification_metrics:
            metric.update(results.surface_precip.data, surface_precip_ref.data)

        precip_flag_ref = None
        if "precip_flag" in results:
            precip_flag_ref = target_config.load_precip_mask(target_data)
            for metric in precip_detection_metrics:
                metric.update(
                    results.precip_flag.data[valid_mask], precip_flag_ref[valid_mask]
                )
        if "probability_of_precip" in results:
            if precip_flag_ref is None:
                precip_flag_ref = target_config.load_precip_mask(target_data)
            for metric in prob_precip_detection_metrics:
                metric.update(
                    results.probability_of_precip.data[valid_mask],
                    precip_flag_ref[valid_mask],
                )

        heavy_precip_flag_ref = None
        if "heavy_precip_flag" in results:
            heavy_precip_flag_ref = target_config.load_heavy_precip_mask(target_data)
            for metric in heavy_precip_detection_metrics:
                metric.update(
                    results.heavy_precip_flag.data[valid_mask],
                    heavy_precip_flag_ref[valid_mask],
                )
        if "probability_of_heavy_precip" in results:
            if heavy_precip_flag_ref is None:
                heavy_precip_flag_ref = target_config.load_heavy_precip_mask(
                    target_data
                )
            for metric in prob_heavy_precip_detection_metrics:
                metric.update(
                    results.probability_of_heavy_precip.data[valid_mask],
                    heavy_precip_flag_ref[valid_mask],
                )

        aux_vars = [
            "radar_quality_index",
            "valid_fraction",
            "precip_fraction",
            "snow_fraction",
            "convective_fraction",
            "stratiform_fraction",
            "hail_fraction",
        ]

        results["surface_precip_ref"] = surface_precip_ref
        for var in aux_vars:
            results[var] = target_data[var]

        if output_path is not None:
            output_path = Path(output_path)
            output_path.mkdir(exist_ok=True, parents=True)
            median_time = input_files.target_file_gridded.name.split("_")[1][:-3]
            results.to_netcdf(output_path / f"results_{median_time}.nc")

        return results


class Evaluator:
    """
    The Evaluator class provides an interface to evaluate a generic retrieval implemented
    by a retrieval callback function using the IPWG SPR dataset.
    """

    def __init__(
        self,
        sensor: str,
        geometry: str,
        retrieval_input: Optional[List[str | Dict[str, Any | InputConfig]]] = None,
        target_config=None,
        ipwgml_path: Optional[Path] = None,
        download: bool = True,
    ):
        """
        Args:
            sensor: The name of SPR sensor
            geometry: The geometry of  the retrieval. 'gridded' for retrievals operating on
                the regridded input observations; 'on_swath' for retrievals operating on the
                nativ swath-based observations.
            retrieval_input: The retrieval inputs to load. Should be a subset of
                ['gmi', 'mhs', 'ancillary', 'geo', 'geo_ir']
            ipwgml_path: An optional path to the location of the ipgml data.
            download: A boolean flag indicating whether or not to download the evaluation files
                 if they are not found in 'ipwgml_path'.
        """
        if ipwgml_path is None:
            ipwgml_path = config.get_data_path()
        else:
            ipwgml_path = Path(ipwgml_path)

        self.sensor = sensor
        self.geometry = geometry

        if retrieval_input is None:
            retrieval_input = ALL_INPUTS
        self.retrieval_input = parse_retrieval_inputs(retrieval_input)

        if target_config is None:
            target_config = TargetConfig()
        self.target_config = target_config

        self.ipwgml_path = ipwgml_path

        self._precip_quantification_metrics = [
            ipwgml.metrics.Bias(),
            ipwgml.metrics.MAE(),
            ipwgml.metrics.MSE(),
            ipwgml.metrics.SMAPE(),
            ipwgml.metrics.CorrelationCoef(),
            ipwgml.metrics.SpectralCoherence(),
        ]
        self._precip_detection_metrics = [
            ipwgml.metrics.POD(),
            ipwgml.metrics.FAR(),
            ipwgml.metrics.HSS(),
        ]
        self._prob_precip_detection_metrics = [ipwgml.metrics.PRCurve()]
        self._heavy_precip_detection_metrics = [
            ipwgml.metrics.POD(),
            ipwgml.metrics.FAR(),
            ipwgml.metrics.HSS(),
        ]
        self._prob_heavy_precip_detection_metrics = [ipwgml.metrics.PRCurve()]

        for geometry in ["gridded", "on_swath"]:
            dataset = f"spr/{self.sensor}/evaluation/{geometry}/"
            for inpt in self.retrieval_input:
                if download:
                    download_missing(
                        dataset + inpt.name, ipwgml_path, progress_bar=True
                    )
                files = sorted(list((ipwgml_path / dataset / inpt.name).glob("*.nc")))
                setattr(self, inpt.name + "_" + geometry, files)

            if getattr(self, f"ancillary_{geometry}", None) is None:
                if download:
                    download_missing(
                        dataset + "ancillary", ipwgml_path, progress_bar=True
                    )
                files = sorted(list((ipwgml_path / dataset / "ancillary").glob("*.nc")))
                setattr(self, "ancillary" + "_" + geometry, files)

            if download:
                download_missing(dataset + "target", ipwgml_path, progress_bar=True)
            files = sorted(list((ipwgml_path / dataset / "target").glob("*.nc")))
            setattr(self, "target_" + geometry, files)

    @property
    def precip_quantification_metrics(self):
        """
        List containing the metrics used to evaluate quantiative precipitation estimates.
        """
        return self._precip_quantification_metrics

    @precip_quantification_metrics.setter
    def precip_quantification_metrics(self, metrics: List[str | Metric]):
        """
        Setter for the 'quantification_metrics' property.
        """
        parsed = []
        for metric in metrics:
            if isinstance(metric, str):
                metric_class = getattr(ipwgml.metrics, metric, None)
                if metric_class is None or type(metric_class) != type:
                    raise ValueError(
                        f"The metric '{metric}' is not known. Please refer to the"
                        f"documentation of the 'ipwgml.metrics' module for available "
                        "metrics."
                    )
                metric = metric_class()
            parsed.append(metric)
        self._precip_quantification_metrics = parsed

    @property
    def precip_detection_metrics(self):
        """
        List containing the metrics used to evaluate precipitation detection.
        """
        return self._precip_detection_metrics

    @precip_detection_metrics.setter
    def set_detection_metric(self, metrics: List[str | Metric]):
        """
        Setter for the 'detection_metrics' property.
        """
        parsed = []
        for metric in metrics:
            if isinstance(metric, str):
                metric_class = getattr(metrics, metric, None)
                if metric_class is None or type(metric_class) != type:
                    raise ValueError(
                        f"The metric '{metric}' is not known. Please refer to the"
                        f"documentation of the 'ipwgml.metrics' module for available "
                        "metrics."
                    )
                metric = metric_class()
            parsed.append(metric)
        self._precip_detection_metrics = metrics

    @property
    def prob_precip_detection_metrics(self):
        """
        List containing the metrics used to evaluate precipitation detection.
        """
        return self._prob_precip_detection_metrics

    @prob_precip_detection_metrics.setter
    def set_prob_precip_detection_metrics(self, metrics: List[str | Metric]):
        """
        Setter for the 'probabilistic_detection_metrics' property.
        """
        parsed = []
        for metric in metrics:
            if isinstance(metric, str):
                metric_class = getattr(metrics, metric, None)
                if metric_class is None or type(metric_class) != type:
                    raise ValueError(
                        f"The metric '{metric}' is not known. Please refer to the"
                        f"documentation of the 'ipwgml.metrics' module for available "
                        "metrics."
                    )
                metric = metric_class()
            parsed.append(metric)
        self._prob_precip_detection_metrics = metrics

    @property
    def heavy_precip_detection_metrics(self):
        """
        List containing the metrics used to evaluate the detection of heavy precipitation.
        """
        return self._heavy_precip_detection_metrics

    @heavy_precip_detection_metrics.setter
    def set_heavy_precip_detection_metrics(self, metrics: List[str | Metric]):
        """
        Setter for the 'heavy_precip_detection_metrics' property.
        """
        parsed = []
        for metric in metrics:
            if isinstance(metric, str):
                metric_class = getattr(metrics, metric, None)
                if metric_class is None or type(metric_class) != type:
                    raise ValueError(
                        f"The metric '{metric}' is not known. Please refer to the"
                        f"documentation of the 'ipwgml.metrics' module for available "
                        "metrics."
                    )
                metric = metric_class()
            parsed.append(metric)
        self._heavy_precip_detection_metrics = metrics

    @property
    def prob_heavy_precip_detection_metrics(self):
        """
        List containing the metrics used to evaluate the probabilistic detection of heavy
        precipitation.
        """
        return self._prob_heavy_precip_detection_metrics

    @prob_precip_detection_metrics.setter
    def set_prob_heavy_precip_detection_metrics(self, metrics: List[str | Metric]):
        """
        Setter for the 'prob_heavy_precip_detection_metrics' property.
        """
        parsed = []
        for metric in metrics:
            if isinstance(metric, str):
                metric_class = getattr(metrics, metric, None)
                if metric_class is None or type(metric_class) != type:
                    raise ValueError(
                        f"The metric '{metric}' is not known. Please refer to the"
                        f"documentation of the 'ipwgml.metrics' module for available "
                        "metrics."
                    )
                metric = metric_class()
            parsed.append(metric)
        self._prob_heavy_precip_detection_metrics = metrics

    def __repr__(self):
        return (
            f"Evaluator(sensor='{self.sensor}', geometry='{self.geometry}', "
            f"ipwgml_path='{self.ipwgml_path}')"
        )

    def __len__(self) -> int:
        """
        The number of collocations available for evaluation.
        """
        return len(self.target_gridded)

    def get_input_files(self, index: int) -> InputFiles:
        """
        Compile retrieval input and target files for a given collocation.

        Args:
            index: The collocation index.

        Return:
            An InputFiles object containing all available input and
            target files for the given collocation.
        """
        if len(self) <= index:
            raise IndexError("'index' exceeds number of availale collocation scenes.")
        return InputFiles(
            self.target_gridded[index],
            self.target_on_swath[index],
            self.gmi_gridded[index] if hasattr(self, "gmi_gridded") else None,
            self.gmi_on_swath[index] if hasattr(self, "gmi_on_swath") else None,
            self.atms_gridded[index] if hasattr(self, "atms_gridded") else None,
            self.atms_on_swath[index] if hasattr(self, "atms_on_swath") else None,
            (
                self.ancillary_gridded[index]
                if hasattr(self, "ancillary_gridded")
                else None
            ),
            (
                self.ancillary_on_swath[index]
                if hasattr(self, "ancillary_on_swath")
                else None
            ),
            self.geo_gridded[index] if hasattr(self, "geo_gridded") else None,
            self.geo_on_swath[index] if hasattr(self, "geo_on_swath") else None,
            self.geo_ir_gridded[index] if hasattr(self, "geo_ir_gridded") else None,
            self.geo_ir_on_swath[index] if hasattr(self, "geo_ir_on_swath") else None,
        )

    def get_input_data(self, scene_index: int) -> xr.Dataset:
        """
        Get retrieval input data for a given scene.

        Args:
            scene_index: An integer specifying the scene for which to load the input data.

        Return:
            An xarray.Dataset containing the retrieval input data.
        """
        input_files = self.get_input_files(scene_index)
        input_data = load_retrieval_input_data(
            input_files=input_files,
            retrieval_input=self.retrieval_input,
            geometry=self.geometry,
        )
        return input_data

    def evaluate_scene(
        self,
        index: int,
        tile_size: int | Tuple[int, int] | None,
        overlap: int | None,
        batch_size: int | None,
        retrieval_fn: Callable[[xr.Dataset], xr.Dataset],
        input_data_format: str,
        track: bool = False,
        output_path: Optional[Path] = None,
    ) -> xr.Dataset:
        """
        Run evaluation on a single scene.

        Args:
            index: An index identifying the scene.
            tile_size: The tile size to use for the retrieval or 'None' to apply no tiling.
            overlap: The overlap to apply for the tiling.
            batch_size: Maximum batch size for tiled spatial and tabular retrievals.
            retrieval_fn: The retrieval callback function.
            input_data_format: Whether the retrieval expects input data in 'tabular' or 'spatial'
                format.
            track: If 'True' will track the retrieval results using the
                evaluator's metrics. If 'False', results will not be tracked.
            output_path: If not 'None', retrieval results will be written to that path.

        Return:
            An xarray.Dataset containing the retrieval results.
        """
        if track:
            precip_quantification_metrics = self.precip_quantification_metrics
            precip_detection_metrics = self.precip_detection_metrics
            prob_precip_detection_metrics = self.prob_precip_detection_metrics
            heavy_precip_detection_metrics = self.heavy_precip_detection_metrics
            prob_heavy_precip_detection_metrics = (
                self.prob_heavy_precip_detection_metrics
            )
        else:
            precip_quantification_metrics = []
            precip_detection_metrics = []
            prob_precip_detection_metrics = []
            heavy_precip_detection_metrics = []
            prob_heavy_precip_detection_metrics = []

        return evaluate_scene(
            input_files=self.get_input_files(index),
            retrieval_input=self.retrieval_input,
            target_config=self.target_config,
            geometry=self.geometry,
            tile_size=tile_size,
            overlap=overlap,
            batch_size=batch_size,
            retrieval_fn=retrieval_fn,
            input_data_format=input_data_format,
            precip_quantification_metrics=precip_quantification_metrics,
            precip_detection_metrics=precip_detection_metrics,
            prob_precip_detection_metrics=prob_precip_detection_metrics,
            heavy_precip_detection_metrics=heavy_precip_detection_metrics,
            prob_heavy_precip_detection_metrics=prob_heavy_precip_detection_metrics,
            output_path=output_path,
        )

    def evaluate(
        self,
        retrieval_fn: Callable[[xr.Dataset], xr.Dataset],
        tile_size: int | Tuple[int, int] | None = None,
        overlap: int | None = None,
        batch_size: int | None = None,
        input_data_format: str = "spatial",
        n_processes: int | None = None,
        output_path: Optional[Path] = None,
    ):
        """
        Run evaluation on complete evaluation dataset.

        Args:
            retrieval_fn: The retrieval callback function.
            tile_size: The tile size to use for the retrieval or 'None' to apply no tiling.
            overlap: The overlap to apply for the tiling.
            batch_size: Maximum batch size for tiled spatial and tabular retrievals.
            input_data_format: The retrieval kind: 'spatial' or 'tabular'.
            output_path: If not 'None', retrieval results will be written to that path.
        """
        precip_quantification_metrics = self.precip_quantification_metrics
        precip_detection_metrics = self.precip_detection_metrics
        prob_precip_detection_metrics = self.prob_precip_detection_metrics
        heavy_precip_detection_metrics = self.heavy_precip_detection_metrics
        prob_heavy_precip_detection_metrics = self.prob_heavy_precip_detection_metrics

        if n_processes is None or n_processes < 2:
            for scene_ind in track(
                range(len(self)),
                description="Evaluating retrieval",
                console=ipwgml.logging.get_console(),
            ):
                self.evaluate_scene(
                    index=scene_ind,
                    tile_size=tile_size,
                    overlap=overlap,
                    batch_size=batch_size,
                    retrieval_fn=retrieval_fn,
                    input_data_format=input_data_format,
                    track=True,
                    output_path=output_path,
                )
                try:
                    self.evaluate_scene(
                        index=scene_ind,
                        tile_size=tile_size,
                        overlap=overlap,
                        batch_size=batch_size,
                        retrieval_fn=retrieval_fn,
                        input_data_format=input_data_format,
                        track=True,
                        output_path=output_path,
                    )
                except Exception as exc:
                    raise exc
                    LOGGER.exception(
                        f"Encountered an error when processing scene {scene_ind}."
                    )
        else:
            pool = ProcessPoolExecutor(max_workers=n_processes)
            tasks = []
            scenes = {}
            for scene_ind in range(len(self)):
                tasks.append(
                    pool.submit(
                        self.evaluate_scene,
                        index=scene_ind,
                        tile_size=tile_size,
                        overlap=overlap,
                        batch_size=batch_size,
                        retrieval_fn=retrieval_fn,
                        input_data_format=input_data_format,
                        track=True,
                        output_path=output_path,
                    )
                )
                scenes[tasks[-1]] = scene_ind

            with Progress() as progress:
                evaluation = progress.add_task(
                    "Evaluating retrieval:", total=(len(tasks))
                )
                for task in as_completed(tasks):
                    try:
                        task.result()
                    except Exception:
                        LOGGER.exception(
                            f"Encountered an error when processing scene {scenes[task]}."
                        )
                    progress.update(evaluation, advance=1)

    def plot_retrieval_results(
        self,
        scene_index: int,
        retrieval_fn: Callable[[xr.Dataset], xr.Dataset],
        input_data_format: str = "spatial",
        tile_size: int | Tuple[int, int] | None = None,
        overlap: int | None = None,
        batch_size: int | None = None,
        swath_boundaries: bool = False,
    ) -> "plt.Figure":
        """
        Plot retrieval results for a given retrieval scene.

        Args:
            scene_index: An integer identifying the scene for which to plot the retrieval
                 results.
            retrieval_fn: The retrieval callback function.
            input_data_format: The retrieval kind: 'spatial' or 'tabular'.
            tile_size: The tile size to use for the retrieval or 'None' to apply no tiling.
            overlap: The overlap to apply for the tiling.
            batch_size: Maximum batch size for tiled spatial and tabular retrievals.
            swath_boundaries: If 'True' will plot swath boundaries of the GPM
                sensor.
        """
        try:
            from ipwgml.plotting import add_ticks
            import cartopy.crs as ccrs
            import matplotlib.pyplot as plt
            from matplotlib.colors import LogNorm
            from matplotlib.gridspec import GridSpec
        except ImportError:
            raise RuntimeError(
                "This function requires matplotlib and cartopy to be installed."
            )

        rqi_levels = [0.5, 0.9]

        results = self.evaluate_scene(
            index=scene_index,
            tile_size=tile_size,
            overlap=overlap,
            batch_size=batch_size,
            retrieval_fn=retrieval_fn,
            input_data_format=input_data_format,
            track=False,
        )

        fname = self.target_gridded[scene_index].name
        median_time = fname.split("_")[-1][:-3]
        date = datetime.strptime(median_time, "%Y%m%d%H%M%S")

        with xr.open_dataset(self.target_gridded[scene_index]) as target_data:
            lons = target_data.longitude.data
            lats = target_data.latitude.data
            surface_precip_full = target_data.surface_precip.data
            rqi = target_data.radar_quality_index

        sp_ret = results.surface_precip.data
        sp_ref = results.surface_precip_ref.data

        lon_ticks = np.arange(
            trunc(lons.min() // 5) * 5.0, ceil(lons.max() // 5) * 5 + 1.0, 5.0
        )
        lat_ticks = np.arange(
            trunc(lats.min() // 5) * 5.0, ceil(lats.max() // 5) * 5 + 1.0, 5.0
        )

        crs = ccrs.PlateCarree()
        fig = plt.figure(figsize=(12, 6))
        gs = GridSpec(2, 3, width_ratios=[1.0, 1.0, 0.075], height_ratios=[1.0, 0.1])
        norm = LogNorm(1e-1, 1e2)

        mask = np.isnan(sp_ref)

        ax = fig.add_subplot(gs[0, 0], projection=crs)
        ax.pcolormesh(lons, lats, np.maximum(sp_ret, 1e-3), cmap=cmap_precip, norm=norm)
        ax.contour(
            lons, lats, rqi, levels=rqi_levels, linestyles=["-", "--"], colors="grey"
        )
        ax.set_title("(a) Retrieved", loc="left")
        add_ticks(ax, lon_ticks, lat_ticks, left=True, bottom=True)
        ax.coastlines()

        if swath_boundaries:
            pixel_inds = target_data.pixel_index
            ax.contour(lons, lats, pixel_inds, levels=[-0.5], linestyles=["--"], colors=["k"])

        ax = fig.add_subplot(gs[0, 1], projection=crs)
        m = ax.pcolormesh(
            lons,
            lats,
            np.maximum(surface_precip_full, 1e-3),
            cmap=cmap_precip,
            norm=norm,
        )

        cntr = ax.contour(
            lons, lats, rqi, levels=rqi_levels, linestyles=["-", "--"], colors="grey"
        )
        ax.set_title("(b) Reference", loc="left")
        add_ticks(ax, lon_ticks, lat_ticks, left=False, bottom=True)
        ax.coastlines()
        if swath_boundaries:
            pixel_inds = target_data.pixel_index
            ax.contour(lons, lats, pixel_inds, levels=[-0.5], linestyles=["--"], colors=["k"])

        fig.suptitle(date.strftime("%Y-%m-%d %H:%M:%S"))

        cax = fig.add_subplot(gs[0, 2])
        plt.colorbar(m, cax=cax, label="Surface precipitation [mm h$^{-1}$]")

        handles, labels = cntr.legend_elements()
        labels = [label.replace("x", "RQI") for label in labels]
        ax = fig.add_subplot(gs[1, :])
        ax.set_axis_off()
        ax.legend(handles=handles, labels=labels, ncol=2, loc="center")

        return fig

    def get_precip_quantification_results(
        self, name: Optional[str] = None, include_baselines: bool = True
    ) -> pd.DataFrame:
        """
        Get scalar results from precipitation estimation metrics as pandas.Dataframe.

        Args:
            name: An optional name for the retrieval algorithm.
            include_baselines: If 'True', results from retrieval baselines will be included
                in the results.

        Return:
            A pandas.DataFrame containing the combined scalar results from
            the 'precip_quantification_metrics' of this Evaluator object.

        """
        results = []
        for metric in self.precip_quantification_metrics:
            res_m = metric.compute()
            drop = [var for var in res_m.variables if len(res_m[var].dims) > 0]
            results.append(res_m.drop_vars(drop))

        results = xr.merge(results).expand_dims("algorithm")
        results["algorithm"] = (("algorithm",), [name])

        if include_baselines:
            results_b = baselines.load_baseline_results(self.sensor)
            vars = list(results.variables.keys())
            results = xr.concat([results, results_b[vars]], dim="algorithm")

        data = {}
        for var in results.variables:
            if var == "algorithm":
                continue
            full_name = results[var].attrs.get("full_name")
            unit = results[var].attrs.get("unit")
            unit_str = "[]" if unit == "" else f"[${unit}$]"
            data[f"{full_name} {unit_str}"] = results[var].data

        return pd.DataFrame(data=data, index=results.algorithm)

    def get_precip_detection_results(
        self, name: Optional[str] = None, include_baselines: bool = True
    ) -> pd.DataFrame:
        """
        Get scalar results from precipitation detection metrics as pandas.Dataframe.

        Args:
            name: An optional name for the retrieval algorithm.
            include_baselines: If 'True', results from retrieval baselines will be included
                in the results.

        Return:
            A pandas.DataFrame containing the combined scalar results from
            the 'precip_detection_metrics' of this Evaluator object.
        """
        results = []
        for metric in self.precip_detection_metrics:
            res_m = metric.compute()
            drop = [var for var in res_m.variables if len(res_m[var].dims) > 0]
            results.append(res_m.drop_vars(drop))

        results = xr.merge(results).expand_dims("algorithm")
        results["algorithm"] = (("algorithm",), [name])

        if include_baselines:
            results_b = baselines.load_baseline_results(self.sensor)
            vars = list(results.variables.keys())
            results = xr.concat([results, results_b[vars]], dim="algorithm")

        data = {}
        for var in results.variables:
            if var == "algorithm":
                continue
            full_name = results[var].attrs.get("full_name")
            unit = results[var].attrs.get("unit")
            unit_str = "[]" if unit == "" else f"[${unit}$]"
            data[f"{full_name} {unit_str}"] = results[var].data

        return pd.DataFrame(data=data, index=results.algorithm)

    def get_prob_precip_detection_results(
        self, name: Optional[str] = None, include_baselines: bool = True
    ) -> pd.DataFrame:
        """
        Get scalar results from probabilistic precipitation detection
        metrics as pandas.Dataframe.

        Args:
            name: An optional name for the retrieval algorithm.
            include_baselines: If 'True', results from retrieval baselines
                will be included in the results.

        Return:
            A pandas.DataFrame containing the combined scalar results from
            the 'prob_precip_detection_metrics' of this Evaluator object.
        """
        results = []
        for metric in self.prob_precip_detection_metrics:
            res_m = metric.compute()
            drop = [var for var in res_m.variables if len(res_m[var].dims) > 0]
            results.append(res_m.drop_vars(drop))

        results = xr.merge(results).expand_dims("algorithm")
        results["algorithm"] = (("algorithm",), [name])

        if include_baselines:
            results_b = baselines.load_baseline_results(self.sensor)
            vars = list(results.variables.keys())
            results = xr.concat([results, results_b[vars]], dim="algorithm")

        data = {}
        for var in results.variables:
            if var == "algorithm":
                continue
            full_name = results[var].attrs.get("full_name")
            unit = results[var].attrs.get("unit")
            unit_str = "[]" if unit == "" else f"[${unit}$]"
            data[f"{full_name} {unit_str}"] = results[var].data

        return pd.DataFrame(data=data, index=results.algorithm)

    def get_heavy_precip_detection_results(
        self, name: Optional[str] = None, include_baselines: bool = True
    ) -> pd.DataFrame:
        """
        Get scalar results from heavy precipitation detection metrics as pandas.Dataframe.

        Args:
            name: An optional name for the retrieval algorithm.
            include_baselines: If 'True', results from retrieval baselines will be included
                in the results.

        Return:
            A pandas.DataFrame containing the combined scalar results from
            the 'heavy_precip_detection_metrics' of this Evaluator object.
        """
        results = []
        for metric in self.heavy_precip_detection_metrics:
            res_m = metric.compute()
            drop = [var for var in res_m.variables if len(res_m[var].dims) > 0]
            results.append(res_m.drop_vars(drop))

        results = xr.merge(results).expand_dims("algorithm")
        results["algorithm"] = (("algorithm",), [name])

        if include_baselines:
            results_b = baselines.load_baseline_results(self.sensor)
            vars = list(results.variables.keys())
            results = xr.concat([results, results_b[vars]], dim="algorithm")

        data = {}
        for var in results.variables:
            if var == "algorithm":
                continue
            full_name = results[var].attrs.get("full_name")
            unit = results[var].attrs.get("unit")
            unit_str = "[]" if unit == "" else f"[${unit}$]"
            data[f"{full_name} {unit_str}"] = results[var].data

        return pd.DataFrame(data=data, index=results.algorithm)

    def get_prob_heavy_precip_detection_results(
        self, name: Optional[str] = None, include_baselines: bool = True
    ) -> pd.DataFrame:
        """
        Get scalar results from probabilistic heavy precipitation detection
        metrics as pandas.Dataframe.

        Args:
            name: An optional name for the retrieval algorithm.
            include_baselines: If 'True', results from retrieval baselines
                will be included in the results.

        Return:
            A pandas.DataFrame containing the combined scalar results from
            the 'prob_heavy_precip_detection_metrics' of this Evaluator object.
        """
        results = []
        for metric in self.prob_heavy_precip_detection_metrics:
            res_m = metric.compute()
            drop = [var for var in res_m.variables if len(res_m[var].dims) > 0]
            res_m = res_m.drop_vars(drop)
            new_names = {name: name + "_heavy" for name in res_m.variables}
            results.append(res_m.rename(new_names))

        results = xr.merge(results).expand_dims("algorithm")
        results["algorithm"] = (("algorithm",), [name])

        if include_baselines:
            results_b = baselines.load_baseline_results(self.sensor)
            vars = list(results.variables.keys())
            results = xr.concat([results, results_b[vars]], dim="algorithm")

        data = {}
        for var in results.variables:
            if var == "algorithm":
                continue
            full_name = results[var].attrs.get("full_name")
            unit = results[var].attrs.get("unit")
            unit_str = "[]" if unit == "" else f"[${unit}$]"
            data[f"{full_name} {unit_str}"] = results[var].data

        return pd.DataFrame(data=data, index=results.algorithm)

    def get_results(self) -> xr.Dataset:
        """
        Combind results from all tracked metrics into a single xarray.Dataset.
        """
        results = []
        for metric in self.precip_quantification_metrics:
            results.append(metric.compute())
        for metric in self.precip_detection_metrics:
            results.append(metric.compute())
        for metric in self.prob_precip_detection_metrics:
            results.append(metric.compute())
        for metric in self.heavy_precip_detection_metrics:
            res = metric.compute()
            vars = res.variables
            res = res.rename(**{name: name + "_heavy" for name in vars})
            results.append(res)
        for metric in self.prob_heavy_precip_detection_metrics:
            res = metric.compute()
            vars = res.variables
            res = res.rename(**{name: name + "_heavy" for name in vars})
            results.append(res)

        results = xr.merge(results)
        return results
