"""
ipwgml.data
===========

Provides functionality to access IPWG ML datasets.
"""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import logging
import multiprocessing
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import re

import click
import requests
from requests_cache import CachedSession
from requests import Session
from rich.progress import Progress

from ipwgml.definitions import (
    ALL_INPUTS,
    FORMATS,
    GEOMETRIES,
    REFERENCE_SENSORS,
    SPLITS,
)
from ipwgml import config
import ipwgml.logging


LOGGER = logging.getLogger(__name__)


BASE_URL = "https://rain.atmos.colostate.edu/gprof_nn/ipwgml"


FILE_REGEXP = re.compile('a href="([\w_]*\.nc)"')


def list_files(relative_path: str, base_url: Optional[str] = None) -> List[str]:
    """
    List files in dataset.

    Args:
        relative_path: The relative path identifying the dataset.
        base_url: An optional base URL that will overwrite the global default base URL.

    Return:
        A list of all files in the dataset.
    """
    if base_url is None:
        base_url = BASE_URL
    url = base_url + "/" + relative_path
    session = Session()
    resp = session.get(url)
    resp.raise_for_status()
    text = resp.text
    files = [relative_path + "/" + match for match in re.findall(FILE_REGEXP, text)]
    return files


def download_file(url: str, destination: Path) -> None:
    """
    Download file from server.

    Args:
        url: A string containing the URL of the file to download.
        destination: The destination to which to write the file.
    """
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        with open(destination, "wb") as output:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    output.write(chunk)


@contextmanager
def progress_bar_or_not(progress_bar: bool) -> Progress | None:
    """
    Context manager for a optional progress bar.
    """
    if progress_bar:
        with Progress(console=ipwgml.logging.get_console()) as progress:
            yield progress
    else:
        yield None


def download_files(
    files: List[str],
    destination: Path,
    progress_bar: bool = True,
    retries: int = 3,
    base_url: Optional[str] = None,
) -> List[str]:
    """
    Download files using multiple threads.

    Args:
        files: A list containing the relative paths of the files to download.
        destination: A Path object pointing to the local path to which to download the files.
        progress_bar: Whether or not to display a progress bar during download.
        retries: The number of retries to perform for failed files.
        base_url: Optional base URL to use for the data download that will overwrite the
             globally defined default base URL.

    Return:
        A list of the downloaded files.
    """
    if base_url is None:
        base_url = BASE_URL

    n_threads = min(multiprocessing.cpu_count(), 8)
    pool = ThreadPoolExecutor(max_workers=n_threads)
    ctr = 0

    failed = []

    if progress_bar and len(files) > 0:
        progress = Progress(console=ipwgml.logging.get_console())
        rel_path = "/".join(next(iter(files)).split("/")[:-1])
        bar = progress.add_task(f"Downloading files from {rel_path}:", total=len(files))
    else:
        progress = None
        bar = None

    while ctr < retries and len(files) > 0:

        tasks = []
        failed = []
        for path in files:
            *path, fname = path.split("/")
            path = "/".join(path)
            output_path = destination / path
            output_path.mkdir(parents=True, exist_ok=True)
            url = base_url + "/" + str(path) + "/" + fname
            tasks.append(pool.submit(download_file, url, output_path / fname))

        with progress_bar_or_not(progress_bar=progress_bar) as progress:
            if progress is not None:
                rel_path = "/".join(next(iter(files)).split("/")[:-1])
                bar = progress.add_task(
                    f"Downloading files from {rel_path}:", total=len(files)
                )
            else:
                bar = None

            for path, task in zip(files, tasks):

                try:
                    task.result()
                    if progress is not None:
                        progress.advance(bar, advance=1)
                except Exception:
                    LOGGER.exception(
                        "Encountered an error when trying to download files %s.",
                        path.split("/")[-1],
                    )
                    failed.append(path)

        ctr += 1
        files = failed

    if len(failed) > 0:
        LOGGER.warning(
            "The download of the following files failed: %s. If the issue persists please consider "
            "submitting an issue at github.com/simonpf/ipwgml.",
            failed,
        )

    return [fle for fle in files if fle not in failed]


def download_missing(
    dataset: str,
    destination: Path,
    base_url: Optional[str] = None,
    progress_bar: bool = False,
) -> None:
    """
    Download missing file from dataset.

    Args:
        datset: A relative URL identifying (parts of) a dataset.
        destination: Path pointing to the local directory containing the IPWGML data.
        base_url: If give, will overwrite the globally defined default URL.
    """
    local_files = set(
        [
            str(path.relative_to(destination))
            for path in (destination / dataset).glob("*.nc")
        ]
    )
    remote_files = set(list_files(dataset, base_url=base_url))
    missing = remote_files - local_files

    downloaded = download_files(missing, destination, base_url=base_url, progress_bar=progress_bar)
    return [destination / fle for fle in downloaded]


def download_dataset(
        dataset_name: str,
        reference_sensor: str,
        input_data: Union[str, List[str]],
        split: str,
        geometry: str,
        format: str,
        base_url: Optional[str] = None
) -> Dict[str, List[Path]]:
    """
    Download IPWGML dataset and return list of local files.

    Args:
        dataset_name: The IPWGML dataset to download.
        reference_sensor: The reference sensor of the dataset.
        input_data: The input data sources for which to download the data.
        split: Which split of the data to download.
        geometry: For which retrieval geometry to download the data.
        format: Which data format to download.
        base_url: The URL from which to download the data.

    Return:
        A dictionary listing locally available files for each input data
        source and the target data.
    """
    ipwgml_path = config.get_data_path()
    dataset = f"spr/{reference_sensor}/{split}/{geometry}/{format}/"

    download_missing(
        dataset + "target",
        ipwgml_path,
        progress_bar=True,
        base_url=base_url
    )
    paths = {
        "target": [ipwgml_path / fle for fle in list_files(dataset + "target")]
    }

    if isinstance(input_data, str):
        input_data = [input_data]
    for inpt in input_data:
        download_missing(dataset + inpt, ipwgml_path, progress_bar=True)
        paths[inpt] = list_files(dataset + inpt)
        paths[inpt] = [ipwgml_path / fle for fle in list_files(dataset + inpt)]

    return paths



@click.command()
@click.option("--data_path", type=str, default=None)
@click.option("--reference_sensors", type=str, default=None)
@click.option("--geometries", type=str, default=None)
@click.option("--formats", type=str, default=None)
@click.option("--splits", type=str, default=None)
@click.option("--inputs", type=str, default=None)
def cli(
    data_path: Optional[str] = None,
    reference_sensors: Optional[str] = None,
    geometries: Optional[str] = None,
    formats: Optional[str] = None,
    splits: Optional[str] = None,
    inputs: Optional[str] = None,
):
    """
    Download the SPR benchmark dataset.
    """
    dataset = "spr"

    if data_path is None:
        data_path = config.get_data_path()
    else:
        data_path = Path(data_path)
        if not data_path.exists():
            LOGGER.error("The provided 'data_path' does not exist.")
            return 1

    if reference_sensors is None:
        reference_sensors = REFERENCE_SENSORS
    else:
        reference_sensors = [sensor.strip() for sensor in reference_sensors.split(",")]
        for sensor in reference_sensors:
            if sensor not in REFERENCE_SENSORS:
                LOGGER.error(
                    "The sensor '%s' is currently not supported. Currently supported reference_sensors "
                    f"are {REFERENCE_SENSORS}."
                )
                return 1

    if geometries is None:
        geometries = GEOMETRIES
    else:
        geometries = [geometry.strip() for geometry in geometries.split(",")]
        for geometry in geometries:
            if geometry not in GEOMETRIES:
                LOGGER.error(
                    "The geometry '%s' is currently not supported. Currently supported geometries"
                    f" are {GEOMETRIES}."
                )
                return 1

    if formats is None:
        formats = FORMATS
    else:
        formats = [format.strip() for format in formats.split(",")]
        for format in formats:
            if format not in formats:
                LOGGER.error(
                    "The format '%s' is currently not supported. Currently supported formats"
                    f" are {FORMATS}."
                )
                return 1

    if splits is None:
        splits = SPLITS
    else:
        splits = [split.strip() for split in splits.split(",")]
        for split in splits:
            if split not in SPLITS:
                LOGGER.error(
                    "The split '%s' is currently not supported. Currently supported splits"
                    f" are {SPLITS}."
                )
                return 1

    if inputs is None:
        inputs = ALL_INPUTS
    else:
        inputs = [inpt.strip() for inpt in inputs.split(",")]
        for inpt in inputs:
            if inpt not in ALL_INPUTS:
                LOGGER.error(
                    "The input '%s' is currently not supported. Currently supported inputs"
                    f" are {ALL_INPUTS}."
                )
                return 1

    LOGGER.info(f"Starting data download to {data_path}.")

    for sensor in reference_sensors:
        for geometry in geometries:
            for inpt in inputs + ["target"]:
                for fmt in formats:
                    for split in splits:
                        if split == "evaluation":
                            dataset = f"spr/{sensor}/{split}/{geometry}/{inpt}"
                        else:
                            dataset = f"spr/{sensor}/{split}/{geometry}/{fmt}/{inpt}"
                        try:
                            download_missing(dataset, data_path, progress_bar=True)
                        except Exception:
                            LOGGER.exception(
                                f"An  error was encountered when downloading dataset '{dataset}'."
                            )

    config.set_data_path(data_path)


def list_local_files_rec(path: Path) -> Dict[str, Any]:
    """
    Recursive listing of ipwgml data files.

    Args:
        path: A path pointing to a directory containing ipwgml files.

    Return:
        A dictionary containing all sub-directories

    """
    netcdf_files = sorted(list(path.glob("*.nc")))
    if len(netcdf_files) > 0:
        return netcdf_files

    files = {}
    for child in path.iterdir():
        if child.is_dir():
            files[child.name] = list_local_files_rec(child)
    return files


def list_local_files() -> Dict[str, Any]:
    """
    List available ipwgml files.
    """
    data_path = config.get_data_path()
    files = list_local_files_rec(data_path)
    return files
