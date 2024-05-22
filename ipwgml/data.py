"""
ipwgml.data
===========

Provides functionality to access IPWG ML datasets.
"""
from concurrent.futures import ThreadPoolExecutor
import logging
import multiprocessing
from pathlib import Path
from typing import List, Optional
import re

import requests
from requests_cache import CachedSession


LOGGER = logging.getLogger(__name__)


BASE_URL = "https://rain.atmos.colostate.edu/gprof_nn/ipwgml"


FILE_REGEXP = re.compile("a href=\"([\w_]*\.nc)\"")


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
    session = CachedSession()
    resp = session.get(url)
    resp.raise_for_status()
    text = resp.text
    files = [relative_path + "/" + match for match in re.findall(FILE_REGEXP, text)]
    return files


def download_file(url: str, destination: Path) -> None:
    """
    Download file from server.
    """
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        with open(destination, "wb") as output:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    output.write(chunk)


def download_files(
        files: List[str],
        destination: Path,
        progress_bar: bool = True,
        retries: int = 3,
        base_url: Optional[str] = None
) -> None:
    """
    Download files using multiple threads.

    Args:
        files: A list containing the relative paths of the files to download.
        destination: A Path object pointing to the local path to which to download the files.
        progress_bar: Whether or not to display a progress bar during download.
        retries: The number of retries to perform for failed files.
        base_url: Optional base URL to use for the data download that will overwrite the
             globally defined default base URL.
    """
    if base_url is None:
        base_url = BASE_URL

    n_threads = min(multiprocessing.cpu_count(), 8)
    pool = ThreadPoolExecutor(max_workers=n_threads)
    ctr = 0

    while ctr < retries and len(files) > 0:

        tasks = []
        failed = []
        for path in files:
            *path, fname = path.split("/")
            path =  "/".join(path)
            output_path = destination / path
            output_path.mkdir(parents=True, exist_ok=True)
            url = base_url + "/" + str(path)
            tasks.append(pool.submit(download_file, url, output_path / fname))

        for path, task in zip(files, tasks):
            try:
                task.result()
            except Exception:
                LOGGER.exception(
                    "Encountered an error when trying to download files %s.",
                    path.split("/")[-1]
                )
                failed.append(path)

        ctr += 1
        files = failed

    if len(failed) > 0:
        LOGGER.warning(
            "The download of the following files failed: %s. If the issue persists please consider "
            "submitting an issue at github.com/simonpf/ipwgml.",
            failed
        )


def download_missing(
        dataset: str,
        destination: Path,
        base_url: Optional[str] = None,
):
    """
    Download missing file from dataset.

    Args:
        datset: A relative URL identifying (parts of) a dataset.
        destination: Path pointing to the local directory containing the IPWGML data.
        base_url: If give, will overwrite the globally defined default URL.
    """
    local_files = set((destination / dataset).glob("*.nc"))
    remote_files = set(list_files(dataset, base_url=base_url))
    missing = remote_files - local_files
    download_files(missing, destination, base_url=base_url)
