"""
ipwg.pytorch.data
=================

Define pytorch dataset classes for loading the IPWG ML benchmark data.
"""
from datetime import datetime
from math import ceil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
import xarray as xr

from ipwgml.input import InputConfig, parse_retrieval_inputs
from ipwgml.target import TargetConfig
from ipwgml.definitions import ALL_INPUTS
from ipwgml.data import download_missing


class SPRTabular(Dataset):
    """
    Dataset class providing access to the IPWG IPR benchmark dataset.
    """
    def __init__(
        self,
        sensor: str,
        geometry: str,
        split: str,
        batch_size: Optional[int] = None,
        shuffle: Optional[bool] = False,
        retrieval_input: List[str | Dict[str, Any] | InputConfig] = None,
        target_config: Optional[TargetConfig] = None,
        ipwgml_path: Optional[Path] = None,
        download: bool = True,
    ):
        """
        Args:
            sensor: The sensor for which to load the benchmark dataset.
            geometry: Whether to load on_swath or regridded observations.
            split: Whether to load training ('train'), validation ('val'), or
                 test ('test') splits.
            batch_size: If given will return batched input data.
            shuffle: Whether or not to shuffle the samples in the dataset.
            retrieval_input: List of the retrieval inputs to load. The list should contain
                names of retrieval input sources ("pmw", "geo", "geo_ir", "ancillary"), dictionaries
                defining the input name and additional input options, or InputConfig. If not explicitly
                specified all available input data is loaded.
            target_config: An optional TargetConfig specifying quality requirements for the retrieval
                target data to load.
            ipwgml_path: Path containing or to which to download the IPWGML data.
            download: If 'True', missing data will be downloaded upon dataset creation. Otherwise, only
                locally available files will be used.
        """
        super().__init__()

        if ipwgml_path is None:
            ipwgml_path = config.get_data_path()
        else:
            ipwgml_path = Path(ipwgml_path)

        if not sensor.lower() in ["gmi", "mhs"]:
            raise ValueError(
                "Sensor must be one of ['gmi', 'mhs']."
            )
        self.sensor = sensor.lower()

        if not geometry.lower() in ["gridded", "on_swath"]:
            raise ValueError(
                "Geomtry must be one of ['gridded', 'on_swath']."
            )
        self.geometry = geometry.lower()

        if not split.lower() in ["train", "val", "test"]:
            raise ValueError(
                "Split must be one of ['train', 'val', 'test']"
            )
        self.split = split
        self.batch_size = batch_size
        self.shuffle = shuffle

        if retrieval_input is None:
            retrieval_input = ALL_INPUTS
        self.retrieval_input = parse_retrieval_inputs(retrieval_input)

        if target_config is None:
            target_config = TargetConfig()
        self.target_config = target_config

        self.pmw_data = None
        self.geo_data = None
        self.geo_ir_data = None
        self.ancillary_data = None
        self.target_data = None

        dataset = f"spr/{self.sensor}/{self.geometry}/tabular/{self.split}/"
        for inpt in self.retrieval_input:
            if download:
                download_missing(dataset + inpt.name, ipwgml_path)
            files = list((ipwgml_path / dataset / inpt.name).glob("*.nc"))
            setattr(self, inpt.name + "_data", xr.load_dataset(files[0]))

        if download:
            download_missing(dataset + "target", ipwgml_path)
        files = list((ipwgml_path / dataset / "target").glob("*.nc"))
        self.target_data = xr.load_dataset(files[0])

        # Determine valid samples and subset data
        surface_precip = self.target_config.load_data(self.target_data)
        valid = np.isfinite(surface_precip)
        self.target_data = self.target_data[{"samples": valid}]
        for inpt in self.retrieval_input:
            input_data = getattr(self, inpt.name + "_data", xr.load_dataset(files[0]))
            input_data = input_data[{"samples": valid}]
            setattr(self, inpt.name + "_data", input_data)

        self.rng = np.random.default_rng(seed=42)
        if self.shuffle:
            self.indices = self.rng.permutation(self.target_data.samples.size)
        else:
            self.indices = np.arange(self.target_data.samples.size)


    def __len__(self) -> int:
        """
        The number of samples in the dataset.
        """
        n_samples = self.target_data.samples.size

        if self.batch_size is None:
            return self.target_data.samples.size

        n_batches = ceil(n_samples / self.batch_size)
        return n_batches


    def __getitem__(self, ind: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Return sample from dataset.

        Args:
                ind: The index identifying the sample.

        Return:
                A tuple ``input, target`` containing a dictionary ``input`` mapping retrieval input
                names to corresponding input tensors and ``target`` a single tensor containing the
                target precipitation.
        """
        if ind >= len(self):
            raise IndexError("Dataset is exhausted.")

        if ind == 0:
            if self.shuffle:
                self.indices = self.rng.permutation(self.target_data.samples.size)
            else:
                self.indices = np.arange(self.target_data.samples.size)

        if self.batch_size is None:
            samples = self.indices[ind]
        else:
            batch_start = ind * self.batch_size
            batch_end = batch_start + self.batch_size
            samples = self.indices[batch_start:batch_end]

        target_data = self.target_data[{"samples": samples}]
        surface_precip = torch.tensor(target_data.surface_precip.data.astype(np.float32))
        target_time = target_data.time

        input_data = {}

        for inpt in self.retrieval_input:
            data = getattr(self, inpt.name + "_data", None)
            if data is None:
                continue
            data = inpt.load_data(data[{"samples": samples}], target_time=target_time)
            for key, arr in data.items():
                if self.batch_size is not None:
                    arr = arr.reshape(-1, arr.shape[-1]).transpose().copy()
                else:
                    arr = arr.ravel()
                input_data[key] = torch.tensor(arr.astype(np.float32))

        return input_data, surface_precip



def get_median_time(filename_or_path: Path | str) -> datetime:
    """
    Get median time from the filanem of a IPWGML SPR training scene.

    Args:
        filename: The filename or path pointing to any spatial training data file.

    Return:
        A datetime object representing the median time of the training scene.
    """
    if isinstance(filename_or_path, Path):
        filename = filename_or_path.name
    else:
        filename = filename_or_path
    date_str = filename.split("_")[-1][:-3]
    median_time = datetime.strptime(date_str, "%Y%m%d%H%M%S")
    return median_time


class SPRSpatial:
    """
    Dataset class providing access to the spatial variant of the satellite precipitation retrieval
    benchmark dataset.
    """
    def __init__(
        self,
        sensor: str,
        geometry: str,
        split: str,
        retrieval_input: List[str | dict[str | Any] | InputConfig] = None,
        target_config: TargetConfig = None,
        ipwgml_path: Optional[Path] = None,
        download: bool = True,
    ):
        """
        Args:
            sensor: The sensor for which to load the benchmark dataset.
            geometry: Whether to load on_swath or regridded observations.
            split: Whether to load training ('train'), validation ('val'), or
                 test ('test') splits.
            retrieval_input: List of the retrieval inputs to load. The list should contain
                names of retrieval input sources ("pmw", "geo", "geo_ir", "ancillary"), dictionaries
                defining the input name and additional input options, or InputConfig. If not explicitly
                specified all available input data is loaded.
            target_config: An optional TargetConfig specifying quality requirements for the retrieval
                target data to load.
            ipwgml_path: Path containing or to which to download the IPWGML data.
            download: If 'True', missing data will be downloaded upon dataset creation. Otherwise, only
                locally available files will be used.
        """
        super().__init__()

        if ipwgml_path is None:
            ipwgml_path = config.get_data_path()
        else:
            ipwgml_path = Path(ipwgml_path)

        if not sensor.lower() in ["gmi", "mhs"]:
            raise ValueError(
                "Sensor must be one of ['gmi', 'mhs']."
            )
        self.sensor = sensor.lower()

        if not geometry.lower() in ["gridded", "on_swath"]:
            raise ValueError(
                "Geomtry must be one of ['gridded', 'on_swath']."
            )
        self.geometry = geometry.lower()

        if not split.lower() in ["train", "val", "test"]:
            raise ValueError(
                "Split must be one of ['train', 'val', 'test']"
            )
        self.split = split

        if retrieval_input is None:
            retrieval_input = ALL_INPUTS
        self.retrieval_input = parse_retrieval_inputs(retrieval_input)

        if target_config is None:
            target_config = TargetConfig()
        self.target_config = target_config

        self.pmw = None
        self.geo = None
        self.geo_ir = None
        self.ancillary = None
        self.target = None

        dataset = f"spr/{self.sensor}/{self.geometry}/spatial/{self.split}/"
        for inpt in self.retrieval_input:
            if download:
                download_missing(dataset + inpt.name, ipwgml_path)
            files = sorted(list((ipwgml_path / dataset / inpt.name).glob("*.nc")))
            setattr(self, inpt.name, np.array(files))


        if download:
            download_missing(dataset + "target", ipwgml_path)
        files = sorted(list((ipwgml_path / dataset / "target").glob("*.nc")))
        self.target = np.array(files)

        self.check_consistency()


    def check_consistency(self):
        """
        Check consistency of training files.

        Raises:
            RuntimeError when the training scenes for any of the inputs is inconsistent with those
            available for the target.
        """
        target_times = set(map(get_median_time, self.target))
        for inpt in self.retrieval_input:
            inpt_times = set(map(get_median_time, getattr(self, inpt.name)))
            if target_times != inpt_times:
                raise RuntimeError(
                    f"Available target times are inconsistent with input files for input {inpt}."
                )


    def __len__(self) -> int:
        """
        The number of samples in the dataset.
        """
        return len(self.target)


    def __getitem__(self, ind: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Load sample from dataset.
        """
        with xr.open_dataset(self.target[ind]) as data:
            target_time = data.time
            target = self.target_config.load_data(data)
            target = torch.tensor(target.astype(np.float32))

        input_data = {}
        for inpt in self.retrieval_input:
            files = getattr(self, inpt.name, None)
            if files is None:
                continue
            data = inpt.load_data(files[ind], target_time=target_time)
            for name, arr in data.items():
                input_data[name] = torch.tensor(arr)

        return input_data, target
