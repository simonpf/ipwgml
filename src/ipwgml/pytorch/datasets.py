"""
ipwg.pytorch.data
=================

This module provides PyTorch dataset classes for loading the SPR data.
The :class:`SPRTabular` will load data in tabular format while the
:class:`SPRSpatial` will load data in spatial format.

"""

from datetime import datetime
from math import ceil
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset
import hdf5plugin
import xarray as xr

from ipwgml.definitions import ALL_INPUTS
from ipwgml.data import download_missing
from ipwgml import config
from ipwgml.input import InputConfig, parse_retrieval_inputs
from ipwgml.target import TargetConfig


class SPRTabular(Dataset):
    """
    Dataset class for SPR data in tabular format.

    For efficiency, the SPRTabular data loads all of the training data into memory
    upon creation and provides the option to perform batching within the dataset
    instead of in the data loader.
    """

    def __init__(
        self,
        reference_sensor: str,
        geometry: str,
        split: str,
        batch_size: Optional[int] = None,
        shuffle: Optional[bool] = True,
        retrieval_input: List[str | Dict[str, Any] | InputConfig] = None,
        target_config: Optional[TargetConfig] = None,
        stack: bool = False,
        subsample: Optional[float] = None,
        ipwgml_path: Optional[Path] = None,
        download: bool = True,
    ):
        """
        Args:
            reference_sensor: The reference_sensor for which to load the benchmark dataset.
            geometry: Whether to load on_swath or regridded observations.
            split: Whether to load training ('training'), validation ('validation'), or
                 test ('testing') splits.
            batch_size: If given will return batched input data.
            shuffle: Whether or not to shuffle the samples in the dataset.
            retrieval_input: List of the retrieval inputs to load. The list should contain
                names of retrieval input sources ("pmw", "geo", "geo_ir", "ancillary"), dictionaries
                defining the input name and additional input options, or InputConfig. If not explicitly
                specified all available input data is loaded.
            target_config: An optional TargetConfig specifying quality requirements for the retrieval
                target data to load.
            stack: If 'False', the input will be loaded as a dictionary containing the input tensors
                from all input dataset. If 'True', the tensors will be concatenated along the
                feature axis and only a single tensor is loaded instead of dictionary.
            subsample: An optional fraction specifying how much of the dataset to load per epoch.
            ipwgml_path: Path containing or to which to download the IPWGML data.
            download: If 'True', missing data will be downloaded upon dataset creation. Otherwise, only
                locally available files will be used.
        """
        super().__init__()

        if ipwgml_path is None:
            ipwgml_path = config.get_data_path()
        else:
            ipwgml_path = Path(ipwgml_path)

        if not reference_sensor.lower() in ["gmi", "atms"]:
            raise ValueError("Reference_Sensor must be one of ['gmi', 'atms'].")
        self.reference_sensor = reference_sensor.lower()

        if not geometry.lower() in ["gridded", "on_swath"]:
            raise ValueError("Geomtry must be one of ['gridded', 'on_swath'].")
        self.geometry = geometry.lower()

        if not split.lower() in ["training", "validation", "testing"]:
            raise ValueError(
                "Split must be one of ['training', 'validation', 'testing']"
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

        self.stack = stack
        self.subsample = subsample

        self.geo_data = None
        self.geo_ir_data = None
        self.ancillary_data = None
        self.target_data = None

        # Load target data and mask
        dataset = f"spr/{self.reference_sensor}/{self.split}/{self.geometry}/tabular/"
        if download:
            download_missing(dataset + "target", ipwgml_path, progress_bar=True)
        files = list((ipwgml_path / dataset / "target").glob("*.nc"))
        if len(files) == 0:
            raise ValueError(
                f"Couldn't find any target data files. "
                " Please make sure that the ipwgml data path is correct or "
                "set 'download' to True to download the file."
            )
        self.target_data = xr.load_dataset(
            files[0],
            engine="h5netcdf"
        )
        valid = ~self.target_config.get_mask(self.target_data)
        self.target_data = self.target_data[{"samples": valid}]

        # Load input data
        for inpt in self.retrieval_input:
            if download:
                download_missing(dataset + inpt.name, ipwgml_path, progress_bar=True)
            files = list((ipwgml_path / dataset / inpt.name).glob("*.nc"))
            if len(files) == 0:
                raise ValueError(
                    f"Couldn't find any input data files for input '{inpt.name}'. "
                    " Please make sure that the ipwgml data path is correct or "
                    "set 'download' to True to download the file."
                )
            input_data = xr.load_dataset(files[0], engine="h5netcdf")
            setattr(self, inpt.name + "_data", input_data[{"samples": valid}])

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
        if self.subsample is not None:
            n_samples = self.subsample * n_samples

        if self.batch_size is None:
            return self.target_data.samples.size

        n_batches = ceil(n_samples / self.batch_size)
        return n_batches

    def __getitem__(
        self, ind: int
    ) -> Tuple[Union[torch.Tensor, Dict[str, torch.Tensor]], torch.Tensor]:
        """
        Return sample from dataset.

        Args:
                ind: The index identifying the sample.

        Return:
            A tuple ``input, target`` containing a the retrieval input data in ``input`` and
            the target data in ``target``. If ``stack`` is 'True', ``input`` is a tensor containing
            all input data, otherwise ``input`` is dictionary mapping the separate input names
            to separate tensors and it is up to the user to combine them.
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
        surface_precip = self.target_config.load_reference_precip(target_data).astype(
            np.float32
        )
        surface_precip = torch.tensor(surface_precip)
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

        if self.stack:
            input_data = torch.cat(list(input_data.values()), -1)

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


def apply(tensors: Any, transform: torch.Tensor) -> torch.Tensor:
    """
    Apply transformation to any container containing torch.Tensors.

    Args:
        tensors: An arbitrarily nested list, dict, or tuple containing
            torch.Tensors.
        transform:

    Return:
        The same containiner but with the given transformation function applied to
        all tensors.
    """
    if isinstance(tensors, tuple):
        return tuple([apply(tensor, transform) for tensor in tensors])
    if isinstance(tensors, list):
        return [apply(tensor, transform) for tensors in tensors]
    if isinstance(tensors, dict):
        return {key: apply(tensor, transform) for key, tensor in tensors.items()}
    if isinstance(tensors, torch.Tensor):
        return transform(tensors)
    raise ValueError("Encountered an unsupported type %s in apply.", type(tensors))


class SPRSpatial:
    """
    Dataset class providing access to the spatial variant of the satellite precipitation retrieval
    benchmark dataset.
    """

    def __init__(
        self,
        reference_sensor: str,
        geometry: str,
        split: str,
        retrieval_input: List[str | dict[str | Any] | InputConfig] = None,
        target_config: TargetConfig = None,
        stack: bool = False,
        augment: bool = True,
        ipwgml_path: Optional[Path] = None,
        download: bool = True,
    ):
        """
        Args:
            reference_sensor: The reference_sensor for which to load the benchmark dataset.
            geometry: Whether to load on_swath or regridded observations.
            split: Whether to load 'training', 'validation', or
                 'testing' splits.
            retrieval_input: List of the retrieval inputs to load. The list should contain
                names of retrieval input sources ("pmw", "geo", "geo_ir", "ancillary"), dictionaries
                defining the input name and additional input options, or InputConfig. If not explicitly
                specified all available input data is loaded.
            target_config: An optional TargetConfig specifying quality requirements for the retrieval
                target data to load.
            stack: If 'False', the input will be loaded as a dictionary containing the input tensors
                from all input dataset. If 'True', the tensors will be concatenated along the feature axis
                and only a single tensor is loaded instead of dictionary.
            augment: If 'True' will apply random horizontal and vertical flips to the input data.
            ipwgml_path: Path containing or to which to download the IPWGML data.
            download: If 'True', missing data will be downloaded upon dataset creation. Otherwise, only
                locally available files will be used.
        """
        super().__init__()

        if ipwgml_path is None:
            ipwgml_path = config.get_data_path()
        else:
            ipwgml_path = Path(ipwgml_path)

        if not reference_sensor.lower() in ["gmi", "atms"]:
            raise ValueError("Reference_Sensor must be one of ['gmi', 'atms'].")
        self.reference_sensor = reference_sensor.lower()

        if not geometry.lower() in ["gridded", "on_swath"]:
            raise ValueError("Geomtry must be one of ['gridded', 'on_swath'].")
        self.geometry = geometry.lower()

        if not split.lower() in ["training", "validation", "testing"]:
            raise ValueError(
                "Split must be one of ['training', 'validation', 'testing']"
            )
        self.split = split

        if retrieval_input is None:
            retrieval_input = ALL_INPUTS
        self.retrieval_input = parse_retrieval_inputs(retrieval_input)

        if target_config is None:
            target_config = TargetConfig()
        self.target_config = target_config

        self.stack = stack
        self.augment = augment

        self.pmw = None
        self.geo = None
        self.geo_ir = None
        self.ancillary = None
        self.target = None

        dataset = f"spr/{self.reference_sensor}/{self.split}/{self.geometry}/spatial/"
        for inpt in self.retrieval_input:
            if download:
                download_missing(dataset + inpt.name, ipwgml_path, progress_bar=True)
            files = sorted(list((ipwgml_path / dataset / inpt.name).glob("*.nc")))
            setattr(self, inpt.name, np.array(files))

        if download:
            download_missing(dataset + "target", ipwgml_path, progress_bar=True)
        files = sorted(list((ipwgml_path / dataset / "target").glob("*.nc")))
        self.target = np.array(files)

        self.check_consistency()
        self.worker_init_fn(0)

    def worker_init_fn(self, w_id: int) -> None:
        """
        Seeds the dataset loader's random number generator.
        """
        seed = int.from_bytes(os.urandom(4), "big") + w_id
        self.rng = np.random.default_rng(seed)

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
            target = self.target_config.load_reference_precip(data)
            target = torch.tensor(target.astype(np.float32))

        input_data = {}
        for inpt in self.retrieval_input:
            files = getattr(self, inpt.name, None)
            if files is None:
                continue
            data = inpt.load_data(
                files[ind],
                target_time=target_time,
            )
            for name, arr in data.items():
                input_data[name] = torch.tensor(arr.astype(np.float32))

        if self.augment:

            flip_h = self.rng.random() > 0.5
            flip_v = self.rng.random() > 0.5

            def transform(tensor: torch.Tensor) -> torch.Tensor:
                """
                Randomly flips a tensor along its two last dimensions.
                """
                dims = tuple()
                if flip_h:
                    dims = dims + (-2,)
                if flip_v:
                    dims = dims + (-1,)
                tensor = torch.flip(tensor, dims=dims)
                return tensor

            input_data = apply(input_data, transform)
            target = apply(target, transform)

        if self.stack:
            input_data = torch.cat(list(input_data.values()), axis=0)

        return input_data, target
