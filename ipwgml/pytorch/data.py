"""
ipwg.pytorch.data
=================

Define pytorch dataset classes for loading the IPWG ML benchmark data.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
import xarray as xr

from ipwgml.data import download_missing


ANCILLARY_VARIABLES = [
    "wet_bulb_temperature",
    "two_meter_temperature",
    "lapse_rate",
    "total_column_water_vapor",
    "surface_temperature",
    "moisture_convergence",
    "leaf_area_index",
    "snow_depth",
    "orographic_wind",
    "10m_wind",
    "mountain_type",
    "land_fraction",
    "ice_fraction",
    "quality_flag",
    "sunglint_angle",
    "airlifting_index"
]



class SPRTabular(Dataset):
    """
    Dataset class providing access to the IPWG IPR benchmark dataset.
    """
    def __init__(
        self,
        sensor: str,
        geometry: str,
        split: str,
        retrieval_input: List[str] = None,
        ancillary_variables: Optional[List[str]] = None,
        ipwgml_path: Optional[Path] = None,
        download: bool = True,
    ):
        """
        Args:
            sensor: The sensor for which to load the benchmark dataset.
            geometry: Whether to load native or regridded observations.
            split: Whether to load training ('train'), validation ('val'), or
                 test ('test') splits.
            retrieval_input: List of the retrieval inputs to load. Available inputs include
                 - 'pmw': Passive microwave observations
                 - 'geo': GOES16 geostationary observations
                 - 'geo_ir': Single-channels geostationary IR observations.
                 - 'ancillary': Ancillary data.
                Defaults to loading all inputs.
            ancillary_variables: List of the ancillary variables to include in the loaded ancillary data.
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

        if not geometry.lower() in ["gridded", "native"]:
            raise ValueError(
                "Geomtry must be one of ['gridded', 'native']."
            )
        self.geometry = geometry.lower()

        if not split.lower() in ["train", "val", "test"]:
            raise ValueError(
                "Split must be one of ['train', 'val', 'test']"
            )
        self.split = split
        self.retrieval_input = retrieval_input

        if ancillary_variables is None:
            ancillary_variables = ANCILLARY_VARIABLES
        self.ancillary_variables = ancillary_variables

        self.pmw = None
        self.geo = None
        self.geo_ir = None
        self.ancillary = None
        self.target = None

        dataset = f"spr/{self.sensor}/{self.geometry}/tabular/{self.split}/"
        for inpt in retrieval_input:
            if inpt.lower() not in ["pmw", "geo", "geo_ir", "ancillary"]:
                raise ValueError(
                    "Encountered invalid input '%s'."
                )
            if download:
                download_missing(dataset + inpt, ipwgml_path)
            files = list((ipwgml_path / dataset / inpt).glob("*.nc"))
            setattr(self, inpt, xr.load_dataset(files[0]))


        if download:
            download_missing(dataset + "target", ipwgml_path)
        files = list((ipwgml_path / dataset / "target").glob("*.nc"))
        self.target = xr.load_dataset(files[0])


    def __len__(self) -> int:
        """
        The number of samples in the dataset.
        """
        return self.target.samples.size

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
        inpt = {}

        if self.pmw is not None:
            obs = self.pmw[{"samples": ind}].observations.data
            inpt["obs_pmw"] = obs
            angs = self.pmw[{"samples": ind}].earth_incidence_angle.data
            inpt["eia_pmw"] = torch.tensor(angs.astype(np.float32))

        if self.geo is not None:
            obs = self.geo[{"samples": ind}].observations.data
            inpt["obs_geo"] = torch.tensor(obs.reshape((obs.shape[0], -1)).astype(np.float32))

        if self.geo_ir is not None:
            obs = self.geo_ir[{"samples": ind}].observations.data
            inpt["obs_ir"] = torch.tensor(obs.astype(np.float32))

        if self.ancillary is not None:
            anc_data = self.ancillary[{"samples": ind}]
            anc = np.stack([anc_data[var].data for var in self.ancillary_variables])
            inpt["ancillary"] = torch.tensor(anc.astype(np.float32))

        target = torch.tensor(self.target[{"samples": ind}].surface_precip.data.astype(np.float32))
        return inpt, target


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
        retrieval_input: List[str] = None,
        ancillary_variables: Optional[List[str]] = None,
        ipwgml_path: Optional[Path] = None,
        download: bool = True,
    ):
        """
        Args:
            sensor: The sensor for which to load the benchmark dataset.
            geometry: Whether to load native or regridded observations.
            split: Whether to load training ('train'), validation ('val'), or
                 test ('test') splits.
            retrieval_input: List of the retrieval inputs to load. Available inputs include
                 - 'pmw': Passive microwave observations
                 - 'geo': GOES16 geostationary observations
                 - 'geo_ir': Single-channels geostationary IR observations.
                 - 'ancillary': Ancillary data.
                Defaults to loading all inputs.
            ancillary_variables: List of the ancillary variables to include in the loaded ancillary data.
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

        if not geometry.lower() in ["gridded", "native"]:
            raise ValueError(
                "Geomtry must be one of ['gridded', 'native']."
            )
        self.geometry = geometry.lower()

        if not split.lower() in ["train", "val", "test"]:
            raise ValueError(
                "Split must be one of ['train', 'val', 'test']"
            )
        self.split = split
        self.retrieval_input = retrieval_input

        if ancillary_variables is None:
            ancillary_variables = ANCILLARY_VARIABLES
        self.ancillary_variables = ancillary_variables

        self.pmw = None
        self.geo = None
        self.geo_ir = None
        self.ancillary = None
        self.target = None

        dataset = f"spr/{self.sensor}/{self.geometry}/spatial/{self.split}/"
        for inpt in retrieval_input:
            if inpt.lower() not in ["pmw", "geo", "geo_ir", "ancillary"]:
                raise ValueError(
                    "Encountered invalid input '%s'."
                )
            if download:
                download_missing(dataset + inpt, ipwgml_path)
            files = list((ipwgml_path / dataset / inpt).glob("*.nc"))
            setattr(self, inpt, np.array(files))


        if download:
            download_missing(dataset + "target", ipwgml_path)
        files = list((ipwgml_path / dataset / "target").glob("*.nc"))
        self.target = np.array(files)


    def __len__(self) -> int:
        """
        The number of samples in the dataset.
        """
        return len(self.target)

    def __getitem__(self, ind: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Load sample from dataset.
        """
        inpt = {}
        if self.pmw is not None:
            with xr.open_dataset(self.pmw[ind]) as data:
                obs = data.observations.data.transpose((2, 0, 1))
                inpt["obs_pmw"] = torch.tensor(obs.astype(np.float32))
                eia = data.earth_incidence_angle.data
                inpt["eia_pmw"] = torch.tensor(obs.astype(np.float32))

        if self.geo is not None:
            with xr.open_dataset(self.geo[ind]) as data:
                obs = data.observations.data.transpose((2, 0, 1))
                inpt["obs_geo"] = torch.tensor(obs.astype(np.float32))

        if self.geo_ir is not None:
            with xr.open_dataset(self.geo_ir[ind]) as data:
                obs = data.observations.data.transpose((2, 0, 1))
                inpt["obs_geo_ir"] = torch.tensor(obs.astype(np.float32))

        if self.ancillary is not None:
            with xr.open_dataset(self.ancillary[ind]) as data:
                for var in self.ancillary_variables:
                    anc = np.stack(data[var].data.astype(np.float32))
                    inpt["ancillary"] = torch.tensor(anc)

        with xr.open_dataset(self.target[ind]) as data:
            target = data.surface_precip.data.astype(np.float32)
            target = torch.tensor(target)

        return inpt, target
