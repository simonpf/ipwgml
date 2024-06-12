"""
ipwgml.input
============

Provides input data records to specify input data configurations.
"""
from abc import ABC, abstractproperty
from copy import copy
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
import xarray as xr

from ipwgml.definitions import ANCILLARY_VARIABLES
from ipwgml.utils import open_if_required


class InputConfig(ABC):
    """
    Base class for input data records used to define what input data to load.
    """
    @classmethod
    def parse(self, inpt: Union[str, Dict[str, Any], "InputConfig"]) -> "InputConfig":

        if isinstance(inpt, InputConfig):
            return inpt
        elif isinstance(inpt, str):
            name = inpt
            kwargs = {}
        elif isinstance(inpt, dict):
            inpt = copy(inpt)
            name = inpt.pop("name", None)
            if name is None:
                raise ValueError(
                    "If a retrieval input is specified using a dict, it must have an entry "
                    "'name'."
                )
            kwargs = inpt
        else:
            raise ValueError(
                f"Unsupported input for parsing an InputConfig: {inpt}"
            )

        if name.lower() == "pmw":
            return PMW(**kwargs)
        elif name.lower() == "ancillary":
            return Ancillary(**kwargs)
        elif name.lower() == "geo":
            return Geo(**kwargs)
        elif name.lower() == "geo_ir":
            return GeoIR(**kwargs)
        raise RuntimeError(
            f"Provided retrieval input name '{name}' is not known."
        )

    @abstractproperty
    def name(self) -> str:
        """
        String representation of the input.
        """

    def __hash__(self):
        """
        Use class name as hash to allow building dictionaries with InputConfigs.
        """
        return hash(self.__class__.__name__)


@dataclass
class PMW(InputConfig):
    """
    InputData record class representing passive-microwave (PMW) observations.
    """
    def __init__(
            self,
            channels: Optional[List[int]] = None,
            include_angles: bool = True
    ):
        """
        Args:
            channels: An optional list of zero-based indices identifying channels to
                load. If 'None', all channels will be loaded.
            include_angles: Wether or not to include the eart-incidence angles of the
                observations in the input.
        """
        self.channels = channels
        self.include_angles = include_angles

    @property
    def name(self) -> str:
        return "pmw"

    def load_data(self, pmw_data_file: Path, target_time: xr.DataArray) -> Dict[str, np.ndarray]:
        """
        Load PMW observations from NetCDF file.

        Args:
            pmw_data_file: A Path object pointing to the file from which to load the input data.
            target_time: Not used.

        Return:
            A dictionary mapping the keys 'obs_pmw' the loaded PMW observations. If 'include_angles'
            is 'True' the dictionary will also containg the earth-incidence angles with the
            key 'eia_pmw'.
        """
        with open_if_required(pmw_data_file) as pmw_data:
            pmw_data = pmw_data[["observations", "earth_incidence_angle"]].transpose("channels", ...)
            if self.channels is not None:
                pmw_data = pmw_data[{"channels": self.channels}]
            else:
                pmw_data - pmw_data[{"channels": slice(0, None)}]

        inpt_data = {
            "obs_pmw": pmw_data["observations"].data,
            "eia_pmw": pmw_data["earth_incidence_angle"].data
        }
        return inpt_data


@dataclass
class Ancillary(InputConfig):
    """
    InputData record class representing retrieval ancillary data.
    """
    def __init__(
            self,
            variables: Optional[List[str]] = None
    ):
        """
        Args:
            variable: A list of strings specifying the ancillary data to load.
        """
        if variables is None:
            variables = ANCILLARY_VARIABLES
        self.variables = variables

    @property
    def name(self):
        return "ancillary"

    def load_data(self, ancillary_data_file: Path, target_time: xr.DataArray) -> xr.Dataset:
        """
        Load ancillary data from NetCDF file.

        Args:
            ancillary_data_file: A Path object pointing to the file from which to load the input data.
            targete_time: Not used.

        Return:
            A dicitonary mapping the single key 'ancillary' to an array containing the data from
            all ancillary variables stacked along the first axis.
        """
        with open_if_required(ancillary_data_file) as ancillary_data:
            data = []
            for var in self.variables:
                data.append(ancillary_data[var].data)

        data = np.stack(data)
        return {"ancillary": data}


@dataclass
class GeoIR(InputConfig):
    """
    InputData record class representing GEO-IR data.
    """
    time_steps: List[int]
    nearest: bool = True

    def __init__(
            self,
            time_steps: Optional[List[int]] = None,
            nearest: bool = False
    ):
        """
        Args:
            time_steps: Optional list of time steps to load.
            nearest: It 'True' only observations from the time nearest to the target
                time will be loaded.
        """
        if time_steps is None:
            time_steps = list(range(8))
        for time_step in time_steps:
            if (time_step < 0) or (7 < time_step):
                raise RuntimeError(
                    "Time steps for GeoIR input must be within [0, 8]."
                )
        self.time_steps = time_steps
        self.nearest = nearest

    @property
    def name(self):
        return "geo_ir"

    def load_data(self, geo_data_file: Path, target_time: xr.DataArray) -> xr.Dataset:
        """
        Load GEO IR data from NetCDF file.

        Args:
            geo_data_file: A Path object pointing to the file from which to load the input data.
            target_time: An xarray.DataArray containing the target times, which will be used to
                to interpolate the input observations to the nearest time step if 'self.nearest'
                is 'True'.

        Return:
            A dicitonary mapping the single key 'obs_geo' to an array containing the GEO IR
            observation from the desired time steps.
        """
        with open_if_required(geo_data_file) as geo_data:
            geo_data = geo_data.transpose("time", ...)
            if self.nearest:
                delta_t = geo_data.time - target_time
                inds = np.abs(delta_t).argmin("time")
                obs = geo_data.observations[{"time": inds}].data[None]
            else:
                obs = geo_data.observations[{"time": self.time_steps}].data
        return {"obs_geo_ir": obs}


@dataclass
class Geo(InputConfig):
    """
    InputData record class representing GEO data.
    """
    time_steps: List[int]
    nearest: bool = True

    def __init__(
            self,
            time_steps: Optional[List[int]] = None,
            nearest: bool = False
    ):
        """
        Args:
            time_steps: Optional list of time steps to load.
            nearest: It 'True' only observations from the time nearest to the target
                time will be loaded.
        """
        if time_steps is None:
            time_steps = list(range(4))
        for time_step in time_steps:
            if (time_step < 0) or (3 < time_step):
                raise RuntimeError(
                    "Time steps for Geo input must be within [0, 3]."
                )
        self.time_steps = time_steps
        self.nearest = nearest

    @property
    def name(self):
        return "geo"

    def load_data(self, geo_data_file: Path, target_time: xr.DataArray) -> xr.Dataset:
        """
        Load GEO data from NetCDF file.

        Args:
            geo_data_file: A Path object pointing to the file from which to load the input data.
            target_time: An xarray.DataArray containing the target times, which will be used to
                to interpolate the input observations to the nearest time step if 'self.nearest'
                is 'True'.

        Return:
            A dicitonary mapping the single key 'obs_geo' to an array containing the GEO
            observation from the desired time steps. The returned array will have the
            time and channel dimensions along the leading axes of the array.
        """
        with open_if_required(geo_data_file) as geo_data:
            geo_data = geo_data.transpose("time", "channel", ...)
            if self.nearest:
                delta_t = geo_data.time - target_time
                inds = np.abs(delta_t).argmin("time")
                if "latitude" in inds.dims:
                    inds = inds.drop_vars(["latitude", "longitude"])
                obs = geo_data.observations[{"time": inds}].transpose("channel", ...).data[None]
            else:
                obs = geo_data.observations[{"time": self.time_steps}].data
        shape = obs.shape
        return {"obs_geo": obs}


def parse_retrieval_inputs(
        inputs: List[str | Dict[str, Any] | InputConfig]
) -> List[InputConfig]:
    """
    Parse retrieval inputs.

    Args:
        inputs: A list specifying retrieval inputs. Each element in the list can
            be a string, a dictionary defining the retrieval input configuration
            or an InputConfig.

    Return:
        A list containing the retrieval input configuration represented using
        InputConfig objects.
    """
    return [InputConfig.parse(inpt) for inpt in inputs]
