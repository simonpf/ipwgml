"""
ipwgml.target
=============

This module provides functionality for defining quality criteria for target data.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr

from ipwgml.utils import open_if_required


@dataclass
class TargetConfig:
    """
    The TargetConfig class is used to specify quality criteria for the precipitation target
    data loaded for training and evaluating precipitation retrievals.

    """
    target: str = "surface_precip"
    min_rqi: float = 1.0
    min_valid_fraction: float = 1.0
    no_snow: bool = False
    no_hail: bool = False
    min_gcf: Optional[float] = None
    max_gcf: Optional[float] = None


    def load_data(self, target_data: Path | str | xr.Dataset) -> np.ndarray:
        """
        Loads retrieval target data from a target file. The method ensure that the correct
        target variable is selected and masks samples not satisfying the quality requirements
        by setting them to NAN.

        Args:
            target_data: A Path or str pointing to a target data file or an xarray.Dataset containing
                the data from a loaded retrieval target file.

        Return:
            A numpy.ndarray containing the loaded target data.
        """
        with open_if_required(target_data) as data:

            target = data[self.target].data

            valid = np.ones_like(target, dtype=bool)

            rqi = data["radar_quality_index"].data
            valid *= self.min_rqi <= rqi

            valid_frac = data["valid_fraction"].data
            valid *= self.min_valid_fraction <= valid_frac

            if self.no_snow:
                snow_frac = data["snow_fraction"].data
                valid *= snow_frac == 0.0

            if self.no_hail:
                hail_frac = data["hail_fraction"].data
                valid *= hail_frac == 0.0

            if self.min_gcf is not None:
                gcf = data["gauge_correction_factor"].data
                valid *= self.min_gcf <= gcf

            if self.max_gcf is not None:
                gcf = data["gauge_correction_factor"].data
                valid *= gcf <= self.min_gcf

        target[~valid] = np.nan
        return target
