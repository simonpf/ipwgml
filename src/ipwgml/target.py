"""
ipwgml.target
=============

The ``ipwgml.target`` module provides the :class:`TargetConfig` class to configure
the loading of the retrieval reference data.

Usage
-----

``TargetConfig`` objects can be passed to the :class:`ipwgml.evaluation.Evaluator` to configure
the MRMS pixels that are used in the evaluation of the retrieval. They can also be passed to
the dataset classes defined in :module:`ipwgml.pytorch.datasets` to exclude low-quality pixels
from the training.

Members
-------
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

    The loaded precipitation values that don't satisfy the quality requirements, will
    be set to NAN. This will cause them to be ignored by the :class:`ipwgml.evaluation.Evaluator`.
    """

    target: str = "surface_precip"
    min_rqi: float = 1.0
    min_valid_fraction: float = 1.0
    no_snow: bool = True
    no_hail: bool = False
    min_gcf: Optional[float] = None
    max_gcf: Optional[float] = None

    def __init__(
        self,
        target: str = "surface_precip",
        min_rqi: float = 1.0,
        min_valid_fraction: float = 1.0,
        no_snow: bool = False,
        no_hail: bool = False,
        min_gcf: Optional[float] = None,
        max_gcf: Optional[float] = None,
        precip_threshold: float = 1e-3,
        heavy_precip_threshold: float = 1e1
    ):
        """
        Args:
            target: The name of the target variable. Should be 'surface_precip' for
                gauge-corrected MRMS surface precipitation downsampled to 0.036-degree
                resolution or 'surface_precip_fpavg' for footprint-average
                preciptiation.
            min_rqi: Pixels with radar-quality index (RQI) below this value will be masked.
            min_valid_fraction: The ``valid_fraction`` represents the fraction of valid
                native-MRMS pixels withing the downsampled 0.036-degree resolution pixels.
                Pixels with ``valid_fractions`` below this value will be masked.
            no_snow: If ``True``, pixels with non-zero snow fraction will be masked.
            no_snow: If ``True``, pixels with non-zero hail fraction will be masked.
            min_gcf: Pixels with a gauge-correction factor less than this value will be
                masked.
            max_gcf: Pixels with a gauge-correction factor greater than this will be
                masked.
            precip_threshold: The threshold to use to distinguish raining from
                non-raining pixels.
            heavy_precip_threshold: The threshold to use to identify heavy
                precipitation.
        """
        self.target = target
        self.min_rqi = min_rqi
        self.min_valid_fraction = min_valid_fraction
        self.no_snow = no_snow
        self.no_hail = no_hail
        self.min_gcf = min_gcf
        self.max_gcf = max_gcf
        self.precip_threshold = precip_threshold
        self.heavy_precip_threshold = heavy_precip_threshold

    def get_mask(self, target_data: Path | str | xr.Dataset) -> np.ndarray:
        """
        Get mask identifying invalid reference samples according to the
        target config's settings.

        Args:
            target_data: A Path or str pointing to a target data file or an xarray.Dataset containing
                the data from a loaded retrieval target file.

        Return:
            A field of bool values identifying the target samples that
            should be ignored.
        """
        with open_if_required(target_data) as data:

            target = data[self.target].data

            valid = np.ones_like(target, dtype=bool)

            # Allow for numerical inaccuracies to avoid noisy masks for min_rqi = 1.0.
            rqi = data["radar_quality_index"].data
            valid *= (rqi - self.min_rqi) > -1e-3

            # Allow for numerical inaccuracies to avoid noisy masks for min_valid_fraction = 1.0.
            valid_frac = data["valid_fraction"].data
            valid *= valid_frac - self.min_valid_fraction > -1e-3

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
        return ~valid

    def load_reference_precip(self, target_data: Path | str | xr.Dataset) -> np.ndarray:
        """
        Loads reference precip field data from a target file. The method ensure that the correct
        target variable is selected and masks samples not satisfying the quality requirements
        by setting them to NAN.

        Args:
            target_data: A Path or str pointing to a target data file or an xarray.Dataset containing
                the data from a loaded retrieval target file.

        Return:
            A numpy.ndarray containing the loaded target data.
        """
        with open_if_required(target_data) as data:
            target = data[self.target].data.copy()
            invalid = self.get_mask(target_data)
            target[invalid] = np.nan
        return target

    def load_precip_mask(self, target_data: Path | str | xr.Dataset) -> np.ndarray:
        """
        Load mask identifying  precipitation identified according to the
        target config object's heavy precipitation threshold.

        Args:
            target_data: A Path or str pointing to a target data file or an xarray.Dataset containing
                the data from a loaded retrieval target file.

        Return:
            A boolean numpy.ndarray containing the heavy precipitation mask.
        """
        with open_if_required(target_data) as data:
            target = data[self.target].data.copy()
        return target >= self.precip_threshold

    def load_heavy_precip_mask(self, target_data: Path | str | xr.Dataset) -> np.ndarray:
        """
        Load mask identifying heavy precipitation identified according to the
        target config object's heavy precipitation threshold.

        Args:
            target_data: A Path or str pointing to a target data file or an xarray.Dataset containing
                the data from a loaded retrieval target file.

        Return:
            A boolean numpy.ndarray containing the heavy precipitation mask.

        """
        with open_if_required(target_data) as data:
            target = data[self.target].data.copy()
        return target >= self.heavy_precip_threshold
