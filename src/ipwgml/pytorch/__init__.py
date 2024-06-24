"""
ipwgml.pytorch
==============

Provides functionality for using the SPR dataset with PyTorch.
"""
from typing import Any, Dict, List

import torch
from torch import nn
import xarray as xr

from ipwgml.input import InputConfig, calculate_input_features


class PytorchRetrieval:
    """
    This class provides a generic retrieval callback function for PyTorch-based
    retrievals.

    The PytorchRetrieval wraps around a torch.nn.Module and extracts the input
    data from the xarray.Dataset provided by the ipwgml.evaluation.Evaluator
    and feeds it into the module. It then transform the output back from
    PyTorch tensors to an xarray.Dataset containing the retrieval results.

    The PytorchRetrieval class expects the module to return a dict containing
    the keys 'surface_precip', 'probability_of_precip', and
    'probability_of_heavy_precip'.
    """
    def __init__(
            self,
            model: nn.Module,
            retrieval_input: List[str | Dict[str, Any] | InputConfig],
            precip_threshold: float = 0.5,
            heavy_precip_threshold: float = 0.5,
            stack: bool = False,
            logits: bool = True,
            device: torch.device = torch.device("cpu"),
            dtype: torch.dtype = torch.float32,
    ):
        """
        Args:
            model: A torch.nn.Module implementing the retrieval.
            retrieval_input: A list defining the retrieval input.
            precip_threshold: The probability threshold to apply to
                transform the 'probability_of_precip' to a 'precip_flag'
                output.
            heavy_precip_threshold: Same as 'precip_threshold' but for
                heavy precip flag output.
            stack: Whether or not the model expects the input data to
                be stacked ('True') or as dictionary.
            logits: Whether the model returns logits instead of probabilities.
            device: A torch.device defining the device on which to perform
                inference.
            dtype: The dtype to which to convert the retrieval input.
        """
        self.model = model.to(device=device).eval()
        self.features = calculate_input_features(retrieval_input, stack=False)
        self.precip_threshold = precip_threshold
        self.heavy_precip_threshold = heavy_precip_threshold
        self.stack = stack
        self.logits = logits
        self.device = device
        self.dtype = dtype

    def __call__(self, input_data: xr.Dataset) -> xr.Dataset:
        """
        Run retrieval on input data.
        """
        feature_dim = 0
        if "scan" in input_data.dims:
            spatial_dims = ("scan", "pixel")
        elif "latitude" in input_data.dims:
            spatial_dims = ("latitude", "longitude")
        else:
            spatial_dims = ()

        if "batch" in input_data.dims:
            dims = ("batch",) + spatial_dims
            feature_dim += 1
        else:
            dims = spatial_dims


        features = self.features
        inpt = {}
        for name in features:
            inpt_data = torch.tensor(input_data[name].data).to(self.device, self.dtype)
            if len(dims) == 1:
                inpt_data = inpt_data.transpose(0, 1)
            inpt[name] = inpt_data

        if self.stack:
            inpt = torch.cat(list(inpt.values()), dim=feature_dim)

        with torch.no_grad():
            pred = self.model(inpt)
            results = xr.Dataset()
            if "surface_precip" in pred:
                results["surface_precip"] = (
                    dims,
                    pred["surface_precip"].select(feature_dim, 0).cpu().numpy()
                )
            if "probability_of_precip" in pred:
                pop = pred["probability_of_precip"].select(feature_dim, 0)
                if self.logits:
                    pop = torch.sigmoid(pop).cpu().numpy()
                results["probability_of_precip"] = (dims, pop)
                precip_flag = self.precip_threshold <= pop
                results["precip_flag"] = (dims, precip_flag)
            if "probability_of_heavy_precip" in pred:
                pohp = pred["probability_of_heavy_precip"].select(feature_dim, 0)
                if self.logits:
                    pohp = torch.sigmoid(pohp).cpu().numpy()
                results["probability_of_heavy_precip"] = (dims, pohp)
                heavy_precip_flag = self.heavy_precip_threshold <= pohp
                results["heavy_precip_flag"] = (dims, heavy_precip_flag)

        return results
