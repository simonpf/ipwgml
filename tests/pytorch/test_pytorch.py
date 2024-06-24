"""
Tests for the ipwgml.pytorch module
===================================
"""

import pytest
import torch
from torch import nn

from ipwgml.evaluation import Evaluator
from ipwgml.pytorch import PytorchRetrieval


class DummyRetrieval(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns dict of zero tensors containing estimates for 'surface_precip',
        'probability_or_precip', 'probability_of_heavy_precip'.

        Args:
            x: The input tensor.

        Return:

        A dictionary of dummy retrieval estimates.
        """
        if x.ndim == 3:
            feature_axis = 0
        elif x.ndim == 4:
            feature_axis = 1
        elif x.ndim == 2:
            feature_axis = 1
        else:
            raise ValueError(
                f"Input tensor has unsupported number of dimensions {x.ndim}"
            )

        pred = torch.zeros_like(x.select(dim=feature_axis, index=0)).unsqueeze(
            dim=feature_axis
        )
        return {
            "surface_precip": pred,
            "probability_of_precip": pred,
            "probability_of_heavy_precip": pred,
        }


@pytest.mark.parametrize("geometry", ["gridded", "on_swath"])
def test_evaluate_tabular(geometry, spr_gmi_evaluation, tmp_path):
    """
    Test evaluation over all files.
    """
    retrieval_input = ["gmi", "ancillary"]
    evaluator = Evaluator(
        "gmi", geometry, retrieval_input, ipwgml_path=spr_gmi_evaluation, download=False
    )

    model = DummyRetrieval()
    retrieval_fn = PytorchRetrieval(model, retrieval_input, stack=True)

    evaluator.evaluate(
        retrieval_fn=retrieval_fn,
        input_data_format="tabular",
        n_processes=1,
        output_path=tmp_path,
    )

    files = list(tmp_path.glob("results_*.nc"))
    assert len(files) > 0

    results = evaluator.get_results()


@pytest.mark.parametrize("geometry", ["gridded", "on_swath"])
def test_evaluate_tabular(geometry, spr_gmi_evaluation, tmp_path):
    """
    Test evaluation over all files.
    """
    retrieval_input = ["gmi", "ancillary"]
    evaluator = Evaluator(
        "gmi", geometry, retrieval_input, ipwgml_path=spr_gmi_evaluation, download=False
    )

    model = DummyRetrieval()
    retrieval_fn = PytorchRetrieval(model, retrieval_input, stack=True)

    evaluator.evaluate(
        retrieval_fn=retrieval_fn,
        input_data_format="spatial",
        batch_size=2,
        n_processes=1,
        output_path=tmp_path,
    )

    files = list(tmp_path.glob("results_*.nc"))
    assert len(files) > 0

    results = evaluator.get_results()
