"""
ipwgml.baselines
================

This module provides access to results from baseline retrievals.
"""
from pathlib import Path
from typing import List, Optional

import xarray as xr

BASELINES_GMI = {
    "imerg_final_v07b_gmi": "IMERG Final V7 (GMI)",
    "gprof_v07a_gmi": "GPROF V7 (GMI)",
}


def load_baseline_results(
        sensor: str,
        baselines: Optional[List[str]] = None

) -> xr.Dataset:
    """
    Load baseline results.

    Args:
        baselines: An optional list containing the baseline names to load. If not give,
            results from all baselines are loaded.

    Return:
        An xarray.Dataset containing the baseline results.
    """
    if sensor.lower() == "gmi":
        BASELINES = BASELINES_GMI
    else:
        raise ValueError("Currently on the sensor GMI is supported.")

    if baselines is None:
        baselines = BASELINES.keys()
    data_path = Path(__file__).parent / "files" / "baselines"
    results = []
    for baseline in baselines:
        if baseline not in BASELINES:
            raise ValueError(
                "Encountered unsupported baseline name '%s'.",
                baseline
            )

        results.append(xr.load_dataset(data_path / (baseline + ".nc")))

    results = xr.concat(results, dim="algorithm")
    results["algorithm"] = (("algorithm",), [BASELINES[name] for name in baselines])
    return results
