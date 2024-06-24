# Retrieval evaluation

The ``ipwgml`` package provides a generic interface to evaluate precipitation
retrievals against the reference precipitation estimates in the SPR dataset. The
evaluation is based on the test period of the SPR dataset, i.e., the year 2023,
but is based on the full collocation scenes. While the evluation framework support retrieval operating on both on-swath and gridded geometries, the retrieval results are evaluated at the regular 0.036-degree latitude-longitude grided used by ``ipwgml``. This ensures that all retrievals are evaluated against the same reference data and avoid distortion of the retrieval accuracy that would occur if gridded and on-swath retrieval would be evaluated on their respective geometries.

## The ``ipwgml.evaluation`` module

The functionality for evaluating precipitation retrieval is implemented by the ``ipwgml.evaluation.Evaluator`` class. To make the evaluator applicable to
any retrieval, the interface to the retrieval algorithm or results to evaluate
is implemented in the form of an opaque retrieval callback function ``retrieval_fn``.

The figure below displays the basic interaction between the evaluator and the
retrieval callback function. The evaluator iterates over the collocations scenes
from the SPR testing period and calls the retrieval callback providing an
``xarray.Dataset`` containing the retrieval input data. The retrieval input can
contains all input data from the SPR dataset but also the coordinates and time
of the reference precipitation estimates. Using this information it is possible
to evaluate retrieval even if the results were produced offline. Examples of how
the evaluator can be used to evaluate IMERG and GPROF retrieval are given in the
Sections [](notebooks/evaluate_imerg.ipynb) and
[](notebooks/evaluate_gprof.ipynb).

```{figure} /figures/evaluation.svg
---
height: 400px
name: evaluation
---
Evaluating arbitrary retrieval using the ``iwpgml.evaluation.Evaulator``. 
The evaluator calls the retrieval callback function ``retrieval_fn`` with the input
data from the collocation scenes from the SPR testing period. The ``retrieval_fn`` is free to use the input data to calculate retrieval results or load results from existing files using the coordinates contained in the retrieval input data. The evaluator then evaluates the results against the reference precipitation estimates and keeps tracks of various accuracy metrics.
```

### Result format

For the evaluator to be able to evaluate the retrieval results, the results must adhere to the expected data format. The evaluator expects an xarray.Dataset with the same dimensions as the provided input data. The table below lists the expected variables. Note that the retrieval can provide any subset of them as they will be evaluated independently.

| Variable name             | Meaning                                       | Data type | Value range
|---------------------------|-----------------------------------------------|----------------|---|
| ``surface_precip``        | The surface precipitation rate in mm h$^{-1}$ | floating point |  $0 \leq$ |
| ``probability_of_precip`` | Estimated probability of observing a raining pixel | floating point | [0, 1] |
| ``precip_flag`` | Flag indicating whether a pixel is predicted to be raining or not | bool | {``True``,``False``} |
| ``probability_of_heavy_precip`` | Estimated probability of observing heavy precipitation (RR > 10 mm $h^{-1}$) | floating point | [0, 1] |
| ``heavy_precip_flag`` | Flag indicating whether a pixel is predicted to contain heavy precipitation| bool | {``True``,``False``} |

### Metrics

Precipitation retrieval are evaluated on five tasks: Precipitation estimation, precipitation detection, probabilistic precipitation detection, heavy precipitation detection and probabilistic detection of heavy precipitation. This aims to ensure that the evaluation covers the most relevant aspects of space-borne precipitation estimates.

#### Precipitation quantification

By default, the quantitative precipitation estimates returned by the retrieval callback function (``surface_precip``) are evaluated using the following metrics.
 
 1. Bias
 2. The mean absolute error (MAE)
 3. the mean squared error (MSE)
 4. The mean absolute percentage error (SMAPE) for precipitation rates exceedin 0.1 mm h$^{-1}$.
 5. The linear correlation coefficient
 6. The spectral coeherence from which the effective resolution is computed

#### Precipitation detection

By default, the deterministic precipitation detection (``precip_flag``) and deterministic heavy precipitation (``heavy_precip_flag``) estimates returned by the retrieval callback function are evaluated using the following metrics.

 1. POD
 2. FAR
 2. Heidke Skill Score (HSS)

#### Probabilistic precipitation detection

The probabilistic precipitation detection (``precip_flag``) and probabilistic heavy precipitation (``heavy_precip_flag``) estimates returned by the retrieval callback function are evaluated using the following metrics.

 1. Precision-recall curve (PR curve)
