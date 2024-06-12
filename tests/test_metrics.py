"""
Tests for the ipwgml.metrics module.
"""
from concurrent.futures import ProcessPoolExecutor
from typing import List

import numpy as np
from scipy.fftpack import idctn
import xarray as xr

from ipwgml.metrics import (
    Bias,
    Metric,
    MSE,
    CorrelationCoef,
    SpectralCoherence
)


def increase_counter(metric):
    """
    Increase buffer attribue of metric.
    """
    with metric.lock:
        metric.buffer += 1.0


def test_shared_memory():
    """
    Ensure that memory between processes is shared and that access is synchronized.
    """
    n_jobs = 1024
    pool = ProcessPoolExecutor(max_workers=8)

    metric = Metric(buffers={"buffer": ((32, 32), np.float32)})

    buffer = metric.buffer

    tasks = []
    for _ in range(n_jobs):
        tasks.append(pool.submit(increase_counter, metric))

    for task in tasks:
        task.result()

    assert np.all(np.isclose(metric.buffer, n_jobs))


def evaluate_normal_preds(metric: Metric) -> None:
    """
    Helper function that evaluates evaluates the given metric with
    random values from two Normal distributions centered on 0 for
    the predictions and 10 for the target values.
    """
    x = xr.DataArray(np.random.normal(size=(100, 100)))
    y = xr.DataArray(np.random.normal(size=(100, 100)) + 10)
    metric.update(x, y)


def test_bias():
    """
    Test calculation of the bias.
    """
    n_jobs = 1024
    pool = ProcessPoolExecutor(max_workers=8)

    bias = Bias()
    tasks = []
    for _ in range(n_jobs):
        tasks.append(pool.submit(evaluate_normal_preds, bias))

    for task in tasks:
        task.result()

    result = bias.compute()
    assert np.isclose(result.bias.data, -1.0, atol=1e-2)


    bias = Bias(relative=False)
    tasks = []
    for _ in range(n_jobs):
        tasks.append(pool.submit(evaluate_normal_preds, bias))

    for task in tasks:
        task.result()

    result = bias.compute()
    assert np.isclose(result.bias.data, -10.0, atol=1e-2)


def test_mse():
    """
    Ensure that the calculated MSE is close to 102.
    """
    n_jobs = 1024
    pool = ProcessPoolExecutor(max_workers=8)

    mse = MSE()
    tasks = []
    for _ in range(n_jobs):
        tasks.append(pool.submit(evaluate_normal_preds, mse))

    for task in tasks:
        task.result()

    result = mse.compute()
    assert np.isclose(result.mse.data, 102, atol=1e-1)


def test_correlation_coef_indep():
    """
    Ensure that the calculated correlation coefficient is close to 0 for
    completely independent random predictions and targets.
    """
    n_jobs = 1024
    pool = ProcessPoolExecutor(max_workers=8)

    corr_coef = CorrelationCoef()
    tasks = []
    for _ in range(n_jobs):
        tasks.append(pool.submit(evaluate_normal_preds, corr_coef))

    for task in tasks:
        task.result()

    result = corr_coef.compute()
    assert np.isclose(result.correlation_coef.data, 0.0, atol=1e-2)


def evaluate_dependent_preds(metric: Metric) -> None:
    """
    Helper function that evaluates evaluates the given metric with
    random values from a Normal distributions where the target
    y is simply y = 2 * x.
    """
    x = xr.DataArray(np.random.normal(size=(100, 100)))
    y = 2.0 * x
    metric.update(x, y)


def evaluate_anticorrelated_preds(metric: Metric) -> None:
    """
    Helper function that evaluates evaluates the given metric with
    random values from a Normal distributions where the target
    y is simply y = - 2 * x.
    """
    x = xr.DataArray(np.random.normal(size=(100, 100)))
    y = -2.0 * x
    metric.update(x, y)



def test_correlation_coef_dep():
    """
    Ensure that the calculated correlation coefficient is close to -1 for
    for perfectly anti-correlated predictions and targets.
    """
    n_jobs = 1024
    pool = ProcessPoolExecutor(max_workers=8)

    corr_coef = CorrelationCoef()
    tasks = []
    for _ in range(n_jobs):
        tasks.append(pool.submit(evaluate_dependent_preds, corr_coef))
    for task in tasks:
        task.result()
    result = corr_coef.compute()
    assert np.isclose(result.correlation_coef.data, 1.0, atol=1e-2)

    corr_coef = CorrelationCoef()
    tasks = []
    for _ in range(n_jobs):
        tasks.append(pool.submit(evaluate_anticorrelated_preds, corr_coef))
    for task in tasks:
        task.result()
    result = corr_coef.compute()
    assert np.isclose(result.correlation_coef.data, -1.0, atol=1e-2)



def evaluate_random_spectral_field(
        metrics: List[Metric],
        size: int = (64, 64),
):
    """
    Generates predictions from a random spectral field by removing all variation with
    scales smaller than size[0] // 4.
    """
    wny = 0.5 * np.arange(size[0]) / (size[0] - 1)
    wnx = 0.5 * np.arange(size[1]) / (size[1] - 1)
    wn = np.sqrt(wny ** 2 + wnx[..., None] ** 2)
    scale = 0.5 / wn

    coeffs = np.random.normal(size=size)
    coeffs /= 0.5 * np.pi * wn ** 2
    coeffs[0, 0] = 0.0

    coeffs_ret = coeffs.copy()
    mask = scale < size[0] // 8
    coeffs_ret[mask] = 0.0

    target = idctn(coeffs, norm="ortho")
    pred = idctn(coeffs_ret, norm="ortho")

    for metric in metrics:
        metric.update(xr.DataArray(pred), xr.DataArray(target))


def test_spectral_coherence():
    """
    Test the calculation of the spectral coherence.
    """
    spectral_coherence = SpectralCoherence(window_size=64, scale=1)
    mse = MSE()

    n_jobs = 128
    pool = ProcessPoolExecutor(max_workers=8)


    tasks = []
    metrics = [spectral_coherence, mse]
    for _ in range(n_jobs):
        tasks.append(pool.submit(evaluate_random_spectral_field, metrics, size=(64, 64)))
    for task in tasks:
        task.result()

    result = spectral_coherence.compute()
    mse = mse.compute()

    closest_scale = result.scales.data[np.where(result.scales > 8)[0][-1]]
    assert result.effective_resolution.data == closest_scale
