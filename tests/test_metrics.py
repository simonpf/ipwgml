"""
Tests for the ipwgml.metrics module.
"""
from concurrent.futures import ProcessPoolExecutor
from typing import List

import numpy as np
from scipy.fftpack import idctn
from scipy import stats
import xarray as xr

from ipwgml.metrics import (
    ValidFraction,
    Bias,
    Metric,
    MAE,
    MSE,
    SMAPE,
    CorrelationCoef,
    SpectralCoherence,
    FAR,
    POD,
    HSS,
    PRCurve
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
    Helper function that  evaluates the given metric with
    random values from two Normal distributions centered on 0 for
    the predictions and 10 for the target values.
    """
    x = np.random.normal(size=(100, 100))
    y = np.random.normal(size=(100, 100)) + 10
    metric.update(x, y)

def evaluate_normal_preds_with_invalid(metric: Metric) -> None:
    """
    Same as evaluate_normal_preds but predictions are set to NAN with a probability
    of 50%.
    """
    x = np.random.normal(size=(100, 100))
    x[np.random.rand(*x.shape) > 0.5] = np.nan
    y = np.random.normal(size=(100, 100)) + 10
    metric.update(x, y)


def test_valid_fraction():
    """
    Ensure that valid fraction is 1 when all inputs are always valid.
    """
    n_jobs = 1024
    pool = ProcessPoolExecutor(max_workers=8)

    valid_frac = ValidFraction()
    tasks = []
    for _ in range(n_jobs):
        tasks.append(pool.submit(evaluate_normal_preds, valid_frac))

    for task in tasks:
        task.result()

    result = valid_frac.compute()
    assert np.isclose(result.valid_fraction.data, 1, rtol=1e-2)


    valid_frac = ValidFraction()
    tasks = []
    for _ in range(n_jobs):
        tasks.append(pool.submit(evaluate_normal_preds_with_invalid, valid_frac))

    for task in tasks:
        task.result()

    result = valid_frac.compute()
    assert np.isclose(result.valid_fraction.data, 0.5, rtol=1e-2)


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
    assert np.isclose(result.bias.data, -100.0, rtol=1e-2)


    bias = Bias(relative=False)
    tasks = []
    for _ in range(n_jobs):
        tasks.append(pool.submit(evaluate_normal_preds, bias))

    for task in tasks:
        task.result()

    result = bias.compute()
    assert np.isclose(result.bias.data, -10.0, atol=1e-2)


def test_mae():
    """
    Ensure that calculated MAE matches the mean of a folded normal distribution.
    """
    n_jobs = 1024
    pool = ProcessPoolExecutor(max_workers=8)

    mae = MAE()
    tasks = []
    for _ in range(n_jobs):
        tasks.append(pool.submit(evaluate_normal_preds, mae))

    for task in tasks:
        task.result()

    result = mae.compute()
    assert np.isclose(result.mae.data, 10.0, atol=1e-2)


def evaluate_fixed(metric: Metric) -> None:
    """
    Helper function the evaluated the given metric with fixed predictions
    with the value 0 and fixed targets with the value 1.
    """
    x = np.zeros((100, 100))
    y = np.ones_like(x)
    metric.update(x, y)


def test_smape():
    """
    Ensure that calculated MAE matches the mean of a folded normal distribution.
    """
    n_jobs = 1024
    pool = ProcessPoolExecutor(max_workers=8)

    smape = SMAPE()
    tasks = []
    for _ in range(n_jobs):
        tasks.append(pool.submit(evaluate_fixed, smape))

    for task in tasks:
        task.result()

    result = smape.compute()
    assert np.isclose(result.smape.data, 200.0, rtol=1e-2)


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
    x = np.random.normal(size=(100, 100))
    y = 2.0 * x
    metric.update(x, y)


def evaluate_anticorrelated_preds(metric: Metric) -> None:
    """
    Helper function that evaluates evaluates the given metric with
    random values from a Normal distributions where the target
    y is simply y = - 2 * x.
    """
    x = np.random.normal(size=(100, 100))
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
        metric.update(pred, target)


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



def evaluate_always(metric):
    """
    Evaluates the given metric with detection prediction that are always true
    and target values that are randomly true or false with equal probability.
    """
    pred = np.ones((100, 100), dtype=bool)
    target = np.random.rand(100, 100) > 0.5
    metric.update(pred, target)

def evaluate_never(metric):
    """
    Evaluates the given metric with detection prediction that are never true
    and target values that are randomly true or false with equal probability.
    """
    pred = np.zeros((100, 100), dtype=bool)
    target = np.random.rand(100, 100) > 0.5
    metric.update(pred, target)


def test_far():
    """
    Test the calculation of the FAR for prediction that are, respectively, always true and
    always negative and ensure that the metric takes on the expected values.
    """
    n_jobs = 128
    pool = ProcessPoolExecutor(max_workers=8)

    metric = FAR()
    tasks = []
    for _ in range(n_jobs):
        tasks.append(pool.submit(evaluate_always, metric))
    for task in tasks:
        task.result()
    far = metric.compute()
    assert np.isclose(far.far.data, 0.5, rtol=1e-2)
    metric.cleanup()

    metric = FAR()
    tasks = []
    for _ in range(n_jobs):
        tasks.append(pool.submit(evaluate_never, metric))
    for task in tasks:
        task.result()
    far = metric.compute()
    assert not np.isfinite(far.far.data)
    metric.cleanup()


def test_pod():
    """
    Test the calculation of the POD for prediction that are, respectively, always true and
    always negative and ensure that the metric takes on the expected values.
    """
    n_jobs = 128
    pool = ProcessPoolExecutor(max_workers=8)

    metric = POD()
    tasks = []
    for _ in range(n_jobs):
        tasks.append(pool.submit(evaluate_always, metric))
    for task in tasks:
        task.result()
    pod = metric.compute()
    assert np.isclose(pod.pod.data, 1.0, rtol=1e-2)

    metric = POD()
    tasks = []
    for _ in range(n_jobs):
        tasks.append(pool.submit(evaluate_never, metric))
    for task in tasks:
        task.result()
    pod = metric.compute()
    assert np.isclose(pod.pod.data, 0.0, rtol=1e-2)


def evaluate_always_right(metric):
    """
    Evaluates the given metric with predictions that are always right.
    """
    target = np.random.rand(100, 100) > 0.5
    pred = target
    metric.update(pred, target)


def evaluate_random(metric):
    """
    Evaluates the given metric with predictions that are always right.
    """
    target = np.random.rand(100, 100) > 0.5
    pred = np.random.rand(100, 100) > 0.5
    metric.update(pred, target)


def test_hss():
    """
    Test the calculation of the HSS using perfect and random predictions and ensure that
    the resulting values are 1.0 and 0.0, respectively.
    """
    n_jobs = 128
    pool = ProcessPoolExecutor(max_workers=8)

    metric = HSS()
    tasks = []
    for _ in range(n_jobs):
        tasks.append(pool.submit(evaluate_always_right, metric))
    for task in tasks:
        task.result()
    hss = metric.compute()
    assert np.isclose(hss.hss.data, 1.0, rtol=1e-2)

    metric = HSS()
    tasks = []
    for _ in range(n_jobs):
        tasks.append(pool.submit(evaluate_random, metric))
    for task in tasks:
        task.result()
    hss = metric.compute()
    assert np.isclose(hss.hss.data, 0.0, atol=1e-2)



def evaluate_random_probability(metric):
    """
    Evaluates the give metrics using a random probability.
    """
    target = np.random.rand(100, 100) > 0.5
    pred = np.random.rand(100, 100)
    metric.update(pred, target)


def test_prcurve():
    """
    Test the calculation of the PR curve using a random prediction on a balanced dataset.
    This should produce a PR curve that is on the diagonal.
    """
    n_jobs = 128
    pool = ProcessPoolExecutor(max_workers=8)

    metric = PRCurve()
    tasks = []
    for _ in range(n_jobs):
        tasks.append(pool.submit(evaluate_random_probability, metric))
    for task in tasks:
        task.result()
    pr_curve = metric.compute()

    assert np.isclose(pr_curve.recall.data[0], 1.0, rtol=5e-2).all()
    assert np.isclose(pr_curve.area_under_curve.data, 0.5, rtol=5e-2)
