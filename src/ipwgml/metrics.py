"""
ipwgml.metrics
==============

Defines the metrics used to evaluate precipitation retrievals.

The metrics objects are for iterative evaluation, i.e., results are passed to
the metric iteratively for each collocation scene. The metric object keeps
track of the necessary quantities required to compute the metrics. Finally, the
value of the metrics over all considered scene can be computed using each
metric's ``compute`` function. The metrics use shared memory to track required
quantities so that evaluation can be performed in parallel using multiple
processes.

Usage
-----

While the metric classes defined here can, in principle, be used on their own,
their intended use is with the :class:`ipwgml.evaluation.Evaluator` class, which
holds the metrics to track in its ``precip_quantification_metrics``,
``precip_detection_metrics``, ``prob_precip_detection_metrics``,
``heavy_precip_detection_metrics``, and ``prob_heavy_precip_detection_metrics``.


.. code-block:: Python

   evaluator.precip_quantification_metrics = [EffectiveResolution()] # Track only bias
   evaluator.precip_detection_metrics = [POD()] # Track only POD
   evaluator.prob_precip_detection_metrics = [PRCurve()] # Track only PR curve
   evaluator.heavy_precip_detection_metrics = [POD()] # Track only POD
   evaluator.prob_heavy_precip_detection_metrics = [PRCurve()] # Track only PR curve


The metrics are used by the :class:`ipgml.evaluation.Evaluator` to
"""
from multiprocessing import shared_memory, Lock, Manager
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.ndimage import binary_erosion
from scipy.fftpack import dctn
import xarray as xr


_MANAGER = None

def get_manager() -> Manager:
    """
    Cached access to multi-processing manager.
    """
    global _MANAGER
    if _MANAGER is None:
        _MANAGER = Manager()
    return _MANAGER


class Metric:
    """
    Base class for metrics that manages shared data arrays and can be used to manage the
    access to those arrays.

    """
    def __init__(
            self,
            buffers: Dict[str, Tuple[Tuple[int], str]]
    ):
        super().__init__()
        self.lock = get_manager().Lock()
        self._buffers = {}
        for name, (shape, dtype) in buffers.items():
            array = np.zeros(shape, dtype=dtype)
            shm = shared_memory.SharedMemory(create=True, size=array.nbytes)
            array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
            array[:] = 0.0
            self._buffers[name] = (shm.name, shape, dtype)
        self.owner = True

    def __getattr__(self, name: str) -> Any:
        buffers = self.__dict__.get("_buffers", None)
        if buffers is not None:
            if name in buffers:
                shm, shape, dtype = buffers[name]
                if isinstance(shm, str):
                    shm = shared_memory.SharedMemory(shm)
                buffers[name] = (shm, shape, dtype)
                return np.ndarray(shape, dtype=dtype, buffer=shm.buf)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


    def reset(self) -> None:
        """
        Reset metric state.

        Sets all buffers associated with the metric to zero, assuming that this is
        a valid initial state. If this is not the case, the child class should overwrite
        the function.
        """
        for name in self._buffers:
            print(name)
            array = getattr(self, name)
            array[:] = 0.0


    def cleanup(self) -> None:
        """
        Remove shared memory
        """
        if hasattr(self, "_buffers"):
            for name, shm in self._buffers.items():
                shm = shm[0]
                if isinstance(shm, str):
                    shm = shared_memory.SharedMemory(shm)
                shm.unlink()


    def __del__(self):
        """
        Close connections to shared memory.
        """
        if hasattr(self, "_buffers"):
            for name, shm in self._buffers.items():
                shm = shm[0]
                if isinstance(shm, str):
                    shm = shared_memory.SharedMemory(shm)
                shm.close()
                if self.owner:
                    pass
                    #shm.unlink()


class QuantificationMetric(Metric):
    """
    Helper class to identify metrics to assess precipitation quantification.
    """

class DetectionMetric(Metric):
    """
    Helper class to identify metrics to assess precipitation detection.
    """

class ProbabilisticDetectionMetric(Metric):
    """
    Helper class to identify metrics to assess probabilistic precipitation detection.
    """

class ValidFraction(QuantificationMetric):
    """
    This metric tracks the number of predictions that are left out because the retrieved
    value is NAN.
    """

    def __init__(self):
        super().__init__(
            buffers={
                "invalid": ((1,), np.int64),
                "counts": ((1,), np.int64),
            }
        )

    def update(self, pred: np.ndarray, target: np.ndarray) -> None:
        """
        Update metric values with given prediction.

        Args:
             pred: An np.ndarray containing the predicted values.
             target: An np.ndarray containing the reference values.
        """
        valid_pred = np.isfinite(pred)
        valid_target = np.isfinite(target)

        with self.lock:
            self.invalid += (~valid_pred * valid_target).astype(np.int64).sum()
            self.counts += valid_target.astype(np.int64).sum()

    def compute(self, name: Optional[str] = None) -> xr.Dataset:
        """
        Calculate the fraction of valid retrieval samples.

        Return:
            An xarray.Dataset containing a single, scalar variable 'valid_fraction' containing the
            fraction of valid retrievals.
        """
        valid_fraction = 1.0 - self.invalid / self.counts
        valid_fraction = xr.Dataset(
            {"valid_fraction": valid_fraction[0]}
        )
        valid_fraction.valid_fraction.attrs["full_name"] = "Valid fraction"
        valid_fraction.valid_fraction.attrs["unit"] = ""
        return valid_fraction


class Bias(QuantificationMetric):
    """
    The bias, or mean error, calculated as the mean value of the difference between
    prediction and target values:

    .. math::

      \\text{Bias} = \\mathbf{E}\{y_\\text{pred} - y_\\text{target}\}

    where the mean is calculated over all results passed to the 'compute' method for
    which the target values are finite.
    """

    def __init__(
            self,
            relative: bool = True
    ):
        """
        Args:
            relative: If True, the bias is calculated as percent of the mean reference
                 precipitation. Else the bias is calculated as absolute value.
        """
        super().__init__(
            buffers={
                "x_sum": ((1,), np.float64),
                "y_sum": ((1,), np.float64),
                "counts": ((1,), np.int64),
            }
        )
        self.relative = relative

    def update(self, prediction: np.ndarray, target: np.ndarray) -> None:
        """
        Update metric values with given prediction.

        Args:
             prediction: An np.ndarray containing the predicted values.
             target: An np.ndarray containing the reference values.
        """
        pred = prediction
        valid = np.isfinite(target)
        pred = pred[valid]
        target = target[valid]

        with self.lock:
            self.x_sum += pred.sum()
            self.y_sum += target.sum()
            self.counts += valid.sum()

    def compute(self, name: Optional[str] = None) -> xr.Dataset:
        """
        Calculate the MSE for all results passed to this metric object.

        Return:
            An xarray.Dataset containing a single, scalar variable 'mse' containing the
            MSE for the assessed results.
        """
        if self.relative:
            bias = 100.0 * (self.x_sum - self.y_sum) / self.y_sum
        else:
            bias = (self.x_sum - self.y_sum) / self.counts

        bias = xr.Dataset({"bias": bias[0]})
        bias.bias.attrs["full_name"] = "Bias"
        bias.bias.attrs["unit"] = "\%" if self.relative else "mm h^{-1}"
        return bias


class MAE(QuantificationMetric):
    """
    The mean-absolute error calculated as the mean value of the absolute value
    of the difference between prediction and target values:

    .. math::
      \\text{MAE} = \\mathbf{E}\\{|y_\\text{pred} - y_\\text{target}|\\}.

    where the mean is calculated over all results passed to the 'compute' method for
    which the target values are finite.
    """

    def __init__(self):
        super().__init__(
            buffers={
                "tot_abs_error": ((1,), np.float64),
                "counts": ((1,), np.int64),
            }
        )

    def update(self, prediction: np.ndarray, target: np.ndarray) -> None:
        """
        Update metric values with given prediction.

        Args:
             prediction: A np.ndarray containing the prediction.
             target: An np.ndarray containing the reference values.
        """
        pred = prediction
        valid = np.isfinite(target)
        pred = pred[valid]
        target = target[valid]

        with self.lock:
            self.tot_abs_error += np.abs(pred - target).sum()
            self.counts += valid.sum()

    def compute(self) -> xr.Dataset:
        """
        Calculate the MAE for all results passed to this metric object.

        Return:
            An xarray.Dataset containing a single, scalar variable 'mae' containing
            the MAE for all assessed estimates.
        """
        mae = xr.Dataset({
            "mae": (self.tot_abs_error / self.counts)[0]
        })
        mae.mae.attrs["full_name"] = "MAE"
        mae.mae.attrs["unit"] = "mm h^{-1}"
        return mae


class SMAPE(QuantificationMetric):
    """
    The symmetric mean absolute percentage error (SMAPE) with threshold :math:`t`.

    .. math::

      \\text{SMAPE}_t = \\mathbf{E}_{t \\leq y_\\text{target}}\\{\\frac{|y_\\text{pred} - y_\\text{target}|}{ 0.5 (|y_\\text{pred}| + |y_\\text{target}|)}\}

    where the mean is calculated over all results passed to the 'compute' method for
    which the target values are finite and for which the absolute value of the
    exceeds the given threshold value.
    """
    def __init__(self, threshold: float = 0.1):
        """
        Args:
            threshold: Minimum target value for samples to be considered in the
                calculation.

        """
        self.threshold = threshold
        super().__init__(
            buffers={
                "tot_rel_error": ((1,), np.float64),
                "counts": ((1,), np.int64),
            }
        )

    def update(self, prediction: np.ndarray, target: np.ndarray) -> None:
        """
        Update metric values with given prediction.

        Args:
             prediction: A np.ndarray containing the prediction.
             target: A np.ndarray containing the reference values.
        """
        pred = prediction
        valid = np.isfinite(target) * np.abs(target) > self.threshold
        pred = pred[valid]
        target = target[valid]

        with self.lock:
            self.tot_rel_error += (
                np.abs(pred - target) / (0.5 * (np.abs(pred) + np.abs(target)))
            ).sum()
            self.counts += valid.sum()

    def compute(self) -> xr.Dataset:
        """
        Calculate the SMAPE for all results passed to this metric object.

        Return:
            An xarray.Dataset containing a single, scalar variable 'smape' representing
            the SMAPE calculated over all results passed to this metric object.

        """
        smape = xr.Dataset({
            "smape": 100.0 * (self.tot_rel_error / self.counts)[0]
        })
        smape.smape.attrs["full_name"] = f"SMAPE$_{{{self.threshold:.2}}}$"
        smape.smape.attrs["unit"] = "%"
        return smape


class MSE(QuantificationMetric):
    """
    The mean-squared error calculated as the mean value of the squared difference between
    prediction and target values:

    .. math::

      \\text{MSE} = (\\mathbf{E}\{y_\\text{pred} - y_\\text{target}\})^2

    where mean is calculated over all results passed to the 'compute' method for
    which the target values are finite.
    """

    def __init__(self):
        super().__init__(
            buffers={
                "tot_sq_error": ((1,), np.float64),
                "counts": ((1,), np.int64),
            }
        )

    def update(self, prediction: np.ndarray, target: np.ndarray) -> None:
        """
        Update metric values with given prediction.

        Args:
             prediction: An np.ndarray containing the predicted values.
             target: An np.ndarray containing the reference values.
        """
        pred = prediction
        valid = np.isfinite(target)
        pred = pred[valid]
        target = target[valid]

        with self.lock:
            self.tot_sq_error += ((pred - target) ** 2).sum()
            self.counts += valid.sum()

    def compute(self) -> xr.Dataset:
        """
        Calculate the MSE for all results passed to this metric object.

        Return:
            An xarray.Dataset containing a single, scalar variable 'mse' representing
            the MSE calculated over all results passed to this metric object.
        """
        mse = xr.Dataset({
            "mse": (self.tot_sq_error / self.counts)[0]
        })
        mse.mse.attrs["full_name"] = "MSE"
        mse.mse.attrs["unit"] = "(mm h^{-1})^2"
        return mse


class CorrelationCoef(QuantificationMetric):
    """
    The linear correlation coefficient between predictions and target values.

    .. math::

      \\text{Correlation coeff.} = \\mathbf{E}\\frac{
      (y_\\text{pred} - \\mu_{y_\\text{pred}})(y_\\text{target} - \\mu{y_\\text{target})}
      }{
       \\sigma_{y_\\text{pred}} \sigma_{y_\\text{target}}
      }


    where the mean is calculated over all results passed to the 'compute' method for
    which the target values are finite and :math:`\\mu` and :math:`\\sigma` are used to denote
    the mean and standard deviations of the distributions of :math:`y_\text{pred}` and
    :math:`y_\\text{target}`.
    """

    def __init__(self):
        super().__init__(
            buffers={
                "x_sum": ((1,), np.float64),
                "x2_sum": ((1,), np.float64),
                "y_sum": ((1,), np.float64),
                "y2_sum": ((1,), np.float64),
                "xy_sum": ((1,), np.float64),
                "counts": ((1,), np.int64),
            }
        )

    def update(self, prediction: np.ndarray, target: np.ndarray) -> None:
        """
        Update metric values with given prediction.

        Args:
             prediction: An np.ndarray containing the predicted values.
             target: An np.ndarray containing the reference values.
        """
        pred = prediction
        valid = np.isfinite(target)

        pred = pred[valid]
        target = target[valid]

        with self.lock:
            self.x_sum += pred.sum()
            self.x2_sum += (pred ** 2).sum()
            self.y_sum += target.sum()
            self.y2_sum += (target ** 2).sum()
            self.xy_sum += (pred * target).sum()
            self.counts += valid.sum()

    def compute(self) -> xr.Dataset:
        """
        Calculate the bias for all results passed to this metric object.

        Return:
            An xarray.Dataset containing a single, scalar variable 'bias' or 'bias_{name}'.

        """
        x_mean = self.x_sum / self.counts
        x2_mean = self.x2_sum / self.counts
        x_sigma = np.sqrt(x2_mean - x_mean ** 2)

        y_mean = self.y_sum / self.counts
        y2_mean = self.y2_sum / self.counts
        y_sigma = np.sqrt(y2_mean - y_mean ** 2)

        xy_mean = self.xy_sum / self.counts

        corr = (xy_mean - x_mean * y_mean) / (x_sigma * y_sigma)
        corr = xr.Dataset({
            "correlation_coef": corr[0]
        })
        corr.correlation_coef.attrs["full_name"] = "Correlation coeff."
        corr.correlation_coef.attrs["unit"] = ""
        return corr



def iterate_windows(valid, window_size):
    """
    Iterate over non-overlapping windows in which all pixels are valid.

    Args:
        valid: A 2D numpy array identifying valid pixels.
        window_size: The size of the windows.

    Return:
        An iterator providing coordinates of randomly chosen windows that
        that cover the valid pixels in the given field.
    """
    conn = np.ones((window_size, window_size))
    valid = binary_erosion(valid, conn)

    row_inds, col_inds = np.where(valid)

    while len(row_inds) > 0:

        ind = np.random.choice(len(row_inds))
        row_c = row_inds[ind]
        col_c = col_inds[ind]

        row_start = row_c - window_size // 2
        row_end = row_start + window_size

        col_start = col_c - window_size // 2
        col_end = col_start + window_size

        yield row_start, col_start, row_end, col_end

        row_lim_lower = row_start - window_size // 2
        row_lim_upper = row_end + window_size // 2
        col_lim_lower = col_start - window_size // 2
        col_lim_upper = col_end + window_size // 2

        invalid = (
            (row_inds > row_lim_lower) *
            (row_inds <= row_lim_upper) *
            (col_inds > col_lim_lower) *
            (col_inds <= col_lim_upper)
        )
        row_inds = row_inds[~invalid]
        col_inds = col_inds[~invalid]


class SpectralCoherence(QuantificationMetric):
    """
    Metric to calculate spectral coherence curves and effective resolution
    for retrieved precipitation fields. Spectral coherence and effective
    resolution are calculated as described in:

    Pfreundschuh, S., Guilloteau, C., Brown, P. J., Kummerow, C. D., and Eriksson,
    P.: GPROF V7 and beyond: assessment of current and potential future versions of
    the GPROF passive microwave precipitation retrievals against ground radar
    measurements over the continental US and the Pacific Ocean, Atmos. Meas. Tech.,
    17, 515–538, https://doi.org/10.5194/amt-17-515-2024, 2024.

    """
    def __init__(self, window_size=32, scale=0.036):
        """
        Args:
            window_size: The size of the window over which the spectral
                coherence is computed.
            scale: Spatial extent of a single pixel. Defaults to 0.036 degree
                which is the resolution used for the gridded data of the
                SPR dataset.
        """
        self.window_size = window_size
        self.scale = scale
        super().__init__(buffers={
            "coeffs_target_sum": ((window_size,) * 2, np.float64),
            "coeffs_target_sum2": ((window_size,) * 2, np.float64),
            "coeffs_pred_sum": ((window_size,) * 2, np.float64),
            "coeffs_pred_sum2": ((window_size,) * 2, np.float64),
            "coeffs_targetpred_sum": ((window_size,) * 2, np.float64),
            "coeffs_targetpred_sum2": ((window_size,) * 2, np.float64),
            "coeffs_diff_sum": ((window_size,) * 2, np.float64),
            "coeffs_diff_sum2": ((window_size,) * 2, np.float64),
            "counts": ((window_size,) * 2, np.int64),
        })


    def update(self, pred: np.ndarray, target: np.ndarray):
        """
        Calculate spectral statistics for all valid sample windows in
        given results.

        Args:
            pred: A np.ndarray containing the predicted precipitation field.
            target: A np.ndarray containing the reference data.
        """
        pred = pred

        valid = np.isfinite(target)
        for rect in iterate_windows(valid, self.window_size):
            row_start, col_start, row_end, col_end = rect
            pred_w = pred[row_start:row_end, col_start:col_end]
            target_w = target[row_start:row_end, col_start:col_end]
            w_pred = dctn(pred_w, norm="ortho")
            w_target = dctn(target_w, norm="ortho")
            with self.lock:
                self.coeffs_target_sum += w_target
                self.coeffs_target_sum2 += w_target * w_target
                self.coeffs_pred_sum += w_pred
                self.coeffs_pred_sum2 += w_pred * w_pred
                self.coeffs_targetpred_sum += w_target * w_pred
                self.coeffs_diff_sum += w_pred - w_target
                self.coeffs_diff_sum2 += (w_pred - w_target) * (w_pred - w_target)
                self.counts += np.isfinite(w_pred)

    def compute(self):
        """
        Calculate error statistics for correlation coefficients by scale.

        Return:
            An 'xarray.Dataset' containing the spectral coherence and efficient resolution
            calculated using all results passed to this metric object.
        """
        corr_coeffs = []
        coherence = []
        energy_pred = []
        energy_target = []
        mse = []

        w_target_s = self.coeffs_target_sum
        w_target_s2 = self.coeffs_target_sum2
        w_pred_s = self.coeffs_pred_sum
        w_pred_s2 = self.coeffs_pred_sum2
        w_targetpred_s = self.coeffs_targetpred_sum
        w_d_s2 = self.coeffs_diff_sum2
        counts = self.counts
        N = self.coeffs_diff_sum2.shape[0]

        sigma_target = w_target_s2 / counts - (w_target_s / counts) ** 2
        sigma_pred = w_pred_s2 / counts - (w_pred_s / counts) ** 2
        target_mean = w_target_s / counts
        pred_mean = w_pred_s / counts
        targetpred_mean = w_targetpred_s / counts
        cc = (
            (targetpred_mean - target_mean * pred_mean) /
            (np.sqrt(sigma_target) * np.sqrt(sigma_pred))
        ).real
        co = np.abs(w_targetpred_s) / (np.sqrt(w_target_s2) * np.sqrt(w_pred_s2))
        co = co.real

        n_y = 0.5 * np.arange(sigma_target.shape[0])
        n_x = 0.5 * np.arange(sigma_target.shape[1])
        n = np.sqrt(
            n_x.reshape(1, -1) ** 2 +
            n_y.reshape(-1, 1) ** 2
        )
        bins = np.arange(min(n_y.max(), n_x.max()) + 1) - 0.5
        counts, _ = np.histogram(n, bins)

        corr_coeffs, _ = np.histogram(n, bins=bins, weights=cc)
        corr_coeffs /= counts
        coherence, _ = np.histogram(n, bins=bins, weights=co)
        coherence /= counts
        energy_pred, _ = np.histogram(n, weights=w_pred_s2, bins=bins)
        energy_target, _ = np.histogram(n, weights=w_target_s2, bins=bins)
        se, _ = np.histogram(n, weights=w_d_s2 / self.counts, bins=bins)

        ns = 1 - (se / energy_target)
        mse = se / counts
        n = 0.5 * (bins[1:] + bins[:-1])
        scales = 0.5 * (N - 1) * self.scale / n

        inds = np.argsort(scales[1:])
        resolved = np.where(coherence[1:][inds] > np.sqrt(1/ 2))[0]
        if len(resolved) == 0:
            res = np.inf
        else:
            res = scales[1:][inds][resolved[0]]

        results = xr.Dataset({
            "scales": (("scales",), scales),
            "spectral_coherence": (("scales"), coherence),
            "effective_resolution": res
        })
        results.spectral_coherence.attrs["full_name"] = "Spectral coherence"
        results.spectral_coherence.attrs["unit"] = ""
        results.effective_resolution.attrs["full_name"] = "Effective resolution"
        results.effective_resolution.attrs["unit"] = r"^\circ"
        return results


class FAR(DetectionMetric):
    """
    Metric to calculate the false alarm rate (FAR) for precipitation detection. The
    FAR is the fraction of false positive predictions and total number of positive
    predictions.

    .. math::
        \\text{FAR} = \\frac{\\#\\text{False positive}}{\\#\\text{True positive} + \\#\\text{False positive}}

    """
    def __init__(self):
        super().__init__(buffers={
            "n_positive": ((1,), np.int64),
            "n_false_positive": ((1,), np.int64),
        })


    def update(self, pred: np.ndarray, target: np.ndarray):
        """
        Args:
            pred: A np.ndarray containing the predictions.
            target: A np.ndarray containing the reference data.
        """
        true = target
        positive = pred
        with self.lock:
            self.n_false_positive += (positive * ~true).astype(np.int64).sum()
            self.n_positive += positive.astype(np.int64).sum()


    def compute(self, name: Optional[str] = None):
        """
        Return:
            An 'xarray.Dataset' containing the false alarm rate for the
            evaluated retrieval.
        """
        far = self.n_false_positive / self.n_positive
        results = xr.Dataset({
            "far": far,
            "far_samples": self.n_positive.copy()
        })
        results.far.attrs["full_name"] = "FAR"
        results.far.attrs["unit"] = ""
        return results


class POD(DetectionMetric):
    """
    Metric to calculate the probability of detection (POD) for precipitation
    detection. The POD is the ratio of true positive predictions and the total
    number of observed events.

    .. math::

        \\text{POD} = \\frac{\\#\\text{true positive}}{\\#\\text{True positive} + \\#\\text{False negative}}
    """
    def __init__(self):
        super().__init__(buffers={
            "n_true": ((1,), np.int64),
            "n_true_positive": ((1,), np.int64),
        })


    def update(self, pred: np.ndarray, target: np.ndarray):
        """
        Args:
            pred: A np.ndarray containing the predictions.
            target: A np.ndarray containing the reference data.
        """
        true = target
        positive = pred
        with self.lock:
            self.n_true_positive += (positive * true).astype(np.int64).sum()
            self.n_true += true.astype(np.int64).sum()


    def compute(self, name: Optional[str] = None):
        """
        Return:
            An 'xarray.Dataset' containing the probability of detection for
            the evaluated retrieval.
        """
        pod = self.n_true_positive / self.n_true
        results = xr.Dataset({
            "pod": pod,
            "pod_samples": self.n_true
        })
        results.pod.attrs["full_name"] = "POD"
        results.pod.attrs["unit"] = ""
        return results


class HSS(DetectionMetric):
    """
    Metric to calculate the Heidke-Skill Score for precipitation detection. The HSS
    is using the formula given `here <https://resources.eumetrain.org/data/4/451/english/msg/ver_categ_forec/uos2/uos2_ko3.htm>`_.
    """
    def __init__(self):
        super().__init__(
            buffers={
                "n_tp": ((1,), np.int64),
                "n_fp": ((1,), np.int64),
                "n_tn": ((1,), np.int64),
                "n_fn": ((1,), np.int64),
            }
        )

    def update(self, pred: np.ndarray, target: np.ndarray):
        """
        Args:
            pred: A np.ndarray containing the predictions.
            target: A np.ndarray containing the reference data.
        """
        true = target
        positive = pred
        with self.lock:
            self.n_tp += (positive * true).astype(np.int64).sum()
            self.n_fp += (positive * ~true).astype(np.int64).sum()
            self.n_tn += (~positive * ~true).astype(np.int64).sum()
            self.n_fn += (positive * ~true).astype(np.int64).sum()

    def compute(self, name: Optional[str] = None):
        """
        Return:
            An 'xarray.Dataset' containing the probability of detection for
            the evaluated retrieval.
        """
        n_pos = self.n_tp + self.n_fp
        n_true = self.n_tp + self.n_fn
        n_neg = self.n_tn + self.n_fn
        n_false = self.n_fp + self.n_tn
        n_tot = n_pos + n_neg

        standard = n_pos / n_tot * n_true / n_tot + n_neg / n_tot * n_false / n_tot
        hss = ((self.n_tp + self.n_tn) / n_tot - standard) / (1.0 - standard)

        results = xr.Dataset({
            "hss": hss,
            "pod_samples": n_tot
        })
        results.hss.attrs["full_name"] = "HSS"
        results.hss.attrs["unit"] = ""
        return results


class PRCurve(ProbabilisticDetectionMetric):
    """
    Calculates the precision recall curve for probabilistic detection results. The precision recall
    curve is a probabilistic detection metrics and thus expects predictions to be probabilities
    normalized to lie within :math:`[0, 1]`. If the probabilities are not normalized the ``range``
    argument can be used to define a customized value range.

    The precision and recall are defined as:

    .. math::

      \\text{Precision} = \\frac{\# \\text{True positive}}{\# \\text{True positive} + \# \\text{False positive}}

    .. math::

      \\text{Recall} = \\frac{\# \\text{True positive}}{\# \\text{True positive} + \# \\text{False negative}}

    Both precision and recall are calculated for a range of detection thresholds, i.e., values of the
    threshold probability above which an even is classified as positive. The values yield a curve
    representing the trade off between recall and precision as the detection threshold is increased.

    """
    def __init__(
            self,
            n_bins: int = 100,
            range: Tuple[float, float] = (0.0, 1.0),
            logarithmic: bool = False
    ):
        if logarithmic:
            self.thresholds = np.logspace(*range, n_bins)
        else:
            self.thresholds = np.linspace(*range, n_bins)
        super().__init__(
            buffers={
                "n_tp": ((n_bins,), np.int64),
                "n_fp": ((n_bins,), np.int64),
                "n_t": ((1,), np.int64),
            }
        )

    def update(self, pred: np.ndarray, target: np.ndarray):
        """
        Args:
            pred: A np.ndarray containing the predicted probabilities.
            target: A np.ndarray containing the true labels.
        """
        pred = pred.reshape(-1, 1)
        target = target.reshape(-1, 1)
        pred = pred >= self.thresholds[None]

        true_positive = pred * target
        false_positive = pred * ~target

        with self.lock:
            self.n_tp += true_positive.astype(np.int64).sum(axis=0)
            self.n_fp += false_positive.astype(np.int64).sum(axis=0)
            self.n_t += target.astype(np.int64).sum()

    def compute(self, name: Optional[str] = None):
        """
        Return:
            An 'xarray.Dataset' containing the the precision and recall values for all
            assessed threshold values as well as the area under the PR-curve.
        """
        precision = self.n_tp / (self.n_tp + self.n_fp)
        recall = self.n_tp / self.n_t

        valid = (self.n_tp + self.n_fp) > 0

        inds = np.argsort(recall[valid])
        auc = np.trapz(precision[valid][inds], x=recall[valid][inds])

        results = xr.Dataset({
            "threshold": (("threshold",), self.thresholds),
            "precision": (("threshold",), precision),
            "recall": (("threshold",), recall),
            "area_under_curve": auc,
        })

        results.area_under_curve.attrs["full_name"] = "AUC"
        results.area_under_curve.attrs["unit"] = ""
        results.precision.attrs["full_name"] = "Precision"
        results.precision.attrs["unit_name"] = ""
        results.recall.attrs["full_name"] = "Recall"
        results.recall.attrs["unit_name"] = ""
        return results
