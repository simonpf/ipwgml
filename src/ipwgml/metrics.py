"""
ipwgml.metrics
==============

Defines the metrics used to evaluate precipitation retrievals.
"""
from multiprocessing import shared_memory, Lock, Manager
from typing import Dict, Optional, Tuple

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


    def __getattr__(self, name):
        buffers = self.__dict__.get("_buffers", None)
        if buffers is not None:
            if name in buffers:
                shm, shape, dtype = buffers[name]
                if isinstance(shm, str):
                    shm = shared_memory.SharedMemory(shm)
                buffers[name] = (shm, shape, dtype)
                return np.ndarray(shape, dtype=dtype, buffer=shm.buf)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __del__(self):
        """
        Close connections to shared memory.
        """
        if hasattr(self, "_buffers"):
            for name, shm in self._buffers.items():
                shm = shm[0]
                if not isinstance(shm, str):
                    shm.close()


class Bias(Metric):
    """
    The bias or mean error calculated as the mean value of the difference between
    prediction and target values: bias = mean(pred - target).

    The mean is calculated over all results passed to the 'compute' method for
    which the target values are finite.
    """

    def __init__(
            self,
            relative: bool = True
    ):
        """
        Args:
            relative: If True, the bias is normalized using the mean of the target.
        """
        super().__init__(
            buffers={
                "x_sum": ((1,), np.float64),
                "y_sum": ((1,), np.float64),
                "counts": ((1,), np.int64),
            }
        )
        self.relative = relative

    def update(self, prediction: xr.DataArray, target: xr.DataArray) -> None:
        """
        Update metric values with given prediction.

        Args:
             prediction: An xarray.DataArray containing the prediction.
             target: An xarray.DataArray containing the reference values.
        """
        pred = prediction.data
        target = target.data
        valid = np.isfinite(target)
        pred = pred[valid]
        target = target[valid]

        with self.lock:
            self.x_sum += pred.sum()
            self.y_sum += target.sum()
            self.counts += valid.sum()

    def compute(self, name: Optional[str] = None) -> xr.Dataset:
        """
        Calculate the bias for all results passed to this metric object.

        Args:
            name: If given, the variable containing the bias will be named 'bias_{name}'.
                If not given, the variable is simply named 'bias'.

        Return:
            An xarray.Dataset containing a single, scalar variable 'bias' or 'bias_{name}'.

        """
        if name is None:
            name = "bias"
            rel_name = ""
        else:
            name = f"{name}_bias"

        if self.relative:
            bias = (self.x_sum - self.y_sum) / self.y_sum
        else:
            bias = (self.x_sum - self.y_sum) / self.counts

        return xr.Dataset({
            name: bias[0]
        })


class MSE(Metric):
    """
    The mean-squared error calculated as the mean value of the squared difference between
    prediction and target values: mse = mean((pred - target)^2).

    The mean is calculated over all results passed to the 'compute' method for
    which the target values are finite.
    """

    def __init__(self):
        super().__init__(
            buffers={
                "tot_sq_error": ((1,), np.float64),
                "counts": ((1,), np.int64),
            }
        )

    def update(self, prediction: xr.DataArray, target: xr.DataArray) -> None:
        """
        Update metric values with given prediction.

        Args:
             prediction: An xarray.DataArray containing the prediction.
             target: An xarray.DataArray containing the reference values.
        """
        pred = prediction.data
        target = target.data
        valid = np.isfinite(target)
        pred = pred[valid]
        target = target[valid]

        with self.lock:
            self.tot_sq_error += ((pred - target) ** 2).sum()
            self.counts += valid.sum()

    def compute(self, name: Optional[str] = None) -> xr.Dataset:
        """
        Calculate the bias for all results passed to this metric object.

        Args:
            name: If given, the variable containing the bias will be named 'bias_{name}'.
                If not given, the variable is simply named 'bias'.

        Return:
            An xarray.Dataset containing a single, scalar variable 'bias' or 'bias_{name}'.

        """
        if name is None:
            name = "mse"
        else:
            name = f"{name}_mse"

        return xr.Dataset({
            name: (self.tot_sq_error / self.counts)[0]
        })


class CorrelationCoef(Metric):
    """
    The linear correlation coefficient between predictions and target values.

    The mean is calculated over all results passed to the 'compute' method for
    which the target values are finite.
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

    def update(self, prediction: xr.DataArray, target: xr.DataArray) -> None:
        """
        Update metric values with given prediction.

        Args:
             prediction: An xarray.DataArray containing the prediction.
             target: An xarray.DataArray containing the reference values.
        """
        pred = prediction.data
        target = target.data
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

    def compute(self, name: Optional[str] = None) -> xr.Dataset:
        """
        Calculate the bias for all results passed to this metric object.

        Args:
            name: If given, the variable containing the bias will be named 'bias_{name}'.
                If not given, the variable is simply named 'bias'.

        Return:
            An xarray.Dataset containing a single, scalar variable 'bias' or 'bias_{name}'.

        """
        if name is None:
            name = "correlation_coef"
        else:
            name = f"{name}_correlation_coef"

        x_mean = self.x_sum / self.counts
        x2_mean = self.x2_sum / self.counts
        x_sigma = np.sqrt(x2_mean - x_mean ** 2)

        y_mean = self.y_sum / self.counts
        y2_mean = self.y2_sum / self.counts
        y_sigma = np.sqrt(y2_mean - y_mean ** 2)

        xy_mean = self.xy_sum / self.counts

        corr = (xy_mean - x_mean * y_mean) / (x_sigma * y_sigma)
        return xr.Dataset({
            name: corr[0]
        })



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


class SpectralCoherence(Metric):
    """
    Metric to calculate spectral statistics of retrieved fields.

    This metrics calculates the spectral energy and coherence between
    the retrieved and reference fields.
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


    def update(self, pred: xr.DataArray, target: xr.DataArray):
        """
        Calculate spectral statistics for all valid sample windows in
        given results.

        Args:
            pred: A xr.DataArray containing the predictions.
            target: A xr.DataArray containing the reference data.
        """
        pred = pred.data
        target = target.data

        valid = np.isfinite(target)
        for rect in iterate_windows(valid, self.window_size):
            row_start, col_start, row_end, col_end = rect
            pred_w = pred[row_start:row_end, col_start:col_end]
            target_w = target[row_start:row_end, col_start:col_end]
            w_pred = dctn(pred_w, norm="ortho")
            w_target = dctn(target_w, norm="ortho")
            self.coeffs_target_sum += w_target
            self.coeffs_target_sum2 += w_target * w_target
            self.coeffs_pred_sum += w_pred
            self.coeffs_pred_sum2 += w_pred * w_pred
            self.coeffs_targetpred_sum += w_target * w_pred
            self.coeffs_diff_sum += w_pred - w_target
            self.coeffs_diff_sum2 += (w_pred - w_target) * (w_pred - w_target)
            self.counts += np.isfinite(w_pred)

    def compute(self, name: Optional[str] = None):
        """
        Calculate error statistics for correlation coefficients by scale.

        Return:
            An 'xarray.Dataset' containing the finalized spectral statistics
            derived from the statistics collected in 'results'.
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

        return xr.Dataset({
            "scales": (("scales",), scales),
            "spectral_coherence": (("scales"), coherence),
            "effective_resolution": res
        })


class FAR(Metric):
    """
    Metric to calculate the false alarm rate for precipitation detection.
    """
    def __init__(self, precipitation_threshold: float = 0.0):
        """
        Args:
            precipitation_threshold:

        """
        self.precipitation_threshold = precipitation_threshold
        super().__init__(buffers={
            "n_positive": ((1,), np.float64),
            "n_false_positive": ((1,), np.float64),
        })


    def update(self, pred: xr.DataArray, target: xr.DataArray):
        """
        Args:
            pred: A xr.DataArray containing the predictions.
            target: A xr.DataArray containing the reference data.
        """
        valid = np.isfinite(target)

        pred = pred.data[valid]
        target = target.data[valid]

        true = target > self.precipitation_threshold
        positive = pred

        self.n_false_positive += (positive * ~true)
        self.n_positive += positive[valid]


    def compute(self, name: Optional[str] = None):
        """
        Return:
            An 'xarray.Dataset' containing the false alarm rate for the
            evaluated retrieval.
        """
        far = self.n_false_positive / self.n_positive
        return xr.Dataset({
            "far": far,
            "far_samples": self.n_positive
        })


class POD(Metric):
    """
    Metric to calculate the probability of detection (POD) for precipitation
    detection.
    """
    def __init__(self, precipitation_threshold: float = 0.0):
        """
        Args:
            precipitation_threshold:

        """
        self.precipitation_threshold = precipitation_threshold
        super().__init__(buffers={
            "n_true": ((1,), np.float64),
            "n_true_positive": ((1,), np.float64),
        })


    def update(self, pred: xr.DataArray, target: xr.DataArray):
        """
        Args:
            pred: A xr.DataArray containing the predictions.
            target: A xr.DataArray containing the reference data.
        """
        valid = np.isfinite(target)

        pred = pred.data[valid]
        target = target.data[valid]

        true = target > self.precipitation_threshold
        positive = pred

        self.n_true_positive += (positive * true)
        self.n_true += positive[valid]


    def compute(self, name: Optional[str] = None):
        """
        Return:
            An 'xarray.Dataset' containing the probability of detection for
            the evaluated retrieval.
        """
        pod = self.n_true_positive / self.n_true
        return xr.Dataset({
            "pod": pod,
            "pod_samples": self.n_true
        })
