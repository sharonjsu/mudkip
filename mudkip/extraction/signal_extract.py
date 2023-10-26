import numpy as np
from scipy.signal import medfilt, savgol_filter
from scipy.interpolate import interp1d
from scipy.ndimage.filters import gaussian_filter1d

from axolotl import utils
from axolotl.traces.filter import butter_lowpass_filter


# TODO make signal extraction class


def extract_rolling_dfof(
    movie, masks, timestamps, rate,
    bg_subtract=True, nonzero=True,
    bg_dilate=None,
    filter_type='gaussian',
    filter_before=True, filter_kws={'ksize': 0.08},
    dfof_kws={}
):
    """
    Extract rolling dfof trace

    Parameters
    ----------
    movie : ndarray
        3D array of movie frames: (T, H, W).
    masks : ndarray
        2D or 3D array of binary masks defining regions of interest (ROIs).
        If 2D, each column represents a separate ROI.
        If 3D, each slice represents a separate mask.
    timestamps : ndarray
        1D array of timestamps corresponding to each frame in `movie`.
    rate : float
        Sampling rate of the movie, in Hz.
    bg_subtract : bool, optional
        Whether to perform background subtraction on the traces.
        Default is True.
    nonzero : bool, optional
        Whether to calculate the mean of nonzero pixels only.
        Default is True.
    bg_dilate : int or tuple, optional
        If specified, the background mask will be dilated using a structuring
        element of size `bg_dilate` before subtraction.
        Default is None.
    filter_type : str or None, optional
        Type of filter to apply to the traces.
        If None, no filtering is applied.
        Default is 'gaussian'.
    filter_before : bool, optional
        Whether to apply the filter before or after calculating dfof.
        Default is True.
    filter_kws : dict, optional
        Keyword arguments to pass to the filter function.
        Default is {'ksize': 0.08}.
    dfof_kws : dict, optional
        Keyword arguments to pass to the `dfof_rolling` function.
        Default is {}.

    Returns
    -------
    traces : ndarray
        2D array of dfof traces, with shape (T, n_rois).
    timestamps : ndarray
        1D array of timestamps corresponding to each trace in `traces`.
    rate : float
        Sampling rate of the output traces, in Hz.
    """

    traces = mean_extraction(movie, masks, nonzero=nonzero)

    if bg_subtract:
        bg = ~masks.sum(0).astype(bool)
        if bg_dilate is not None:
            import cv2
            bg = (~bg).astype(np.uint8)
            kernel = np.ones((bg_dilate), np.uint8)
            bg = cv2.dilate(bg, kernel, iterations=1)
            bg = ~bg.astype(bool)
        bg_trace = mean_extraction(movie, bg[None, :], nonzero=nonzero)
        min_trace = np.min(traces, 0, keepdims=True)
        traces -= bg_trace
        traces -= np.min(traces, 0, keepdims=True) - min_trace

    # filter traces before dfof calculation
    if filter_before and filter_type is not None:
        traces, timestamps, rate = filter_traces(
            traces, timestamps, rate, filter_type, **filter_kws
        )

    traces = dfof_rolling(traces, rate, **dfof_kws)

    # filter traces after dfof calculation
    if not filter_before and filter_type is not None:
        traces, timestamps, rate = filter_traces(
            traces, timestamps, rate, filter_type, **filter_kws
        )

    return traces, timestamps, rate


def interp_traces(traces, timestamps, interp_rate=25):
    """
    Interpolate traces to a higher sampling rate.
    
    Parameters
    ----------
    traces : array_like
        The input array of traces to be interpolated. The shape should be 
        (n_samples, n_features).
    timestamps : array_like
        The timestamps associated with the traces. The shape should be 
        (n_samples,).
    interp_rate : float, optional
        The interpolation rate in Hz (samples/second) of the output traces.
        Default is 25.

    Returns
    -------
    array_like
        The interpolated traces with shape (n_samples_new, n_features).
    array_like
        The interpolated timestamps with shape (n_samples_new,).
    float
        The new sampling rate in Hz (samples/second).
    """
    interp = interp1d(timestamps, traces, axis=0, assume_sorted=True)
    timestamps = np.arange(timestamps[0], timestamps[-1], 1/interp_rate)
    traces = interp(timestamps)
    return traces, timestamps, interp_rate


def filter_traces(
    traces, timestamps, rate,
    method='median', ksize=0.24, interp_rate=None, **kwargs
):
    """
    Interpolate and median filter dfof trace

    Parameters
    ----------
    dfof : numpy.ndarray
        dfof array with shape t x ROIs.
    timestamps : numpy.ndarray
        timestamps for dfof.
    rate : float
        rate of acquisition in Hz.
    ksize : float
        kernel size for median filter in seconds.
    interp_rate : float
        rate to interpolate to before applying filter in Hz. Set to
        None, if you want to skip this step

    Returns
    -------
    dfof : numpy.ndarray
    timestamps : numpy.ndarray
    rate : float
    """

    # interpolate data to different rate
    if interp_rate is not None:
        traces, timestamps, rate = interp_traces(
            traces, timestamps, interp_rate
        )

    # apply filter
    if method == 'medfilt':
        traces = np.apply_along_axis(
            medfilt, 0, traces,
            int(utils.round(rate * ksize, 'odd')), **kwargs
        )
    elif method == 'gaussian':
        traces = gaussian_filter1d(
            traces, axis=0, sigma=(rate * ksize), **kwargs
        )
    elif method == 'butterworth':
        traces = np.apply_along_axis(
            butter_lowpass_filter,
            0, traces, fs=rate,
            cutoff=1 / ksize,
            **kwargs
        )
    elif method == 'savgol':
        polyorder = kwargs.pop('polyorder', 2)
        traces = savgol_filter(
            traces,
            int(utils.round(rate * ksize, 'odd')),
            polyorder=polyorder,
            axis=0,
            **kwargs
        )
    elif method is None:
        pass
    else:
        raise NameError(f'method {method} for filter dfof does not exist')

    return traces, timestamps, rate


def mean_extraction(movie, masks, nonzero=True, nans=True):
    """
    Return traces from masks (possibly weighted)
    by calculating the mean pixel value for each mask
    at each time point.

    Parameters
    ----------
    movie : array_like
        A 3D array representing the movie frames.
    masks : array_like
        A 2D or 3D array of boolean or numeric values.
        A single 2D mask can be provided, or a 3D array containing multiple masks.
        In the latter case, a 2D trace will be returned for each mask.
    nonzero : bool, optional
        If True, the mean is calculated only over non-zero elements in the mask.
    nans : bool, optional
        If True, NaN values in the trace are ignored when calculating the mean.

    Returns
    -------
    traces : array_like
        A 1D or 2D array containing the mean trace(s) calculated from the masks.
    """
    # multiple masks
    if masks.ndim == 3:
        signal = np.zeros((movie.shape[0], masks.shape[0]))
        for idx, mask in enumerate(masks):
            signal[:, idx] = mean_extraction(movie, mask, nonzero=nonzero, nans=nans)
        return signal

    # movie: T x h x w - mask: h x w
    # trace: T
    if masks.dtype == np.bool:
        if nans:
            return np.nanmean(movie[:, masks], axis=-1)
        else:
            return np.mean(movie[:, masks], axis=-1)

    A = movie * masks[None, ...]
    if nonzero:
        return utils.nonzero_mean(A, (1, 2))
    elif nans:
        return np.nanmean(A, (1, 2))
    else:
        return np.mean(A, (1, 2))


def dfof_rolling(
    traces, rate, window=30, method='nanmean', **kwargs
):
    """
    Compute delta F over F (dfof) from rolling window.

    Parameters
    ----------
    traces : numpy.ndarray
        Array of fluorescence traces.
    rate : float
        Sampling rate (Hz).
    window : int, optional
        Window size for rolling window (in seconds). Default is 30.
    method : str or callable, optional
        Method to use for baseline calculation. Can be a numpy function or
        any other callable that accepts an `axis` argument. Default is 'nanmean'.
    **kwargs
        Additional arguments to pass to `method`.

    Returns
    -------
    numpy.ndarray
        Array of dfof traces with the same shape as `traces`.
    """
    

    window = int(window * rate)  # convert to frames

    rolled_traces = utils.simple_rolling_window(
        traces, window, axis=0, border='edges'
    ).astype(float)

    if isinstance(method, str):
        method = getattr(np, method)
    baseline = method(rolled_traces, axis=1, **kwargs)

    return (traces - baseline)/baseline
