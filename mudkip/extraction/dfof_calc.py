"""Caiman-style dFoF calculation
"""

import warnings
import numpy as np
import scipy
from scipy import fftpack
from scipy.ndimage import percentile_filter
from typing import Optional, Tuple


def calculate_dfof(fluorescence, frames_window=3000, background=None, verbose=0, max_q=None, q=None):
    """
    Calculate deldtafo over f from raw fluorescence traces
    Relative measure of the change in fluorescence intensity over a baseline fluorescence value.

    Parameters
    ----------
    fluorescence : ndarray of shape (n_frames x n_rois)
        raw fluorescence
    frames_window : int, optional
        Windows size to use to calculate percentile background in, by default 3000

    Returns
    -------
    dfof 
    """
    n_frames, n_rois = fluorescence.shape
    
    if background is not None:
        if background.ndim == 1:
            background = background[:, None]
    
    # just use first frames for calculation of percentile
    if q is None:
        percentiles = []
        for i in range(n_frames // frames_window):
            fl_ = fluorescence[i*frames_window:(i+1)*frames_window]
            p_, _ = df_percentile(fl_, axis=0)
            percentiles.append(p_)
        percentiles = np.nanmedian(percentiles, axis=0)
        if max_q is not None:
            percentiles[percentiles > max_q] = max_q
    else:
        percentiles = np.broadcast_to(q, n_rois)
    
    if verbose:
        print(f"Percentiles found for each ROI: {percentiles}")
    
    if frames_window is None or frames_window > n_frames:
        baseline = np.stack([
            np.percentile(f, q) 
            for f, q in
            zip(fluorescence.T, percentiles)
        ])  # shape of n_rois
        if background is None:
            baseline_bg = 0
        else:
            baseline_bg = np.stack([
                np.percentile(f, q) 
                for f, q in
                zip(background.T, percentiles)
            ])  # 1 or n_rois
    else:
        baseline = np.stack([
            percentile_filter(f, q, frames_window) 
            for f, q in
            zip(fluorescence.T, percentiles)
        ]).T  # shape of n_frames, n_rois
        if background is None:
            baseline_bg = 0
        else:
            baseline_bg = np.stack([
                percentile_filter(f, q, frames_window) 
                for f, q in
                zip(background.T, percentiles)
            ]).T  # shape of n_frames, 1 or n_rois
            
    dfof = (fluorescence - baseline) / (baseline + baseline_bg)
    return dfof


# COPIED AND ADJUSTED from caiman project
def kde(data: np.ndarray, N: Optional[int]=None, MIN: Optional[float]=None, MAX: Optional[float]=None) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates the kernel density estimate of a given data set.
    
    Parameters
    ----------
    data (array_like): 1-D data set for the density estimate.
    N (int, optional): Number of bins for histogram calculation. Defaults to 2**12.
    MIN (float, optional): Lower bound of the range on which to calculate the density.
        Defaults to minimum of data minus 10% of the range of the data.
    MAX (float, optional): Upper bound of the range on which to calculate the density.
        Defaults to maximum of data plus 10% of the range of the data.
    
    Returns
    -------
    bandwidth (float): Estimated bandwidth of the density.
    mesh (array_like): Bin centers used in density calculation.
    density (array_like): Calculated density estimate.
    cdf (array_like): Cumulative density function of the estimated density.
    
    Notes
    -----
    The density estimate is normalized such that the integral over the entire range is equal to 1.
    """
    # kernel density estimate
    # Parameters to set up the mesh on which to calculate
    N = 2**12 if N is None else int(2**scipy.ceil(scipy.log2(N)))
    if MIN is None or MAX is None:
        minimum = min(data)
        maximum = max(data)
        Range = maximum - minimum
        MIN = minimum - Range / 10 if MIN is None else MIN
        MAX = maximum + Range / 10 if MAX is None else MAX

    # Range of the data
    R = MAX - MIN

    # Histogram the data to get a crude first approximation of the density
    M = len(data)
    DataHist, bins = scipy.histogram(data, bins=N, range=(MIN, MAX))
    DataHist = DataHist / M
    DCTData = fftpack.dct(DataHist, norm=None)

    I = [iN * iN for iN in range(1, N)]
    SqDCTData = (DCTData[1:] / 2)**2

    # The fixed point calculation finds the bandwidth = t_star
    guess = 0.1
    try:
        t_star = scipy.optimize.brentq(fixed_point, 0, guess, args=(M, I, SqDCTData))
    except ValueError:
        print('Oops!')
        return None

    # Smooth the DCTransformed data using t_star
    SmDCTData = DCTData * scipy.exp(-scipy.arange(N)**2 * scipy.pi**2 * t_star / 2)
    # Inverse DCT to get density
    density = fftpack.idct(SmDCTData, norm=None) * N / R
    mesh = [(bins[i] + bins[i + 1]) / 2 for i in range(N)]
    bandwidth = scipy.sqrt(t_star) * R

    density = density / scipy.trapz(density, mesh)
    cdf = np.cumsum(density) * (mesh[1] - mesh[0])

    return bandwidth, mesh, density, cdf


def fixed_point(t, M, I, a2):
    """
    Caculates a fixed point of a nonlinear dynamical system defined by a function f using an iterative approach
    
    Parameters 
    ----------
    t : 1D np.array
        Initial guess for the fixed point 
    I : float64
    M : float64
    a2: float64
    Returns
    -------
    Fixed point
    """
    l = 7
    I = scipy.float64(I)
    M = scipy.float64(M)
    a2 = scipy.float64(a2)
    f = 2 * scipy.pi**(2 * l) * scipy.sum(I**l * a2 * scipy.exp(-I * scipy.pi**2 * t))
    for s in range(l, 1, -1):
        K0 = scipy.prod(range(1, 2 * s, 2)) / scipy.sqrt(2 * scipy.pi)
        const = (1 + (1 / 2)**(s + 1 / 2)) / 3
        time = (2 * const * K0 / M / f)**(2 / (3 + 2 * s))
        f = 2 * scipy.pi**(2 * s) * scipy.sum(I**s * a2 * scipy.exp(-I * scipy.pi**2 * time))
    return t - (2 * M * scipy.sqrt(scipy.pi) * f)**(-2 / 5)


def df_percentile(inputData, axis=None):
    """
    Extracts the percentile of the data where the mode occurs and its value.
    
    Parameters
    ----------
    inputData : array_like
        Input data from which to compute the percentile and mode.
    axis : int or None, optional
        Axis along which to compute the percentile and mode. If None, the percentile and mode are computed for the entire array. Default is None.
    
    Returns
    -------
    data_prct : ndarray
        Array containing the computed percentiles of the mode for each axis. If axis is not None, the shape is determined by removing the specified axis from the input data shape.
    val : ndarray
        Array containing the values of the mode for each axis. If axis is not None, the shape is determined by removing the specified axis from the input data shape.
    
    Notes
    -----
    This function is used to determine the filtering level for DF/F extraction. Please note that the computation can be inaccurate for short traces.
    """
    if axis is not None:
        def fnc(x):
            return df_percentile(x)

        result = np.apply_along_axis(fnc, axis, inputData)
        data_prct = np.take(result, 0, axis=axis)
        val = np.take(result, 1, axis=axis)
    else:
        # Create the function that we can use for the half-sample mode
        err = True
        while err:
            try:
                bandwidth, mesh, density, cdf = kde(inputData)
                err = False
            except:
                warnings.warn('Percentile computation failed. Duplicating and trying again.')
                if not isinstance(inputData, list):
                    inputData = inputData.tolist()
                inputData += inputData
            else:
                data_prct = cdf[np.argmax(density)] * 100
                val = mesh[np.argmax(density)]
                if data_prct >= 100 or data_prct < 0:
                    warnings.warn('Invalid percentile computed possibly due short trace. Duplicating and recomuputing.')
                    if not isinstance(inputData, list):
                        inputData = inputData.tolist()
                    inputData *= 2
                    err = True
                if np.isnan(data_prct):
                    warnings.warn('NaN percentile computed. Reverting to median.')
                    data_prct = 50
                    val = np.median(np.array(inputData))

    return data_prct, val

