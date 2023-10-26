"""a bunch of functions for filtering signals
"""

from scipy.signal import butter, filtfilt, iirfilter, lfilter


def butter_lowpass(cutoff, fs, order=5):
    """
    Butterworth lowpass filter design.

    Parameters:
        cutoff (float): Cutoff frequency of the filter.
        fs (float): Sampling frequency.
        order (int): Order of the filter. Default is 5.

    Returns:
        tuple: Numerator (b) and denominator (a) coefficients of the filter transfer function.
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5, axis=-1, **kwargs):
    """
    Apply a Butterworth lowpass filter to the input data.

    Parameters:
        data (numpy.ndarray): Input data to be filtered.
        cutoff (float): Cutoff frequency of the filter.
        fs (float): Sampling frequency.
        order (int): Order of the filter. Default is 5.
        axis (int): Axis along which the filter is applied. Default is -1.
        **kwargs: Additional keyword arguments to be passed to the filtfilt function.

    Returns:
        numpy.ndarray: Filtered output data.
    """
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data, axis=axis, **kwargs)
    return y


def butter_bandpass(lowcut, highcut, fs, order=5):
    """
    Butterworth bandpass filter design.

    Parameters:
        lowcut (float): Lower cutoff frequency of the filter.
        highcut (float): Upper cutoff frequency of the filter.
        fs (float): Sampling frequency.
        order (int): Order of the filter. Default is 5.

    Returns:
        tuple: Numerator (b) and denominator (a) coefficients of the filter transfer function.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5, axis=-1, **kwargs):
    """
    Apply a Butterworth bandpass filter to the input data.

    Parameters:
        data (numpy.ndarray): Input data to be filtered.
        lowcut (float): Lower cutoff frequency of the filter.
        highcut (float): Upper cutoff frequency of the filter.
        fs (float): Sampling frequency.
        order (int): Order of the filter. Default is 5.
        axis (int): Axis along which the filter is applied. Default is -1.
        **kwargs: Additional keyword arguments to be passed to the filtfilt function.

    Returns:
        numpy.ndarray: Filtered output data.
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data, axis=axis, **kwargs)
    return y


def notchfilt(
    data, dt, band=2, freq=60, ripple=None,
    order=2, filter_type='butter'
):
    """
    from:
    https://stackoverflow.com/questions/35565540
    /designing-an-fir-notch-filter-with-python
    Apply a notch filter to the input data.
    
    Parameters
    ----------
    data : numpy.array
        data samples
    dt : float
        time between samples
    band : float
        bandwidth around the centerline frequency
    freq : float
        centerline frequency
    ripple : float
        maximum passband ripplethat is allowed in dB
    order : int
        filter order
    filter_type : str
        'butter', 'bessel', 'cheby1', 'cheby2', 'ellip'
    """

    fs = 1 / dt
    nyq = fs / 2.0
    low = freq - band / 2.0
    high = freq + band / 2.0
    low = low / nyq
    high = high / nyq
    b, a = iirfilter(
            order, [low, high], rp=ripple, btype='bandstop',
            analog=False, ftype=filter_type)
    filtered_data = lfilter(b, a, data)
    return filtered_data
