"""
Class for temporal filter extraction

preprocess_traces()

TemporalFilterExtraction()

_handle_Xy
_retrieve_filter
_x_regression
fit(self, X, y, fit_nonlinearity=False)
predict(self, X)
get_y_pred(self, X)
get_y_filtered
get_nonlinearity
"""

import numpy as np
from scipy.fftpack import fft, ifft
from scipy.signal import lfilter, decimate
from scipy.stats import binned_statistic
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

from axolotl.utils.windowing import simple_rolling_window
from sklearn.base import RegressorMixin, BaseEstimator


def preprocess_traces(
    traces, stimulus, timestamps=None,
    rate=None, start_time=0, end_time=None,
    downsampling_factor=10,
    mean_center=True, verbose=True
):
    """
    preprocess traces (time series data ) for temporal filter extraction

    Parameters
    ----------
    traces : array-like, shape (n_samples, n_features)
        The trace data to preprocess.
    stimulus : array-like, shape (n_samples, n_stim_features)
        The stimulus data to preprocess.
    timestamps : array-like or None, optional (default=None)
        The timestamps for each sample in `traces`.
    rate : float or None, optional (default=None)
        The sampling rate of `traces` in Hz. If `None`, it will be inferred from `timestamps`.
    start_time : float, optional (default=0)
        The start time of the traces to keep in seconds.
    end_time : float or None, optional (default=None)
        The end time of the traces to keep in seconds. If `None`, it will be set to the maximum timestamp.
    downsampling_factor : int, optional (default=10)
        The factor to downsample the traces and stimulus by. Set to 1 to keep original sampling rate.
    mean_center : bool, optional (default=True)
        Whether to mean-center the traces and stimulus.
    verbose : bool, optional (default=True)
        Whether to print information about the returned values.

    Returns
    -------
    cut_traces : array-like, shape (n_samples, n_features)
        The preprocessed trace data.
    cut_stimulus : array-like, shape (n_samples, n_stim_features)
        The preprocessed stimulus data.
    vals : dict
        Dictionary containing the following keys:
        - 'rate': float, the sampling rate in Hz
        - 'cut_trace_mean': array-like, shape (n_features,), the mean of the preprocessed traces
        - 'cut_stimulus_mean': array-like, shape (n_stim_features,), the mean of the preprocessed stimulus
    """

    if rate is None:
        assert timestamps is not None, "need to supply timestamps if rate is None"

        rate = 1/np.mean(np.diff(timestamps)) # changed rate to return in Hz

    elif timestamps is None:
        assert rate is not None, "need to supply rate if timestamps is None"

        timestamps = np.arange(0, traces.shape[0]/rate, 1/rate)

    if downsampling_factor > 1:
        timestamps = timestamps[::downsampling_factor]
        traces = decimate(traces, downsampling_factor)
        stimulus = decimate(stimulus, downsampling_factor,axis=0)

        rate = 1/np.mean(np.diff(timestamps)) # downsampled rate

    if end_time is None:
        end_time = np.max(timestamps) + 1

    assert traces.shape[0] == stimulus.shape[0], 'traces and stimulus not the same size'
    tbool = (timestamps >= start_time) & (timestamps < end_time)
    cut_traces = traces[tbool]
    cut_stimulus = stimulus[tbool]

    if mean_center:
        trace_mean = np.mean(cut_traces, axis=0) # this is a scalar value
        cut_traces -= trace_mean
        stimulus_mean = np.mean(cut_stimulus, axis=0) # this is a 1d array
        cut_stimulus -= stimulus_mean

    else:
        trace_mean = None
        stim_mean = None

    cut_vals = {}
    cut_vals['rate'] = rate
    cut_vals['cut_trace_mean'] = np.squeeze(trace_mean)
    cut_vals['cut_stimulus_mean'] = np.squeeze(stimulus_mean)

    if verbose:
        print('returns cut_vals dict with keys "rate", "cut_trace_mean" '
              'and "cut_stimulus_mean"')

    return cut_traces, cut_stimulus, cut_vals


class TemporalFilterExtraction(BaseEstimator, RegressorMixin):
    """
    sklearn-compatible temporal filter extraction class.
    Based on Bacchus and Meister (2002); Behnia et al. (2014)
    """

    def __init__(
        self, rate,
        filter_time=1,
        window_time=0.1,
        eps=0.01, zeropad=1, normalize_autocov=True,
        nonlin_method=None,
        fit_spatial=True, # specific method to weight spatial contribution
        **nonlin_kwargs
    ):
        """
        """

        assert isinstance(zeropad, int)

        self.filter_time = filter_time
        self.window_time = window_time
        self.eps = eps
        self.fit_spatial = fit_spatial
        self.zeropad = zeropad
        self.normalize_autocov = normalize_autocov
        self.rate = rate
        self.window_size = int(rate * window_time)
        self.filter_size = int(rate * filter_time)
        self.nonlin_method = nonlin_method
        self.nonlin_kwargs = nonlin_kwargs # e.g. statistic = 'mean', bins=50

        self.linear_filter = None

        self.nonlin_params = None # stored as dictionary
        self.bin_means = None
        self.bin_edges = None

    def _handle_Xy(self, X, y):
        """method used in fit
        """

        # cast as numpy arrays
        X = np.asarray(X)
        y = np.asarray(y)

        # check dimensionality of input and output
        assert y.ndim == 1, 'y must be one-dimensional'

        if X.ndim == 2:
            y = y[:, None]

        return X, y

    def _retrieve_filter(self, X, y):
        """method used in fit
        this is where the covariance and autocovariance are calculated
        """

        # rolling window dimensions: windows x filter_size x features/targets
        y_rolled = simple_rolling_window(
            y, self.filter_size, self.window_size, axis=0)
        X_rolled = simple_rolling_window(
            X, self.filter_size, self.window_size, axis=0)

        # FFT of all windows
        fft_traces = fft(y_rolled, axis=1, n=y_rolled.shape[1]*self.zeropad)
        fft_stimulus = fft(X_rolled, axis=1, n=X_rolled.shape[1]*self.zeropad)

        cov = np.mean(
            fft_traces * np.conj(fft_stimulus),
            axis=0)

        autocov_stim = np.mean(
            fft_stimulus * np.conj(fft_stimulus),
            axis=0)

        if self.normalize_autocov:

            autocov_stim_eps = (
                autocov_stim
                + self.eps
                * np.abs(np.mean(autocov_stim, axis=0, keepdims=True))
            )

            autocov_normalization_factor = (
                np.max(np.abs(autocov_stim), axis=0, keepdims=True)
                /np.max(np.abs(autocov_stim_eps), axis=0, keepdims=True)
            )

            autocov_normalized = (
                autocov_stim_eps
                * autocov_normalization_factor
            )

        else:
            autocov_normalized = autocov_stim

        crosscorr = cov/autocov_normalized

        linear_filter = ifft(
            crosscorr, n=crosscorr.shape[0]//self.zeropad,
            axis=0
            )

        return linear_filter

    def _x_regression(self, X, y):
        """used in fit if X is two dimensional
        each spatial component is weighed as well
        """

        if self.fit_spatial and X.ndim == 2:
            #this essentially does spatio-temporal filter extraction
            y_pred = self.get_y_filtered(X)

            # regress over convolved linear prediction
            x_weights, _, _, _ = np.linalg.lstsq(
                y_pred, y, rcond=None)

            return x_weights

        elif not self.fit_spatial and X.ndim == 2:

            return np.ones(X.shape[1])

    def _sigmoid(self, x, k, L, x0, b):
        """
        parameterized method used for get_nonlinearity

        sig = L / (1 + np.exp(-k*(x-x0))) - b

        Parameters
        ----------
        x: array
        x0: the x-value of the sigmoid's midpoint
        L: the curve's maximum value
        k: the logistic growth rate or steepness of the curve
        b: the y-axis (vertical) offset

        Returns
        -------
        sig: array of function outputs
        """

        sig = L / (1 + np.exp(-k*(x-x0))) - b

        return sig

    def _softplus(self, x, a, b, c, d, k):
        """
        parameterized method used for get_nonlinearity

        soft = c*np.log(1 + np.exp(a*x + b))**k + d

        Parameters
        ----------
        x: array
        a: controls sharpness of "elbow"
        b: +ve translates curve to the left
        c: multiplicative angle/slope
        d: vertical shift
        k: exponent makes more polynomial

        Returns
        -------
        soft: array of function outputs
        """

        soft = c*np.log(1+np.exp(a*x+b))**k + d

        return soft


    def fit(self, X, y, fit_nonlinearity=False):
        """
        fit temporal (and possibly spatial) filter

        Parameters
        ----------
        X : numpy.array
            A one- or two-dimensional array of the stimulus
        y : numpy.array
            A one-dimensional array of the response
        fit_nonlinearity : boolean
            If the nonlinearity should be fit or not.
        """

        X, y = self._handle_Xy(X, y)

        # set filter and cross correlation
        self.linear_filter = self._retrieve_filter(X, y) # << want to set this to real

        # extract independent spatial component if necessary
        self.x_weights = self._x_regression(X, y)

        if fit_nonlinearity:
            y_pred = self.get_y_pred(X)
            self.nonlinearity = self.get_nonlinearity(
                    np.squeeze(y), y_pred)

        else:
            self.nonlinearity = None

        self.complete_filter = (
            np.real(self.linear_filter)
            * np.real(self.x_weights).T)

        self.temporal_filter = np.real(
            np.dot(self.linear_filter, self.x_weights))

        return self

    def predict(self, X):
        """
        """
        if self.linear_filter is None:
            raise Exception("fit before predicting")

        X = np.asarray(X)
        y_pred = self.get_y_pred(X)

        if self.nonlinearity is None:
            return y_pred
        else:
            return self.nonlinearity(y_pred)

    def get_y_pred(self, X):
        """get linear prediction after fitting
        """
        y_pred = self.get_y_filtered(X)

        if self.x_weights is not None:
            y_pred = np.dot(y_pred, self.x_weights)
            y_pred = np.squeeze(y_pred)

        return np.real(y_pred)

    def get_y_filtered(self, X):
        """Do linear multiplication in space and convolution in time
        """
        if self.linear_filter.ndim == 2:
            return np.array([
                lfilter(ifilter, 1, iX)
                for ifilter, iX in zip(self.linear_filter.T, X.T)
            ]).T
        else:
            return lfilter(self.linear_filter, 1, X, axis=0)

    def get_nonlinearity(self, y, y_pred):
        """get a nonlinear function from a linear prediction

        Returns
        -------
        nonlinearity : a callable function
        """

        if self.nonlin_method is None or self.nonlin_method == 'interp':

            y, y_pred, _ = binned_statistic(
                y_pred, y, **self.nonlin_kwargs)

            # remove NaNs from binned_statistic!! y is the bin means and y_pred
            # is the bin edges...while there are no NaNs in the bin edges array,
            # there can b NaNs in the bin means array...this is problematic because
            # calling interp1d with NaNs present in input values
            # results in undefined behaviour.

            # for a more comprehensive interpolation, see utils.interpolation
            # interp_to_finites
            mask = np.isnan(y)
            y[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), y[~mask])

            y_pred = np.mean([y_pred[:-1], y_pred[1:]], axis=0)

            nonlinearity = interp1d(
                y_pred, y,
                bounds_error=False, fill_value='extrapolate'
            )

            #x = x[~numpy.isnan(x)]



        elif self.nonlin_method == 'sigmoid':
            # This was included as an alternative to interp1d so that we could
            # have a handle on the nonlinearity parameters

            # y, y_pred, _ = binned_statistic(
            #     y_pred, y, **self.nonlin_kwargs)
            #
            # y_pred = np.mean([y_pred[:-1], y_pred[1:]], axis=0)

            # bin in one line of code
            # note that curve_fit cannot handle NaNs, so bins cannot be too larger
            # empirically 1000 bins is already too large
            bin_means, bin_edges, bin_number = binned_statistic(
                y_pred[:len(y)], y, **self.nonlin_kwargs) # default bins=10

            # this avoids issues with nans
            mask = np.isnan(bin_means)
            bin_means[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), bin_means[~mask])


            # save these values for plotting
            self.bin_means = bin_means
            self.bin_edges = bin_edges
            print('number of bins for nonlinearity: {:.2f}'.format(len(bin_means)))

            # set initial values
            p0 = [1.5,10,1.1,-5]
            xdata = bin_edges[:-1]
            ydata = bin_means

            mask = np.isnan(ydata)
            ydata[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), ydata[~mask])

            print('>> evaluate sigmoid')

            # this weights the curvefit such that the higher values are weighted less
            sigma = np.ones(len(ydata))
            sigma[:int(3*len(sigma)/4)] = 0.1

            popt, pcov = curve_fit(self._sigmoid, xdata, ydata, p0, sigma = sigma,maxfev=100000)
            #self.popt = popt

            sigma_params={}
            # save these values for database storage
            sigma_params['k'] = popt[0]
            sigma_params['L'] = popt[1]
            sigma_params['x0'] = popt[2]
            sigma_params['b'] = popt[3]

            # return a function, like interp1d.
            # the lambda function allows "currying"
            nonlinearity = lambda x: self._sigmoid(x,**sigma_params)

            self.nonlin_params = sigma_params

        elif self.nonlin_method == 'softplus':

            bin_means, bin_edges, bin_number = binned_statistic(
                y_pred[:len(y)], y, **self.nonlin_kwargs) # default bins=10

            # this avoids issues with nans
            mask = np.isnan(bin_means)
            bin_means[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), bin_means[~mask])

            # save these values for plotting
            self.bin_means = bin_means
            self.bin_edges = bin_edges
            print('number of bins for nonlinearity: {:.2f}'.format(len(bin_means)))

            # set initial values
            p0=[1.1,0.9,1.1,-0.001,1.1] # a,b,c,d,k

            bounds = ([0,-1,-150,-1,0.0001],[500,1,150,1,5])
            xdata = bin_edges[:-1]
            ydata = bin_means

            mask = np.isnan(ydata)
            ydata[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), ydata[~mask])

            # this weights the curvefit such that the higher values are weighted less
            sigma = np.ones(len(ydata))
            sigma[:int(3*len(sigma)/4)] = 0.1

            print('>> evaluate softplus')
            popt, pcov = curve_fit(self._softplus, xdata, ydata, p0,sigma=sigma,bounds=bounds,maxfev=100000)
            #self.popt = popt
            print('p0: ',p0)
            print('popt: ',popt)
            soft_params={}
            # save these values for database storage
            soft_params['a'] = popt[0]
            soft_params['b'] = popt[1]
            soft_params['c'] = popt[2]
            soft_params['d'] = popt[3]
            soft_params['k'] = popt[4]

            # return a function, like interp1d.
            # the lambda function allows "currying"
            nonlinearity = lambda x: self._softplus(x,**soft_params)

            self.nonlin_params = soft_params
        else:
            raise NameError(
                f"nonlin method {self.nonlin_method} does not exist.")


        return nonlinearity
