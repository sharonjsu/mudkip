"""
Filtering of diagonal bands in movies
"""

import warnings
import cv2
import numpy as np
from scipy.ndimage.filters import maximum_filter
from math import pow


def notch_reject_filter(shape, d0=9, u_k=0, v_k=0, filter_type='ideal', order=1):
    """
    Notch reject filtering, remove or attenuate specific frequencies from a signal

    Parameters
    ----------
    shape : int pair
        Dimensions of the filter matrix
    d0 : int, optional 
        Frequency parameter, controls the width ot the frequency notch to be filtered out
    u_k, v_k : float64, optional 
        Horizontal coordinates of the center 
    filter_type : str, optional 
        3 options : 'ideal', 'gaussian' or 'butter'
    order : int, optional
        Power order of the butter filter (useless if not butter filter)

    Returns
    -------
        Filter matrix H
    """
    P, Q = shape
    # Initialize filter with zeros
    H = np.zeros((P, Q))

    # Traverse through filter
    for u in range(0, P):
        for v in range(0, Q):
            # Get euclidean distance from point D(u,v) to the center (u_k, v_k)
            D_uv = np.sqrt((u - P / 2 + u_k) ** 2 + (v - Q / 2 + v_k) ** 2)
            D_muv = np.sqrt((u - P / 2 - u_k) ** 2 + (v - Q / 2 - v_k) ** 2)

            if filter_type == 'gaussian':
                H[u, v] = (1 - np.exp(-0.5 * (D_uv * D_muv / pow(d0, 2))))
                
            elif filter_type == 'butter':
                H[u, v] = (1.0 / (1 + pow((d0 * d0) / (D_uv * D_muv), order)))   
                
            else:
                if D_uv <= d0 or D_muv <= d0:
                    H[u, v] = 0.0
                else:
                    H[u, v] = 1.0

    return H


def detect_peaks(image, filter_relative_size=3, edge_ignore=5):
    """
    Takes an image and detect the peaks using the local maximum filter.

    Parameters
    ----------
    image : 2 dimensional numpy array or PIL
    filter_relative_size : int, optional 
        Size of the neighbourhood
    edge_ignore : int, optional 
        Size of the margin in which to ignore edge artifacts (values set to 0)

    Returns
    -------
        local max : Boolean mask of the peaks (i.e. 1 when the pixel's value is the neighborhood maximum, 0 otherwise)
    """

    # apply the local maximum filter; all pixel of maximal value
    # in their neighborhood are set to 1
    local_max = maximum_filter(
        image,
        size=tuple(s//filter_relative_size for s in image.shape)
    ) == image
    # local_max is a mask that contains the peaks we are looking for

    # we obtain the final mask, containing only peaks, by removing edge artifacts
    local_max[:edge_ignore] = False
    local_max[-edge_ignore:] = False
    local_max[:, :edge_ignore] = False
    local_max[:, -edge_ignore:] = False

    return local_max


class MovieArtifactRemoval:

    def __init__(
        self, movie, 
        # number of frames to use to determine prominent frequency bands
        mags_stop=300,
        # size of notch filter - e.g. for gaussian this corresponds to the SD
        filter_size=5,
        # tolerance for ignoring center strong slow frequency components
        tol_radius=10,
        filter_type='gaussian',
        zeropad=100, 
        # relative size of each peak in shape per pixels
        filter_relative_size=3, 
        edge_ignore=5,
        n_masks=2  # number of expected frequencies to filter - if no match no filter will be applied
    ):
        self.movie = movie
        self.shape = movie.shape
        self.zeropad = zeropad
        self.n_masks = n_masks
        
        # zeropad movie for filtering
        if self.zeropad is not None:
            t, h, w = self.shape
            new_h = self.zeropad * 2 + h
            new_w = self.zeropad * 2 + w
            movie = np.zeros((t, new_h, new_w), dtype=self.movie.dtype)
            movie[:, self.zeropad:-self.zeropad, self.zeropad:-self.zeropad] = self.movie
            self.movie = movie
            self.shape = movie.shape
        
        self.mags_stop = mags_stop
        self.filter_relative_size = filter_relative_size
        self.edge_ignore = edge_ignore
        self.filter_size = filter_size
        self.tol_radius = tol_radius
        self.filter_type = filter_type

        # to calculate
        self._mags = None
        self._pimage = None
        self._maxfreq = None
        self._filtered = None
        # center indices for each frequency band to filter
        self._cidcs = None
        # final notch filter applied in the frequency domain
        self._notch_filter = None

    def empty(self):
        return np.empty(self.shape)

    @property
    def f1(self):
        return np.sort(np.fft.fftfreq(self.shape[-2]))

    @property
    def f2(self):
        return np.sort(np.fft.fftfreq(self.shape[-1]))

    @property
    def F2(self):
        return np.meshgrid(self.f2, self.f1)[0]

    @property
    def F1(self):
        return np.meshgrid(self.f2, self.f1)[1]

    def _mags_routine(self):
        mags = self.empty()
        if self.mags_stop is None:
            movie = self.movie
        else:
            movie = self.movie[:self.mags_stop]
        for i, frame in enumerate(movie):
            fft = np.fft.fft2(frame)
            fshift = np.fft.fftshift(fft)
            mag = 20 * np.log(np.abs(fshift))
            mags[i] = mag
        return mags

    @property
    def mags(self):
        if self._mags is None:
            self._mags = self._mags_routine().mean(0)
        return self._mags

    @property
    def pimage(self):
        # location of peaks as a boolean
        if self._pimage is None:
            pimage = detect_peaks(
                self.mags, 
                filter_relative_size=self.filter_relative_size, 
                edge_ignore=self.edge_ignore
            )
            edge_ignore = self.edge_ignore       
            while np.sum(pimage) <= 1 and edge_ignore:
                edge_ignore = edge_ignore - 1
                pimage = detect_peaks(
                    self.mags, filter_relative_size=self.filter_relative_size, edge_ignore=edge_ignore
                )
            self._pimage = pimage
        return self._pimage
    
    @property
    def cidcs(self):
        if self._cidcs is None:
            # filter routine
            shape = self.pimage.shape  # h and w
            idcs = np.array(np.where(self.pimage))
            middle = np.array(shape) / 2
            # centered indices
            cidcs = idcs.T - middle
            if self.tol_radius is not None:
                accept = np.linalg.norm(cidcs, ord=2, axis=-1) > self.tol_radius
                cidcs = cidcs[accept]  
            self._cidcs = cidcs
        
        return self._cidcs   
    
    @property
    def notch_filter(self):
        if self._notch_filter is None:
            # notch filters
            shape = self.pimage.shape
            notch = np.ones(shape)
            for cidx in self.cidcs:
                notch = notch * notch_reject_filter(
                    shape, self.filter_size, *cidx, filter_type=self.filter_type
                )
            self._notch_filter = notch
        return self._notch_filter              

    def _filter_routine(self):        
        if len(self.cidcs) == 0:
            # no stripes present
            warnings.warn("Nothing to mask in the power spectrum")
            movie = self.movie
        elif (self.n_masks is not None) and len(self.cidcs) != self.n_masks and (self.tol_radius is not None):
            # this indicates that it is not just stripes or no clear stripes present
            warnings.warn(f"Not exactly {self.n_masks} points in the power spectrum to mask: {len(self.cidcs)}")
            movie = self.movie
        else:
            movie = self.empty()
            # filter each frame
            for i, frame in enumerate(self.movie):
                f = np.fft.fft2(frame)
                fshift = np.fft.fftshift(f)
                notch_reject_center = fshift * self.notch_filter
                notch_reject = np.fft.ifftshift(notch_reject_center)
                inverse_notch_reject = np.fft.ifft2(notch_reject)
                movie[i] = frame = np.abs(inverse_notch_reject)
            
        # remove zero padding from movie
        if self.zeropad is not None:
            movie = movie[:, self.zeropad:-self.zeropad, self.zeropad:-self.zeropad]
 
        return movie

    @property
    def filtered(self):
        if self._filtered is None:
            self._filtered = self._filter_routine()
        return self._filtered
