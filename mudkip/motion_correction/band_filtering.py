"""
Filtering of diagonal bands in movies
"""

import warnings
import cv2
import numpy as np
from scipy.ndimage.filters import maximum_filter
from math import pow
from .artifact_removal import notch_reject_filter, detect_peaks


class MovieProcessor:

    def __init__(
        self, movie, mags_stop=300,
        default_maxfreq=1.0,
        subtract=False,
        mask=False,
        sep=5,
        surround=12,
        tol_radius=10,
        filter_type='ideal',
        **kwargs
    ):
        self.movie = movie
        self.min = movie.min()
        self.max = movie.max()
        self.shape = movie.shape
        self.mags_stop = mags_stop
        self.kwargs = kwargs
        self.subtract = subtract
        self.default_maxfreq = default_maxfreq
        self.mask = mask
        self.sep = sep
        self.surround = surround
        self.tol_radius = tol_radius
        self.filter_type = filter_type

        # to calculate
        self._mags = None
        self._pimage = None
        self._maxfreq = None
        self._filtered = None

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

    @property
    def h1(self):
        return np.hamming(self.shape[-2])

    @property
    def h2(self):
        return np.hamming(self.shape[-1])

    def hfilter(self, r):
        return np.sqrt(np.outer(self.h1, self.h2)) ** r

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
        if self._pimage is None:
            pimage = detect_peaks(self.mags, **self.kwargs)
            edge_ignore = self.kwargs.pop('edge_ignore', 5)
            while np.sum(pimage) <= 1 and edge_ignore:
                edge_ignore -= 1
                pimage = detect_peaks(
                    self.mags, **self.kwargs, edge_ignore=edge_ignore
                )
            self._pimage = pimage
        return self._pimage

    @property
    def maxfreq(self):
        if self._maxfreq is None:
            idcs = np.where(self.pimage)
            f1_peaks = self.f1[idcs[0]]
            f2_peaks = self.f2[idcs[1]]
            try:
                maxfreq = np.linalg.norm([f1_peaks, f2_peaks], ord=2, axis=0).max()
            except ValueError:
                maxfreq = 0

            if maxfreq:
                self._maxfreq = maxfreq
            else:
                warnings.warn("Maximum frequency not found!")
                self._maxfreq = self.default_maxfreq
        return self._maxfreq

    @property
    def hhigh(self):
        return self.hfilter(1 / self.maxfreq)

    @property
    def hlow(self):
        return self.hfilter(2 / self.maxfreq)

    def _filter_routine(self):
        if self.mask == 2:
            shape = self.pimage.shape
            idcs = np.array(np.where(self.pimage))
            middle = np.array(shape) / 2
            # centered indices
            cidcs = idcs.T - middle
            if self.tol_radius is not None:
                accept = np.linalg.norm(cidcs, ord=2, axis=-1) > self.tol_radius
                cidcs = cidcs[accept]
            
            if len(cidcs) == 0:
                warnings.warn("Nothing to mask in the power spectrum")
                return self.movie
            if len(cidcs) != 2 and (self.tol_radius is not None):
                # raise ValueError(f"Not exactly two points in the power spectrum to mask: {len(cidcs)}")
                warnings.warn(f"Not exactly two points in the power spectrum to mask: {len(cidcs)}")
                return self.movie
                
            # notch filters
            notch = np.ones(shape)
            for cidx in cidcs:
                notch = notch * notch_reject_filter(shape, self.sep, *cidx, filter_type=self.filter_type)
            
            movie = self.empty()
            for i, frame in enumerate(self.movie):
                f = np.fft.fft2(frame)
                fshift = np.fft.fftshift(f)
                notch_reject_center = fshift * notch
                notch_reject = np.fft.ifftshift(notch_reject_center)
                inverse_notch_reject = np.fft.ifft2(notch_reject)
                movie[i] = frame = np.abs(inverse_notch_reject)
                
            return movie        
            
        elif self.mask:
            #identify the mask
            idcs = np.where(self.pimage)
            middle = np.array(self.pimage.shape) / 2
            print(f"middel location: {middle}")
            masks = []
            surrounds = []
            for hidx, widx in zip(*idcs):
                idx = np.array([hidx, widx])
                if self.tol_radius is not None:
                    if np.linalg.norm(middle - idx, ord=2) < self.tol_radius:
                        continue
                print(f"idx location: {idx}")
                zeros = np.zeros(self.pimage.shape).astype(bool)
                zeros[np.maximum(hidx-self.sep, 0):hidx+self.sep, np.maximum(widx-self.sep, 0):widx+self.sep] = True
                masks.append(zeros.copy())
                zeros[np.maximum(hidx-self.surround, 0):hidx+self.surround, np.maximum(widx-self.surround, 0):widx+self.surround] = True
                surrounds.append(zeros.copy())
                
            if not masks:
                warnings.warn("Nothing to mask in the power spectrum")
                return self.movie
            if len(masks) not in [2] and (self.tol_radius is not None):
                warnings.warn(f"Not exactly two points in the power spectrum to mask: {len(masks)}")
                return self.movie

            movie = self.empty()
            for i, frame in enumerate(self.movie):
                f = cv2.dft(frame.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)
                f_shifted = np.fft.fftshift(f)
                f_complex = f_shifted[:, :, 0]*1j + f_shifted[:, :, 1]
                    
                assert len(masks) <= 2
                # mask power spectrum
                f_filtered = f_complex.copy()
                for mask, surround in zip(masks, surrounds):
                    f_filtered[mask] = np.median(f_filtered[surround & ~mask])

                f_filtered_shifted = np.fft.fftshift(f_filtered)
                inv_img = np.fft.ifft2(f_filtered_shifted)  # inverse F.T.
                filtered_img = np.abs(inv_img)
                movie[i] = filtered_img
            
            movie = np.clip(movie, 0, 2**16-1).astype(np.uint16)
            return movie
            
        else:
            movie = self.empty()
            for i, frame in enumerate(self.movie):
                f = cv2.dft(frame.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)
                f_shifted = np.fft.fftshift(f)
                f_complex = f_shifted[:, :, 0]*1j + f_shifted[:, :, 1]
                f_filtered = self.hhigh * f_complex
                if self.subtract:
                    f_filtered_h = self.hlow * f_complex
                    f_filtered = f_filtered - f_filtered_h

                f_filtered_shifted = np.fft.fftshift(f_filtered)
                inv_img = np.fft.ifft2(f_filtered_shifted)  # inverse F.T.
                filtered_img = np.abs(inv_img)
                movie[i] = filtered_img

            movie = (movie - movie.min()) / (movie.max() - movie.min())
            movie = movie * (self.max - self.min) + self.min

            return movie

    @property
    def filtered(self):
        if self._filtered is None:
            self._filtered = self._filter_routine()
        return self._filtered


def remove_diagonal_bands(movie, repeats=3, zeropad=None, **kwargs):
    """
    Remove diagonal bands for movie

    Parameters
    ----------
    movie : 3D numpy array
        The first dimension is time, and the remaining dimensions are spatial dimensions (height and width)
    repeats : int, optional 
        Number of the type the MovieProcessor filter should be applied to the movie 
    zeropad : int, optional
        Size of the zero-padding to apply to the movie before processing it

    **kwargs : dictionary of additional parameters that can be passed to the MovieProcessor, optional
    
    Results 
    -------
        Returns the movie with the diagonal bands removes
    """
    if zeropad is not None:
        t, h, w = movie.shape
        new_h = zeropad * 2 + h
        new_w = zeropad * 2 + w
        newmovie = np.zeros((t, new_h, new_w), dtype=movie.dtype)
        newmovie[:, zeropad:-zeropad, zeropad:-zeropad] = movie
        movie = newmovie

    for i in range(repeats):
        movie = MovieProcessor(movie, **kwargs).filtered
        
    if zeropad is not None:
        movie = movie[:, zeropad:-zeropad, zeropad:-zeropad]
    
    return movie
