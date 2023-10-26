"""
Functions for complete extraction (segmentation and dfof)
"""

try:
    import cv2
except ImportError as e:
    raise ImportError(
        f"{e}\n\nSignal extraction package requires cv2 (opencv). "
        "Install cv2 using conda: `conda install -c conda-forge opencv`."
    )
import numpy as np
from scipy.ndimage.measurements import center_of_mass as com
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

from axolotl import utils
from axolotl.image_processing.extraction.roi_detect import watershed_rois
from axolotl.image_processing.extraction.signal_extract import \
    extract_rolling_dfof
from axolotl.traces.filter import butter_lowpass_filter


def locally_weighted_proj(movie, axis=0, hwin=17, wwin=27, method='std', mmethod='mean'):
    """
    Compute locally weighted projection for each pixel in a movie.

    Parameters
    ----------
    movie : 3D - ndarray, shape (T, Y, X)
        The movie to compute LWP on.
    axis : int, optional
        The axis to project over.
    hwin : int, optional
        The size of the window in the temporal domain.
    wwin : int, optional
        The size of the window in the spatial domain.
    method : callable or str, optional
        Function used to calculate the temporal mean.
    mmethod : callable or str, optional
        Function used to calculate the spatial mean.

    Returns
    -------
    ndarray, shape (T, Y, X)
        The locally weighted projection for each pixel in the movie.
    """
    if isinstance(method, str):
        method = getattr(np, method)
    if isinstance(mmethod, str):
        mmethod = getattr(np, mmethod)

    assert movie.ndim == 3
    assert (axis == 0) or (axis == -3)

    t = method(movie, axis = axis)
    mt = mmethod(utils.rolling_window(t, window = (hwin, wwin)), axis=(-1, -2))

    hwin_h = (hwin - 1) // 2
    wwin_h = (wwin - 1) // 2
    height = np.broadcast_to(np.ones((hwin_h))[:, None], (hwin_h, mt.shape[1]))
    width = np.broadcast_to(np.ones((wwin_h))[None, :], (t.shape[0], wwin_h))
    mt = np.vstack([height * mt[:1, :], mt, height * mt[-1:, :]])
    return t / np.hstack([width * mt[:, :1], mt, width * mt[:, -1:]])


def watershed_extracted_data_for_database(
    rate,
    timestamps,
    movie,
    proj_method='std',
    proj_kws={},
    # template blurring
    gaussian_blur_template=False,
    gaussian_blur_kws={
        'ksize': (5, 5),
        'sigmaX': 1,
        'sigmaY': 1
    },
    # watershed parameters
    thresh=2, upper_thresh=4,
    lower_thresh=1, min_size=5, sign=-1, edge_limit=5,
    # signal extraction parameters
    bg_subtract=True, nonzero=True,
    filter_type='gaussian',
    filter_before=True, filter_kws={'ksize': 0.08},
    dfof_kws={}, 
    temporal_smooth=False,
    temporal_smooth_kws={},  
    temporal_smooth_factor=10, 
    temporal_smooth_type='butter', 
    bg_dilate=None,
):
    """
    Apply Watershed Roi Detection algorithm and using dfof rolling
    window signal extraction for a particular 2p movie.

    Parameters
    ----------
    rate : float
        Sampling rate of the acquisition system.
    timestamps : ndarray
        The timestamps of the acquired movie.
    movie : ndarray
        3D array representing the 2p movie.
    proj_method : str or callable
        Method for computing the projection of the movie. Can be a string such as "std" or a callable function.
    proj_kws : dict
        Optional keyword arguments passed to the projection method.
    gaussian_blur_template : bool, default False
        Whether to apply a Gaussian blur to the projection image before computing the masks.
    gaussian_blur_kws : dict
        Optional keyword arguments for the Gaussian blur.
    thresh : float, default 2
        Threshold for the Watershed algorithm.
    upper_thresh : float, default 4
        Upper threshold for the Watershed algorithm.
    lower_thresh : float, default 1
        Lower threshold for the Watershed algorithm.
    min_size : int, default 5
        Minimum size of a mask.
    sign : int, default -1
        Sign for computing the masks.
    edge_limit : int, default 5
        Edge limit for the Watershed algorithm.
    bg_subtract : bool, default True
        Whether to subtract a rolling background from the signal.
    nonzero : bool, default True
        Whether to remove any pixels with zero values in the mask.
    filter_type : str, default 'gaussian'
        Type of filter to apply to the signal. Can be 'gaussian' or 'median'.
    filter_before : bool, default True
        Whether to filter the movie before computing the dfof or after.
    filter_kws : dict
        Optional keyword arguments for the filter.
    dfof_kws : dict
        Optional keyword arguments for the dfof computation.
    temporal_smooth : bool, default False
        Whether to apply temporal smoothing to the movie before extracting the signals.
    temporal_smooth_kws : dict
        Optional keyword arguments for the temporal smoothing.
    temporal_smooth_factor : float, default 10
        Factor for computing the number of knots for spline smoothing.
    temporal_smooth_type : str, default 'butter'
        Type of temporal smoothing. Can be 'butter' or 'spline'.
    bg_dilate : bool or None, default None
        Whether to dilate the background to increase the area for background estimation. None uses the default value.

    Returns:
    --------
    dict
        A dictionary containing extracted ROI signals with corresponding metadata.
    """
    if temporal_smooth:
        if temporal_smooth_type == 'butter':
            movie = butter_lowpass_filter(movie, temporal_smooth_factor, rate, axis=0, **temporal_smooth_kws)
        elif temporal_smooth_type == 'spline':
            shape = movie.shape
            movie = movie.reshape(movie.shape[0], -1)
            X = timestamps[:, None]
            knots_per_second = temporal_smooth_factor
            n_knots = int(np.round((timestamps.max()-timestamps.min())*knots_per_second, 0))
            spline = SplineTransformer(n_knots, **temporal_smooth_kws)
            linear_model = LinearRegression()
            pipe = make_pipeline(spline, linear_model)
            pipe.fit(X, movie)
            movie = pipe.predict(X)
            movie = movie.reshape(shape)
        else:
            raise NameError(f"Temporal smooth type unknown: {temporal_smooth_type}")
    # project and pass to watershed_rois
    normalized_movie = (
        (movie - np.min(movie))
        / (np.max(movie) - np.min(movie)) * 255
    ).astype(np.uint8)

    if isinstance(proj_method, str):
        proj_method = getattr(np, proj_method)

    proj = proj_method(normalized_movie, axis=0, **proj_kws)

    if gaussian_blur_template:
        proj = cv2.GaussianBlur(
            proj,
            borderType=cv2.BORDER_REPLICATE,
            **gaussian_blur_kws,
        )

    masks = watershed_rois(
        proj,
        thresh=thresh,
        upper_thresh=upper_thresh,
        lower_thresh=lower_thresh,
        min_size=min_size,
        sign=sign,
        edge_limit=edge_limit
    )

    if len(masks):  # skip if no rois exist
        traces, timestamps, rate = extract_rolling_dfof(
            movie, masks, timestamps, rate,
            bg_subtract=bg_subtract,
            nonzero=nonzero,
            filter_type=filter_type,
            filter_before=filter_before,
            filter_kws=filter_kws,
            dfof_kws=dfof_kws, 
            bg_dilate=bg_dilate
        )

    container = []

    for cell_id, mask in enumerate(masks):
        metadata = {
            'com': com(mask),
            'pixels': np.sum(mask),
            'weighted_mask': utils.scale(
                mask * proj * -sign, has_nan=False
            )
        }
        label = f"watershed_cell{cell_id}"

        container.append(dict(
            metadata=metadata,
            label=label,
            cell_id=cell_id,
            mask=mask,
            rate=rate,
            timestamps=timestamps,
            signal=traces[:, cell_id]
        ))

    return {'Roi': container}


def handdrawn_data_for_database(
    rate,
    timestamps,
    movie,
    proj_method='mean',
    proj_kws={},
    # template blurring
    gaussian_blur_template=False,
    gaussian_blur_kws={
        'ksize': (5, 5),
        'sigmaX': 1,
        'sigmaY': 1
    },
    # signal extraction parameters
    bg_subtract=True, nonzero=True,
    filter_type='gaussian',
    filter_before=True, filter_kws={'ksize': 0.08},
    dfof_kws={}
):
    """
    Extracts hand-drawn ROIs from a movie and returns a dictionary that can be stored
    in a database. For each ROI, the function computes metadata such as center of mass
    and number of pixels, and extracts the signal using a rolling delta F/F algorithm.

    Parameters:
    -----------
    rate : float
        Sampling rate of the movie, in Hz.
    timestamps : array-like
        Timestamps of each frame in the movie, in seconds.
    movie : array-like
        3D array representing the movie, with shape (frames, height, width).
    proj_method : str or callable, optional
        Method to use for projection of the movie along the time axis. Can be a string
        with the name of a numpy function (e.g. 'mean', 'max'), or a callable that takes
        a 3D array and returns a 2D array. Defaults to 'mean'.
    proj_kws : dict, optional
        Additional keyword arguments to pass to the projection method.
    gaussian_blur_template : bool, optional
        Whether to apply a Gaussian blur to the projection before segmentation.
        Defaults to False.
    gaussian_blur_kws : dict, optional
        Keyword arguments to pass to the cv2.GaussianBlur function, if gaussian_blur_template
        is True.
    bg_subtract : bool, optional
        Whether to subtract a background estimate from the movie frames before computing
        delta F/F. Defaults to True.
    nonzero : bool, optional
        Whether to threshold the movie frames to remove negative values before computing
        delta F/F. Defaults to True.
    filter_type : str, optional
        Type of filter to apply to the movie frames before computing delta F/F. Can be
        'gaussian' or 'uniform'. Defaults to 'gaussian'.
    filter_before : bool, optional
        Whether to apply the filter before or after background subtraction. Defaults to True.
    filter_kws : dict, optional
        Keyword arguments to pass to the filter function, if filter_type is 'gaussian'.
    dfof_kws : dict, optional
        Keyword arguments to pass to the extract_dfof function.

    Returns:
    --------
    dict
        A dictionary with a single key 'Roi', whose value is a list of dictionaries, one
        for each ROI detected in the movie. Each ROI dictionary has the following keys:
            - 'metadata': a dictionary with metadata about the ROI, such as 'com' (center
              of mass) and 'pixels' (number of pixels).
            - 'label': a string identifying the ROI, of the form 'handrawn_cellX', where X
              is an integer.
            - 'cell_id': the integer X mentioned above.
            - 'mask': a boolean mask with the same shape as the movie, indicating which
              pixels belong to the ROI.
            - 'rate': the sampling rate used to compute the signal.
    """
    from .hand_drawn_roi_segmentation import hdroi

    # project and pass to watershed_rois
    normalized_movie = (
        (movie - np.min(movie))
        / (np.max(movie) - np.min(movie)) * 255
    ).astype(np.uint8)

    if isinstance(proj_method, str):
        proj_method = getattr(np, proj_method)

    proj = proj_method(normalized_movie, axis=0, **proj_kws)

    if gaussian_blur_template:
        proj = cv2.GaussianBlur(
            proj,
            borderType=cv2.BORDER_REPLICATE,
            **gaussian_blur_kws,
        )

    names, masks = hdroi(proj)
    names = (np.array(names) - np.min(names)).astype(int)

    if len(masks):  # skip if no rois exist
        traces, timestamps, rate = extract_rolling_dfof(
            movie, masks, timestamps, rate,
            bg_subtract=bg_subtract,
            nonzero=nonzero,
            filter_type=filter_type,
            filter_before=filter_before,
            filter_kws=filter_kws,
            dfof_kws=dfof_kws
        )

    container = []

    for cell_id, mask in zip(names, masks):
        metadata = {
            'com': com(mask),
            'pixels': np.sum(mask)
        }
        label = f"handrawn_cell{cell_id}"

        container.append(dict(
            metadata=metadata,
            label=label,
            cell_id=cell_id,
            mask=mask,
            rate=rate,
            timestamps=timestamps,
            signal=traces[:, cell_id]
        ))

    return {'Roi': container}
