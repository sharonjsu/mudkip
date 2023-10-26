"""functions for roi detection
"""

try:
    import cv2
except ImportError as e:
    raise ImportError(
        f"{e}\n\nSignal extraction package requires cv2 (opencv). "
        "Install cv2 using conda: `conda install -c conda-forge opencv`."
    )
import numpy as np

from axolotl import utils


# TODO make ROI detection class

def watershed_rois(
    proj, thresh=2, upper_thresh=2.5,
    lower_thresh=1.5, min_size=9, sign=-1, edge_limit=0, 
    return_background=False, remove_edge_rois=False
):
    """
    Detect ROIs using watershed.
    Works well for largely separated ROIs and clearly identifiable
    via a threshold.

    Parameters
    ----------
    proj : numpy.array
        projection of movie.
    thresh : float
        threshold for projection for active parts.
    upper_thresh : float
        sure foreground of projection
    lower_thresh : float
        unknown parts of projection
    min_size : int
        mininum size of ROI in pixels.
    sign : {-1, 1}
        If -1 the thresholded images are flipped.
    edge_limit : int
        Subtract this number from the height and width of the mask
        before thresholding the size of the ROI (min_size).
    return_background : bool
        Whether to return mask for background
    remove_edge_rois : bool
        Whether to remove pixels from ROIs that are within the 
        edge_limit.

    Returns
    -------
    masks : numpy.array
        masks of each ROI (ROIs * y * x)
    """
    # height and width of projectiondd
    height, width = proj.shape

    # threshold image
    image = sign * utils.binary_threshold(proj, thresh)
    image = np.uint8(image)

    # sure foreground
    fg = sign * utils.binary_threshold(proj, upper_thresh)
    fg = np.uint8(fg)

    # unknown
    unknown = sign * utils.binary_threshold(proj, lower_thresh)
    unknown = np.uint8(unknown)

    # get connected components
    _, markers = cv2.connectedComponents(fg)
    markers += 1
    markers[unknown == 1] = 0

    # apply watershed
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(image, markers)
    markers[markers == 1] = 0

    # print number of rois
    unique = np.unique(markers)
    print(
        f"Found {len(unique) - 2} ROIs using "
        "watershed before size thresholding."
    )

    # create array of masks
    masks = np.array([
        markers == u
        for u in unique
        if (u > 0) and
        # limiting the edge and size thresholding
        (
            np.sum((markers == u)[
                edge_limit:height-edge_limit,
                edge_limit:width-edge_limit
            ]) > min_size)
        ])
    
    if remove_edge_rois:
        masks[:, :edge_limit] = False
        masks[:, height-edge_limit:] = False
        masks[:, :, :edge_limit] = False
        masks[:, :, width-edge_limit:] = False
        sizes = masks.sum(axis=(1, 2))
        masks = masks[sizes > min_size]

    print(f"{len(masks)} ROIs left.")

    if return_background:
        return masks, markers == 0
    return masks
