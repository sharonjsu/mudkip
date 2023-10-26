"""
Code to handle hand-drawn ROI (hdroi) segmentation
Fully drawn from roipoly: https://github.com/jdoepfert/roipoly.py
"""

import matplotlib
matplotlib.use('Qt5Agg')

from matplotlib import pyplot as plt
import numpy as np
from roipoly import MultiRoi
import cv2


def hdroi(image):
    """
    Allows to interactively draw multiple regions of interests
    Parameters
    ---------
    image: matplotlib image to select ROIs over

    Returns
    --------
    roi_names: int
        names of ROIs
    masks : numpy float64.array
        binary masks,  each row corresponds to a different ROI and each column corresponds to a pixel in the image. 
        The value of each element is 1 if the pixel is inside the ROI, else 0 
    """
    
    # Show the image
    fig = plt.figure()
    plt.imshow(image, interpolation='nearest', cmap="Greys")
    plt.title("Click on the button to add a new ROI")

    # Draw multiple ROIs
    multi_roi = MultiRoi()

    # # Draw all ROIs
    plt.imshow(image, interpolation='nearest', cmap="Greys")
    masks = []
    roi_names = []
    for name, roi in multi_roi.rois.items():
        roi.display_roi()
        roi.display_mean(image)
        masks.append(roi.get_mask(image))
        roi_names.append(int(name))
    plt.show()

    masks = np.stack(masks)
    return roi_names, masks


if __name__ == '__main__':

    image = cv2.imread('test.png', cv2.IMREAD_GRAYSCALE)
    test_image = cv2.equalizeHist(image)
    my_masks = hdroi(test_image)
 
