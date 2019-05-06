from PIL import Image
import numpy as np
import scipy as sp
import matplotlib.colors as colors
from sklearn.cluster import DBSCAN
from math import ceil, sqrt

"""
Inputs:

    rgbimg:         [M,N,3] numpy array containing (uint, 0-255) color image

    hueleftthr:     Scalar constant to select maximum allowed hue in the
                    yellow-green region

    huerightthr:    Scalar constant to select minimum allowed hue in the
                    blue-purple region

    satthr:         Scalar constant to select minimum allowed saturation

    valthr:         Scalar constant to select minimum allowed value

    monothr:        Scalar constant to select minimum allowed monochrome
                    brightness

    maxpoints:      Scalar constant maximum number of pixels to forward to
                    the DBSCAN clustering algorithm

    proxthresh:     Proximity threshold to use for DBSCAN, as a fraction of
                    the diagonal size of the image

Outputs:

    borderseg:      [K,2,2] Nested list containing K pairs of x- and y- pixel
                    values for drawing the tree border

    X:              [P,2] List of pixels that passed the threshold step

    labels:         [Q] List of cluster labels for points in Xslice (see
                    below)

    Xslice:         [Q,2] Reduced list of pixels to be passed to DBSCAN

"""

def findtree(rgbimg, hueleftthr=0.2, huerightthr=0.95, satthr=0.7, 
             valthr=0.7, monothr=220, maxpoints=5000, proxthresh=0.04):

    # Convert rgb image to monochrome for
    gryimg = np.asarray(Image.fromarray(rgbimg).convert('L'))
    # Convert rgb image (uint, 0-255) to hsv (float, 0.0-1.0)
    hsvimg = colors.rgb_to_hsv(rgbimg.astype(float)/255)

    # Initialize binary thresholded image
    binimg = np.zeros((rgbimg.shape[0], rgbimg.shape[1]))
    # Find pixels with hue<0.2 or hue>0.95 (red or yellow) and saturation/value
    # both greater than 0.7 (saturated and bright)--tends to coincide with
    # ornamental lights on trees in some of the images
    boolidx = np.logical_and(
                np.logical_and(
                  np.logical_or((hsvimg[:,:,0] < hueleftthr),
                                (hsvimg[:,:,0] > huerightthr)),
                                (hsvimg[:,:,1] > satthr)),
                                (hsvimg[:,:,2] > valthr))
    # Find pixels that meet hsv criterion
    binimg[np.where(boolidx)] = 255
    # Add pixels that meet grayscale brightness criterion
    binimg[np.where(gryimg > monothr)] = 255

    # Prepare thresholded points for DBSCAN clustering algorithm
    X = np.transpose(np.where(binimg == 255))
    Xslice = X
    nsample = len(Xslice)
    if nsample > maxpoints:
        # Make sure number of points does not exceed DBSCAN maximum capacity
        Xslice = X[range(0,nsample,int(ceil(float(nsample)/maxpoints)))]

    # Translate DBSCAN proximity threshold to units of pixels and run DBSCAN
    pixproxthr = proxthresh * sqrt(binimg.shape[0]**2 + binimg.shape[1]**2)
    db = DBSCAN(eps=pixproxthr, min_samples=10).fit(Xslice)
    labels = db.labels_.astype(int)

    # Find the largest cluster (i.e., with most points) and obtain convex hull   
    unique_labels = set(labels)
    maxclustpt = 0
    for k in unique_labels:
        class_members = [index[0] for index in np.argwhere(labels == k)]
        if len(class_members) > maxclustpt:
            points = Xslice[class_members]
            hull = sp.spatial.ConvexHull(points)
            maxclustpt = len(class_members)
            borderseg = [[points[simplex,0], points[simplex,1]] for simplex
                          in hull.simplices]

    return borderseg, X, labels, Xslice