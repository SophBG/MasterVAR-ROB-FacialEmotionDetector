import numpy as np
from skimage.feature import local_binary_pattern, hog

def extract_lbp(img_gray, P=8, R=1, method='uniform', n_bins=59):
    # img_gray: 2D uint8
    lbp = local_binary_pattern(img_gray, P, R, method=method)
    # histogram
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_bins+1), density=True)
    return hist.astype(np.float32)

def extract_hog(img_gray, pixels_per_cell=(8,8), cells_per_block=(2,2), orientations=9):
    # return HOG feature vector
    hog_vec = hog(img_gray, orientations=orientations, pixels_per_cell=pixels_per_cell,
                  cells_per_block=cells_per_block, block_norm='L2-Hys', feature_vector=True)
    return hog_vec.astype(np.float32)
