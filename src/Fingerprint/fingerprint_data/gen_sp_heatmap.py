
import numpy as np
from scipy.stats import multivariate_normal
import cv2


if __name__ == '__main__':
    x, y = np.mgrid[0:512:1, 0:512:1]
    pos = np.empty(x.shape + (2,))

    pos[:, :, 0] = x
    pos[:, :, 1] = y

    rv = multivariate_normal([50, 100], [[5, 0], [0, 5]])
    pdf = rv.pdf(pos)
    pdf = pdf / (np.max(pdf))

    pdf = np.expand_dims(pdf,axis=2)

    cv2.imshow('omg',pdf)
    cv2.waitKey()
