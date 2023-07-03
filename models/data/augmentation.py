import numpy as np
import pandas as pd


def random_affine(data, scale = (0.8, 1.5), shift = (-0.1, 0.1), degree = (-15, 15), p = 0.5):
    if np.random.rand() < p:
        if scale is not None:
            scale = np.random.uniform(*scale)
            data = scale * data
        if shift is not None:
            shift = np.random.uniform(*shift)
            data = shift + data
        if degree is not None:
            degree = np.random.uniform(*degree)
            radian = degree / 180.  * np.pi
            c = np.cos(radian)
            s = np.sin(radian)
            rotation_matrix = np.array([[c, -s][s, c]]).T
            data[..., :2] = data[..., :2] @ rotation_matrix
    pass


