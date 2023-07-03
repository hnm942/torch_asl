import numpy as np
from scipy.interpolate import interp1d
from functools import partial


# step 1
def flip_hand(lip_points, rhand_points):
    center = lip_points[:, 0:1, 0].mean(1, keepdims = True)
    rhand_points[..., 0] = 2 * center - rhand_points[..., 0]
    return rhand_points

def get_interesting_landmark(df):
    # get lip, left hand, right hand points from data
    lip_points, lhand_points, rhand_points = df[:, :-42], df[:, -42:-21], df[:, -21:]
    # get hand which have less NaN values
    lhand_points = lhand_points if lhand_points.isnan().sum() < rhand_points.isnan().sum() else flip_hand(lip_points, rhand_points)
    return lip_points, lhand_points, rhand_points

# step 2: Augmentation




# step 3: Random