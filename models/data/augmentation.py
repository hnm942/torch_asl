import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from functools import partial


# basic augment
def random_affine(data, scale  = (0.8, 1.5), shift  = (-0.1, 0.1), degree = (-15, 15), p = 0.5):
    if np.random.rand() < p:
        if scale is not None:
            scale = np.random.uniform(*scale)
            data = scale * data
        if shift is not None:
            shift = np.random.uniform(*shift, (1,1,data.shape[-1]))
            data = data + shift
        if degree is not None:
            degree = np.random.uniform(*degree)
            radian = degree / 180 * np.pi
            c = np.cos(radian)
            s = np.sin(radian)
            rotate = np.array([[c, -s], [s, c]]).T
            data[...,:2] = data[...,:2] @ rotate
    return data

def random_interpolate(data, scale = (0.8, 1.5), shift = (-0.1, 0.1), p = 0.5): 
    if np.random.rand() < p:
        num_frames = len(data)
        scale = np.random.uniform(*scale)
        shift = np.random.uniform(*shift)
        # get original time 
        original_time = np.arange(num_frames) # num frames in 1 action pharse
        interpolate1D = interp1d(original_time, data, axis = 0, fill_value="extrapolate")
        new_time = np.linspace(0 + shift, len(data) - 1 + shift, int(round(num_frames * scale)), endpoint= True)
        data = interpolate1D(new_time).astype(np.float32)
        return data

def random_noise(data, sigma, p = 0.5):
    if np.random.rand() < p:
        data = data + np.random.normal(0, sigma, size = data.shape)
    return data

def random_maskout(data, mask_prob = 0.15, p = 0.5):
    if np.random.rand() < p:
        data[np.random.rand(len(data)) < mask_prob] = 0.5
    return data

def rotate_points(data, center, alpha):
    # convert from degree to radian
    radian = alpha / 180. * np.pi 
    # compute cos, sin 
    c = np.cos(radian)
    s = np.sin(radian)
    # compute rotation matrix
    rotation_matrix = np.array([[c, -s], 
                                [s, c]])
    # transform to center coordinate to rotate
    translate_points = (data - center).reshape(-1, 2)
    rotated_points = np.dot(translate_points, rotation_matrix.T).T.reshape(*data.shape)
    # convert to original coordinate
    return rotate_points + center

# augment hand
def random_hand_rotate(data, hand_landmark, degree = (-4, 4), joint_prob = 0.15, p = 0.5):
    if np.random.rand() < p:
        for tree in hand_landmark.hand_trees:
            if np.random.rand() < joint_prob:
                alpha = np.random.rand(*degree)
                center = data[:, tree[0:1], :2] 
                data[:, tree[1:], :2] = rotate_points(data[:, tree[1:], :2], center, alpha)
    return data

def random_hand_scale(data, hand_landmark, scale = (-0.05, 0.05), joint_prob = 0.15, p = 0.5):
    if np.random.rand() < p:
        for tree in hand_landmark.hand_trees:
            if np.random.rand() < joint_prob:
                percent = np.random.uniform(*scale)
                target = data[:, tree[0:1], :2]
                source = data[:, tree[1:2], :2]
                data[:, tree[1:2], :2] += (target - source) * percent
    return data

def random_hand_shift(data, shift = (-0.01, 0.01), p = 0.5):
    if np.random.rand() < p:
        center = data[:, 0:1, 0] 
        shift = np.random.uniform(*shift)
        data[..., :2] = data[..., :2] + shift
    return data

def random_flip_hand(data, lhand_start, lhand_end, rhand_start, rhand_end, p =0.5):
    if np.random.rand() < p:
        center = data[:, 0:1, :2]
        lhand = data[:, lhand_start:lhand_end]
        rhand = data[:, rhand_start:rhand_end]
        lhand_end[..., 0], rhand[..., 0] = 2 * center - lhand[..., 0], 2 * center -rhand[..., 0]
        data[:, lhand_start:lhand_end], data[:, rhand_start:rhand_end] = lhand, rhand
        return data
    
# augment lip landmarks

def random_lip_scale(data, scale = (0.9, 1.2), p = 0.5):
    if np.random.rand() < p:
        scale = np.random.uniform(*scale)
        data = data * scale
    return data

def random_lip_dropout(data, p = 0.2):
    mask = np.random.rand(data.shapep[1]) < p
    data[:, mask] = np.nan
    return data

def random_lip_flip(data, p = 0.5):
    if np.random.rand() < p:
        center = data[:, 0:1, 0].mean(1, keepdims = True)
        data[:, :, 0] =  2 * center - data[:, :, 0]
        data[:, 4:22] , data[:, 22:40] = data[22:40], data[:, 4:22]
    return data

# fix param of function
random_hand_op_h2 = partial(random_hand_rotate, joint_prob = 0.2, p = 0.8)
random_hand_op_h4 = partial(random_hand_rotate, joint_prob = 0.4, p = 0.9)
random_hand_op_h5 = partial(random_hand_rotate, joint_prob = 0.8, p = 1.0)
random_hand_op_h4_1 = lambda hand: random_hand_scale(random_hand_rotate(hand, joint_prob = 0.4, p = 0.9), joint_prob = 0.4, p = 0.9)

random_lip_op_l1 = partial(random_lip_flip, p = 0.5)
random_lip_op_l2 = partial(random_lip_dropout, p = 0.2)
aug2 = lambda pos: random_interpolate(random_affine(pos, p = 0.8), p = 0.3)
