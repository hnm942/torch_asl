import numpy as np

class LipPoints:
    def __init__(self):
         # for lip points
        self.lip_points= np.concatenate([
            [0],
            [13],
            [14],
            [17], # MID POINTS
            [267, 269, 270, 409, 291], # LEFT UPPER
            [312, 311, 310, 415, 308], # LEFT MID_UPPER
            [317, 402, 318, 324], # LEFT MID_LOWER
            [314, 405, 321, 375], # LEFT LOWER
            [37, 39, 40, 185, 61],
            [82, 81, 80, 191, 78],
            [87, 178, 88, 95],
            [84, 181, 91, 146]    ])
        self.flaten_lip_points = np.array([0, 61, 185, 40, 39, 37, 267, 269, 270, 409, 291, 146, 91, 181, 84, 17, 314, 405, 321, 375, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308])
        self.simple_lip_points = np.concatenate([
            [ 0],
            [13],
            [14],
            [17],
            [291], [267],  # LEFT
            [308], [312],
                [317],
                [314],
            [61], [37],  # RIGHT
            [78], [82],
                [87],
                [84],
        ])
        self.lip_dict = {v: i for i, v in enumerate(self.lip_points)}
        self.lip_angles = np.array([
                *[[0, 37, 61, 84, 17, 314, 291, 267, 0, 37][i:i + 3] for i in range(8)],
                *[[13, 82, 78, 87, 14, 317, 308, 312, 13, 82][i:i + 3] for i in range(8)],
            ])
        self.flatten_lip_angles  = np.array([[self.lip_dict[self.flaten_lip_points[self.lip_dict[_]]] for _ in _] for _ in self.lip_angles])
        self.new_lip_angles = np.array([[self.lip_dict[_] for _ in _]] for _ in self.lip_angles)

class HandLandmark:
    def __init__(self):
        self.hand_routes = [[0, *range(1, 5)],
                            [0, *range(5, 9)],
                            [0, *range(9, 13)],
                            [0, *range(13, 17)]]
        self.hand_angles = np.array(sum([[route[i:i+3] for i in range(len(route) - 1)] for route in self.hand_routes], []))
        self.hand_edges = np.array(sum([[route[i:i+1] for i in range(len(route) -1)] for route in self.hand_routes], []))
        self.hand_trees = sum([[np.array(route[i:]) for i in range(len(route))] for route in self.hand_routes], [])

class BodyLandmark:
    def __init__(self):
        self.body_indices = np.array([16, 14, 12, 11, 13, 15])
        self.body_dict = {v: i for i, v in enumerate(self.body_indices)}
        self.body_indices = self.body_indices + 468 + 21
        self.body_angles = np.array(
            [[16, 14, 12],
             [14, 12, 11],
             [11, 13, 15],
             [13, 11, 12],]
        )
        self.body_angles = np.array([self.body_dict[_] for _ in _] for _ in self.body_angles)
        self.body_edges = np.array(
            [[16, 14], [14, 12],
             [12, 11],
             [11, 13], [13, 15]],
        )
        self.body_edges = np.array([[self.body_dict[_] for _ in _] for _ in self.body_egdes])


class EyeLandmark:
    def __init__(self):
        self.left_indices = np.concatenate(
            [[263, 466, 388, 387, 386, 385, 384, 398, 362],
             [249, 390, 373, 374, 380, 381, 382]]
        )
        self.right_indices = np.concatenate(
            [[33, 246, 161, 160, 159, 158, 157, 173, 133],
             [7, 163, 144, 145, 153, 154, 155],]
        )
        self.eye_indices = np.concatenate(self.left_indices, self.right_indices)
