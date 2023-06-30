import numpy as np




# for lip points
lip_points= np.concatenate([
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
    [84, 181, 91, 146]
])
flaten_lip_points = np.array([0, 61, 185, 40, 39, 37, 267, 269, 270, 409, 291, 146, 91, 181, 84, 17, 314, 405, 321, 375, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308])
simple_lip_points = np.concatenate([
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
lip_dict = {v: i for i, v in enumerate(lip_points)}
lip_angles = np.array([
        *[[0, 37, 61, 84, 17, 314, 291, 267, 0, 37][i:i + 3] for i in range(8)],
        *[[13, 82, 78, 87, 14, 317, 308, 312, 13, 82][i:i + 3] for i in range(8)],
    ])
flatten_lip_angles  = np.array([[lip_dict[flaten_lip_points[lip_dict[_]]] for _ in _] for _ in lip_angles])
new_lip_angles = np.array([[lip_dict[_] for _ in _]] for _ in lip_angles)
