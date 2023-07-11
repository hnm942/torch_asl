import numpy as np
import pandas as pd
import torch
from torch.utils import data
from models.utils.config import ASLConfig
from models.data import preprocess, augmentation, landmark_indices
import torch.nn.functional as F

class AslDataset(data.Dataset):
    def __init__(self, df, npy_path, cfg, phase = "train"):
        self.df = df
        self.max_len = cfg.max_position_embeddings  - 1
        self.phase = phase
        #load npy:
        print("load data in: {}".format(npy_path))
        self.npy = np.load(npy_path)[:, 1:]
        self.npy = self.npy.reshape(-1, 82, 3)
        print("load data successful with shape {}".format(self.npy.shape))
        self.hand_landmarks = landmark_indices.HandLandmark()
        self.lip_landmarks = landmark_indices.LipPoints()
    def normalize(self, data):
        ref = data.flatten()
        ref = data[~data.isnan()]
        mu, std = ref.mean(), data.std()
        return (data - mu) / std
    
    def get_landmarks(self, data):
        landmarks = self.npy[data.idx:data.idx + data.length]
        # augumentation if training
        if self.phase == "train":
            # random interpolation
            landmarks = augmentation.aug2(landmarks)
            # random rotate left hand and right hand
            landmarks[:, -42:-21] = augmentation.random_hand_op_h4(landmarks[:, -42:-21])
            landmarks[:,-21: ] = augmentation.random_hand_op_h4(landmarks[:, -21:])
        # convert data to torch
        landmarks = torch.tensor(landmarks.astype(np.float32))
        # sereparate part of body
        lip, lhand, rhand = landmarks[:, :-42], landmarks[:, -42:-21], landmarks[:, -21:]
        if self.phase == "train":
            if np.random.rand() < 0.5:
                lhand, rhand = rhand, lhand
                lhand[..., 0] *= -1
                rhand[..., 0] *= -1
                lip[:, 4:22], lip[:, 22:40] = lip[:, 22:40], lip[:, 4:22]
                lip[..., 0] *= -1
        # check nan 
        lhand = lhand if lhand.isnan().sum() < rhand.isnan().sum() else preprocess.flip_hand(lip, rhand)
        # combine it together
        landmarks = self.norm(torch.cat([lip, lhand], 1))
        # Motion features consist of future motion and history motion,
        offset = torch.zeros_like(pos[-1:])
        movement = pos[:-1] - pos[1:]
        dpos = torch.cat([movement, offset])
        rdpos = torch.cat([offset, -movement])

        ld = torch.linalg.vector_norm(lhand[:,self.dis_idx0,:2] - lhand[:,self.dis_idx1,:2], dim = -1)
        lipd = torch.linalg.vector_norm(lip[:,self.dis_idx2,:2] - lip[:,self.dis_idx3,:2], dim = -1)

        lsim = F.cosine_similarity(lhand[:,self.hand_landmarks.hand_angles[:,0]] - lhand[:,self.hand_landmarks.hand_angles[:,1]],
                                   lhand[:,self.hand_landmarks.hand_angles[:,2]] - lhand[:,self.hand_landmarks.hand_angles[:,1]], -1)
        lipsim = F.cosine_similarity(lip[:,self.lip_landmarks.lip_angles[:,0]] - lip[:,self.lip_landmarks.lip_angles[:,1]],
                                     lip[:,self.lip_landmarks.lip_angles[:,2]] - lip[:,self.lip_landmarks.lip_angles[:,1]], -1)
        pos = torch.cat([
            pos.flatten(1),
            dpos.flatten(1),
            rdpos.flatten(1),
            lipd.flatten(1),
            ld.flatten(1),
            lipsim.flatten(1),
            lsim.flatten(1),
        ], -1)
        pos = torch.where(torch.isnan(pos), torch.tensor(0.0, dtype = torch.float32).to(pos), pos)
        if len(pos) > self.max_len:
            pos = pos[np.linspace(0, len(pos), self.max_len, endpoint = False)]
        return pos

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, indices):
        data = self.df.iloc[indices] # row in file csv
        landmark = self.get_landmarks(data) 
        
        
        return {}
    
config = ASLConfig(max_position_embeddings= 90)
# create df in numpy
npy_path = "/workspace/data/asl_numpy_dataset/train_landmarks/train_npy.npy"
df = pd.read_csv("/workspace/data/asl_numpy_dataset/train.csv")
asl_dataset = AslDataset(df, npy_path, config)
asl_dataset.__getitem__(0)
# print(asl_dataset.__len__())
