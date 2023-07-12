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
        self.dis_idx0, self.dis_idx1 = torch.where(torch.triu(torch.ones((21, 21)), 1) == 1)
        self.dis_idx2, self.dis_idx3 = torch.where(torch.triu(torch.ones((20, 20)), 1) == 1)

    def normalize(self, data):
        ref = data.flatten()
        ref = data[~data.isnan()]
        mu, std = ref.mean(), data.std()
        return (data - mu) / std
    
    def get_landmarks(self, data):
        landmarks = self.npy[data.idx:data.idx + data.length]
        print("before augmentation and preprocessing: ", landmarks.shape)
        # augumentation if training
        if self.phase == "train":
            # random interpolation
            landmarks = augmentation.aug2(landmarks)
            print("after interpolation " ,landmarks.shape)
            # random rotate left hand and right hand
            landmarks[:, -42:-21] = augmentation.random_hand_rotate(landmarks[:, -42:-21], self.hand_landmarks, joint_prob = 0.2, p = 0.8)
            landmarks[:,-21: ] = augmentation.random_hand_rotate(landmarks[:, -21:], self.hand_landmarks, joint_prob = 0.2, p = 0.8)
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
        landmarks = self.normalize(torch.cat([lip, lhand], 1))
        # Motion features consist of future motion and history motion,
        offset = torch.zeros_like(landmarks[-1:])
        movement = landmarks[:-1] - landmarks[1:]
        dpos = torch.cat([movement, offset])
        rdpos = torch.cat([offset, -movement])
        ld = torch.linalg.vector_norm(lhand[:,self.dis_idx0,:2] - lhand[:,self.dis_idx1,:2], dim = -1)
        lipd = torch.linalg.vector_norm(lip[:,self.dis_idx2,:2] - lip[:,self.dis_idx3,:2], dim = -1)
        lsim = F.cosine_similarity(lhand[:,self.hand_landmarks.hand_angles[:,0]] - lhand[:,self.hand_landmarks.hand_angles[:,1]],
                                   lhand[:,self.hand_landmarks.hand_angles[:,2]] - lhand[:,self.hand_landmarks.hand_angles[:,1]], -1)
        lipsim = F.cosine_similarity(lip[:,self.lip_landmarks.flatten_lip_angles[:,0]] - lip[:,self.lip_landmarks.flatten_lip_angles[:,1]],
                                     lip[:,self.lip_landmarks.flatten_lip_angles[:,2]] - lip[:,self.lip_landmarks.flatten_lip_angles[:,1]], -1)
        landmarks = torch.cat([
            landmarks.flatten(1),
            dpos.flatten(1),
            rdpos.flatten(1),
            lipd.flatten(1),
            ld.flatten(1),
            lipsim.flatten(1),
            lsim.flatten(1),
        ], -1)
        landmarks = torch.where(torch.isnan(landmarks), torch.tensor(0.0, dtype = torch.float32).to(landmarks), landmarks)
        if len(landmarks) > self.max_len:
            landmarks = landmarks[np.linspace(0, len(landmarks), self.max_len, endpoint = False)]
        return landmarks

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, indices):
        data = self.df.iloc[indices] # row in file csv
        landmark = self.get_landmarks(data) 
        attention_mask = torch.zeros(self.max_len)
        attention_mask[:len(landmark)] = 1
        phrase = data["phrase"]
        phrase = '#' + phrase + '$'
        
        phrase = torch.strings.bytes_split(phrase)
        phrase = self.table.lookup(phrase)
        phrase = torch.pad(phrase, paddings=[[0, 64 - torch.shape(phrase)[0]]])  
        return {"inputs_embeds": landmark, "attention_mask": attention_mask}, int(phrase)

    
