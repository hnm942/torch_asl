import numpy as np
import pandas as pd
import torch
from torch.utils import data
from models.utils.config import ASLConfig
from models.data import preprocess, augmentation, landmark_indices
import torch.nn.functional as F
import json
import dask.array as da

class AslDataset(data.Dataset):
    def __init__(self, df, npy_path, character_to_prediction_index_path, cfg, device, phase = "train"):
        self.df = df
        self.max_landmark_size = cfg.max_landmark_size 
        self.max_phrase_size = cfg.max_phrase_size
        self.phase = phase
        self.initStaticHashTable(character_to_prediction_index_path)
        #load npy:
        # print("load data in: {}".format(npy_path))
        npy = da.from_npy_stack(npy_path)[:, 1:]
        # print("load npy from dash: ", self.npy.shape)
        self.npy = self.npy.reshape(-1, 82, 3)
        # npy_x, npy_y, npy_z = npy[:, :82][:, :, np.newaxis], npy[:, 82:164][:, :, np.newaxis], npy[:, 164:][:, :, np.newaxis]
        # self.npy = np.concatenate((npy_x, npy_y, npy_z), axis = 2)
        print("load data successful with shape {}".format(self.npy.shape))
        self.hand_landmarks = landmark_indices.HandLandmark()
        self.lip_landmarks = landmark_indices.LipPoints()
        self.dis_idx0, self.dis_idx1 = torch.where(torch.triu(torch.ones((21, 21)), 1) == 1)
        self.dis_idx2, self.dis_idx3 = torch.where(torch.triu(torch.ones((20, 20)), 1) == 1)
        self.device = device
        
    def normalize(self, data):
        ref = data.flatten()
        ref = ref[~torch.isnan(ref)]
        mu, std = ref.mean(), ref.std()
        return (data - mu) / std
    
    def get_landmarks(self, data):
        landmarks = self.npy[data.idx:data.idx + data.length].compute().copy()
        # print("load landmarks with shape: ", landmarks.shape)
        # print("before augmentation and preprocessing: ", landmarks.shape)
        # augumentation if training
        # print(landmarks.shape)
        if self.phase == "train":
            # random interpolation
            landmarks = augmentation.aug2(landmarks)

            # print("after interpolation " ,landmarks.shape)
            # random rotate left hand and right hand
            landmarks[:, -42:-21] = augmentation.random_hand_rotate(landmarks[:, -42:-21], self.hand_landmarks, joint_prob = 0.2, p = 0.8)
            landmarks[:,-21: ] = augmentation.random_hand_rotate(landmarks[:, -21:], self.hand_landmarks, joint_prob = 0.2, p = 0.8)
        # convert data to torch
        landmarks = torch.tensor(landmarks.astype(np.float32))
        # print("convert data to torch shape: ", landmarks.shape)
        # sereparate part of body
        lip, lhand, rhand = landmarks[:, :-40], landmarks[:, -42:-21], landmarks[:, -21:]
        return lip, lhand, rhand
        # print("lip shape: {}, lhand shape: {}, rhand shape: {}".format(lip.shape, lhand.shape, rhand.shape))
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
        landmarks = landmarks.flatten(1)
        landmarks = torch.where(torch.isnan(landmarks), torch.tensor(0.0, dtype = torch.float32).to(landmarks), landmarks)
        if len(landmarks) > self.max_landmark_size:
            # print("before use max len: ",landmarks.shape)
            landmarks = landmarks[np.linspace(0, len(landmarks), self.max_landmark_size, endpoint = False)]
            # print("after use max len: ",landmarks.shape)
        return landmarks
    
    def get_landmarks(self, data):
        landmarks = self.npy[data.idx:data.idx + data.length].compute().copy()
        # print("load landmarks with shape: ", landmarks.shape)
        # print("before augmentation and preprocessing: ", landmarks.shape)
        # augumentation if training
        # print(landmarks.shape)
        if self.phase == "train":
            # random interpolation
            landmarks = augmentation.aug2(landmarks)

            # print("after interpolation " ,landmarks.shape)
            # random rotate left hand and right hand
            landmarks[:, -42:-21] = augmentation.random_hand_rotate(landmarks[:, -42:-21], self.hand_landmarks, joint_prob = 0.2, p = 0.8)
            landmarks[:,-21: ] = augmentation.random_hand_rotate(landmarks[:, -21:], self.hand_landmarks, joint_prob = 0.2, p = 0.8)
        # convert data to torch
        landmarks = torch.tensor(landmarks.astype(np.float32))
        # print("convert data to torch shape: ", landmarks.shape)
        # sereparate part of body
        lip, lhand, rhand = landmarks[:, :-40], landmarks[:, -42:-21], landmarks[:, -21:]
        return lip, lhand, rhand
        # print("lip shape: {}, lhand shape: {}, rhand shape: {}".format(lip.shape, lhand.shape, rhand.shape))
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
        landmarks = landmarks.flatten(1)
        landmarks = torch.where(torch.isnan(landmarks), torch.tensor(0.0, dtype = torch.float32).to(landmarks), landmarks)
        if len(landmarks) > self.max_landmark_size:
            # print("before use max len: ",landmarks.shape)
            landmarks = landmarks[np.linspace(0, len(landmarks), self.max_landmark_size, endpoint = False)]
            # print("after use max len: ",landmarks.shape)
        return landmarks

        # Motion features consist of future motion and history motion,
        # print("[dataloader] offset shape", landmarks[-1:].shape)
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
        
        # print("[dataloader] landmark shape", landmarks.shape)
        # print("[dataloader] dpos shape", dpos.shape)
        # print("[dataloader] rdpos shape", rdpos.shape)
        # print("[dataloader] ld shape", ld.shape)
        # print("[dataloader] lipd shape", lipd.shape)
        # print("[dataloader] lsim shape", lsim.shape)
        # print("[dataloader] lipsim shape", lipsim.shape)
        landmarks = torch.cat([
            landmarks.flatten(1),
            dpos.flatten(1),
            rdpos.flatten(1),
            lipd.flatten(1),
            ld.flatten(1),
            lipsim.flatten(1),
            lsim.flatten(1),
        ], -1)
        # print("landmarks output shape: ", landmarks.shape)
        landmarks = torch.where(torch.isnan(landmarks), torch.tensor(0.0, dtype = torch.float32).to(landmarks), landmarks)
        if len(landmarks) > self.max_len:
            # print("before use max len: ",landmarks.shape)
            landmarks = landmarks[np.linspace(0, len(landmarks), self.max_len, endpoint = False)]
            # print("after use max len: ",landmarks.shape)
        return landmarks

    def initStaticHashTable(self, path):
        with open(path, "r") as f:
            self.char_to_num = json.load(f)
            self.num_to_char = {j: i for i, j in self.char_to_num.items()}
        char = list(self.char_to_num.keys())
        num = list(self.char_to_num.values())
        tensor_num = torch.tensor(num)
        
        self.table = torch.nn.EmbeddingBag(len(char), len(char), sparse=False)
        self.table.weight.data.copy_(tensor_num)
        self.table.register_buffer('offsets', torch.tensor([0, len(char)]))
        
        def lookup_fn(keys):
            return torch.nn.functional.embedding(keys, self.table.weight)
        
        self.table.lookup = lookup_fn


    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, indices):
        data = self.df.iloc[indices] # row in file csv
        landmark = self.get_landmarks(data) 
        if landmark.shape[0] < self.max_landmark_size:
            temp = torch.zeros((self.max_landmark_size, landmark.shape[1]))
            temp[:landmark.shape[0], :] = landmark
            landmark = temp
        attention_mask = torch.zeros(self.max_landmark_size)
        attention_mask[:len(landmark)] = 1
        attention_mask = (attention_mask != 0 )
        phrase = data["phrase"]
        phrase = '#' + phrase + '$'
        phrase = torch.tensor([self.char_to_num[c] for c in phrase])
        target_mask = torch.zeros(self.max_phrase_size)
        target_mask[:phrase.size(0)] = 1
        target_mask = (target_mask != 0)
        phrase = torch.nn.functional.pad(phrase, pad=(0, self.max_phrase_size - phrase.shape[0]))
        landmark = landmark.to(self.device)
        attention_mask = attention_mask.to(self.device) != 0
        phrase = phrase.to(self.device)
        target_mask = target_mask.to(self.device) != 0
        return {"inputs_embeds": landmark, "attention_mask": attention_mask}, {"target": phrase, "target_mask": target_mask}
    
