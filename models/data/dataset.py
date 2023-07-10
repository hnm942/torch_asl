import numpy as np
import pandas as pd
import torch
from torch.utils import data
from models.utils.config import ASLConfig
from models.data import preprocess, augmentation
class AslDataset(data.Dataset):
    def __init__(self, df, npy_path, cfg, phase = "train"):
        self.df = df
        self.max_len = cfg.max_position_embeddings  - 1
        self.phase = phase
        #load npy:
        print("load data in: {}".format(npy_path))
        self.npy = np.load(npy_path)
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
