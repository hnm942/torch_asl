import numpy as np
import pandas as pd
import torch
from torch.utils import data

class AslDataset(data.Dataset):
    def __init__(self, df, cfg, phase):
        self.df = df
        self.max_len = cfg.config.max_position_embeddings  - 1
        self.phase = phase
        self.dis_idx0, self.dis_idx1 = torch.where(torch.triu(torch.ones((21, 21)), 1) == 1)
        self.dis_idx2, self.dis_idx3 = torch.where(torch.triu(torch.ones((20, 20)), 1) == 1)

    def normalize(self, data):
        ref = data.flatten()
        ref = data[~data.isnan()]
        mu, std = ref.mean(), data.std()
        return (data - mu) / std
    
    def get_landmarks(self, data):
        pass

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, indices):
        data = self.df.iloc[indices]
        return super().__getitem__(indices)
    
# data_path = "/workspace/data/asl-short-dataset/train.csv"
# data = pd.read_csv(data_path)
# print(data.iloc[0])


