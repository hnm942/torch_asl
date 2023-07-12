import numpy as np
import pandas as pd
from models.utils.config import ASLConfig
from models.data.dataset import AslDataset


config = ASLConfig(max_position_embeddings= 90)
# create df in numpy
npy_path = "/workspace/data/asl_numpy_dataset/train_landmarks/5414471.parquet.npy"
df = pd.read_csv("/workspace/data/asl_numpy_dataset/train.csv")
asl_dataset = AslDataset(df, npy_path, config)
asl_dataset.__getitem__(0)
# print(asl_dataset.__len__())
