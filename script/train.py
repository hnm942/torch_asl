import numpy as np
import pandas as pd
from models.utils.config import ASLConfig
from models.data.dataset import AslDataset
from models.transformer.als_transformer import Transformer

config = ASLConfig(max_position_embeddings= 90)
# create df in numpy
npy_path = "/workspace/data/asl_numpy_dataset/train_landmarks/5414471.parquet.npy"
df = pd.read_csv("/workspace/data/asl_numpy_dataset/train.csv")
character_to_prediction_index_path = "/workspace/data/asl_numpy_dataset/character_to_prediction_index.json"
asl_dataset = AslDataset(df, npy_path, character_to_prediction_index_path, config)
a, b = asl_dataset.__getitem__(0)
model = Transformer()
model.eval()
model(a["inputs_embeds"])


# print(asl_dataset.__len__())
