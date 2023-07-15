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
num_hid = 64
num_head = 2
num_feed_forward = 128
source_maxlen = 100
target_maxlen = 100
num_layers_enc = 4
num_layers_dec = 1
num_classes = 10

model = Transformer(
    num_hid=num_hid,
    num_head=num_head,
    num_feed_forward=num_feed_forward,
    source_maxlen=source_maxlen,
    target_maxlen=target_maxlen,
    num_layers_enc=num_layers_enc,
    num_layers_dec=num_layers_dec,
    num_classes=num_classes
)
model.eval()
model(a["inputs_embeds"], b)


# print(asl_dataset.__len__())
