import numpy as np
import pandas as pd
# for develop
import sys
import torch
from tqdm import tqdm

sys.path.append("/workspace/src/torch_asl")


from models.utils.config import ASLConfig
from models.data.dataset import AslDataset
from models.transformer.als_transformer import Transformer
from torch.utils.data import DataLoader
from torch.utils.data import random_split

config = ASLConfig(max_landmark_size = 96, max_phrase_size = 64)
# create df in numpy
npy_path = "/workspace/data/asl_numpy_dataset/train_landmarks/train_npy"
df = pd.read_csv("/workspace/data/asl_numpy_dataset/train.csv")

character_to_prediction_index_path = "/workspace/data/asl_numpy_dataset/character_to_prediction_index.json"
# a, b = asl_dataset.__getitem__(0)
num_hid = 183
num_head = 3
num_feed_forward = 96
source_maxlen = 96
target_maxlen = 64
num_layers_enc = 4
num_layers_dec = 1
num_classes = 59
learning_rate = 0.01
num_epochs = 50
# print(train_loader.__len__())
# i = 0
# for i, batch in enumerate(train_loader):
#     print(i)
#     # i = i + 1
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

df = df[df['length'] >= 50]
len_data = df.shape[0]
split = 0.3
val_size = int(0.3 * len_data)
train_size = len_data - val_size
shuffled_df = df.sample(frac=1).reset_index(drop=True)
train_df = shuffled_df[:train_size]
val_df = shuffled_df[train_size:]
# print(train_df, val_df)
train_dataset = AslDataset(train_df, npy_path, character_to_prediction_index_path, config, device, phase= "train")
val_dataset = AslDataset(val_df, npy_path, character_to_prediction_index_path, config, device, phase= "validation")
train_loader = DataLoader(train_dataset , batch_size = 32, shuffle = True, drop_last = True)
val_loader = DataLoader(val_dataset, batch_size = 32, shuffle = True, drop_last = True)


model = Transformer(
    num_hid=num_hid,
    num_head=num_head,
    num_feed_forward=num_feed_forward,
    source_maxlen=source_maxlen,
    target_maxlen=target_maxlen,
    num_layers_enc=num_layers_enc,
    num_layers_dec=num_layers_dec,
    num_classes=num_classes, 
    device = device
)
model.to(device)

checkpoint = torch.load('/workspace/src/torch_asl/checkpoints/checkpoint_epoch_17.pth')
with torch.no_grad():
    model.eval()
    # Perform inference using the example landmarks
    for batch in val_loader:
        landmark = batch[0]['inputs_embeds']
        break
    inference_result = model.inference(landmark)

# Print the generated token sequence
print("Generated Token Sequence:")
print(inference_result[0])