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
npy_path = "/workspace/data/asl_dash_dataset/train_landmarks/train_npy"
df = pd.read_csv("/workspace/data/asl_dash_dataset/train.csv")

df = df[df['length'] >= 50]
len_data = df.shape[0]
split = 0.3
val_size = int(0.3 * len_data)
train_size = len_data - val_size
train_df, val_df = random_split(df, [train_size, val_size])
character_to_prediction_index_path = "/workspace/data/asl_dash_dataset/character_to_prediction_index.json"
# a, b = asl_dataset.__getitem__(0)
character_to_prediction_index_path = "/workspace/data/asl_dash_dataset/character_to_prediction_index.json"
# a, b = asl_dataset.__getitem__(0)
num_hid = 183
num_head = 3
num_feed_forward = 96
source_maxlen = 96
target_maxlen = 96
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
device = torch.device("cuda:0")

train_dataset = AslDataset(train_df.dataset, npy_path, character_to_prediction_index_path, config, device)
val_dataset = AslDataset(val_df.dataset, npy_path, character_to_prediction_index_path, config, device)
train_loader = DataLoader(train_dataset , batch_size = 1, shuffle = True, drop_last = True)
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
    device= device
)

checkpoint = torch.load('/workspace/src/torch_asl/checkpoints/checkpoint_epoch_6.pth')

model.load_state_dict(checkpoint["state_dict"]) 
model.eval()
model.to(device)
with torch.no_grad():
    for batch in train_loader:
        landmark_input, phrase = batch 
        landmark = landmark_input['inputs_embeds']
        phrase = phrase.int()
        outputs = model.encoder(landmark)
        preds = torch.argmax(outputs, dim = 2).cpu().numpy()
        print("[target]: ", phrase)
        print("[predict]: ", preds)
        # for i in range(bs):
        #     # print(phrase[i, :])
        #     target = "".join(train_dataset.num_to_char[_.cpu().item()] for _ in phrase[i, :])
        #     prediction = ""
        #     for j in range(preds[i].shape[0]):
        #         prediction += train_dataset.num_to_char[preds[i, j]]
        #         if preds[i, j] == 3:
        #             break
        #     print("[target]: ", phrase)
        #     print("[predict]: ", preds)
        #     break
        break
# export preds:
