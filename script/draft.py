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
from models.transformer.embedding import TokenEmbedding, LandmarkEmbedding
from models.transformer.encoder import TransformerEncoder
from models.transformer.decoder import TransformerDecoder
from torch import nn
import matplotlib.pyplot as plt

config = ASLConfig(max_landmark_size = 96, max_phrase_size = 64)
# create df in numpy
npy_path = "/workspace/data/asl_numpy_dataset/train_landmarks/train_npy"
df = pd.read_csv("/workspace/data/asl_numpy_dataset/train.csv")

df = df[df['length'] >= 50]
len_data = df.shape[0]
split = 0.3
val_size = int(0.3 * len_data)
train_size = len_data - val_size
train_df, val_df = random_split(df, [train_size, val_size])
character_to_prediction_index_path = "/workspace/data/asl_numpy_dataset/character_to_prediction_index.json"
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
device = torch.device("cpu")

train_dataset = AslDataset(train_df.dataset, npy_path, character_to_prediction_index_path, config, device,  phase = "train")
val_dataset = AslDataset(val_df.dataset, npy_path, character_to_prediction_index_path, config, device, phase = "val")
train_loader = DataLoader(train_dataset , batch_size = 32, shuffle = True, drop_last = True)
val_loader = DataLoader(val_dataset, batch_size = 32, shuffle = True, drop_last = True)
# for batch in train_loader:
#     landmark_input, phrase = batch 
#     landmark, landmark_mask = landmark_input['inputs_embeds'], landmark_input["attention_mask"]
#     break


# batch = val_dataset[0]
# enc_input, dec_input = batch 
# landmark = enc_input["inputs_embeds"]
# landmark_mask= enc_input["attention_mask"] 
# phrase = dec_input["target"]
# phrase_mask = dec_input["target_mask"]
lip, lhand, rhand = val_dataset.get_landmarks(val_dataset.df.iloc[0])
print(lip[0].shape)
plt.scatter(rhand[0, :, 0], rhand[0, :, 1])
plt.show()
# attn

# print(batch)

# #### TEST EMBEDDING #############

# enc_emb = LandmarkEmbedding(num_hid,  96, device).to(device)
# enc_input = enc_emb(landmark)
# print("enc_intput: shape", landmark.shape, ", value: ", landmark)
# print("enc_intput: shape", enc_input.shape, ", value: ", enc_input)
# target_emb = TokenEmbedding(num_vocab=num_classes, maxlen=config.max_phrase_size, num_hid=num_hid, device=device)
# dec_input = target_emb(phrase)
# print("dec_intput: shape", phrase.shape, ", value: ", phrase)
# print("dec_intput: shape", dec_input.shape, ", value: ", dec_input)

# enc_layer = TransformerEncoder(num_hid, num_head, num_feed_forward, device)
# enc_x = enc_layer(enc_input)
# print("enc_layer: shape", enc_x.shape, ", value: ", enc_x)
# dec_layer = TransformerDecoder(num_hid, num_head, num_feed_forward, device)
# dec_x = dec_layer(enc_input, dec_input)
# print("enc_layer: shape", dec_x.shape, ", value: ", dec_x)
# classifier = nn.Linear(num_hid, num_classes)
# cls_x = classifier(dec_x)
# print("classifier: shape", cls_x.shape, ", value: ", cls_x)

