import numpy as np
import pandas as pd
# for develop
import sys
import torch

sys.path.append("/workspace/src/torch_asl")


from models.utils.config import ASLConfig
from models.data.dataset import AslDataset
from models.transformer.als_transformer import Transformer
from torch.utils.data import DataLoader

config = ASLConfig(max_position_embeddings= 96)
# create df in numpy
npy_path = "/workspace/data/asl_numpy_dataset/train_landmarks/5414471.parquet.npy"
df = pd.read_csv("/workspace/data/asl_numpy_dataset/train.csv")
character_to_prediction_index_path = "/workspace/data/asl_numpy_dataset/character_to_prediction_index.json"
asl_dataset = AslDataset(df, npy_path, character_to_prediction_index_path, config)
a, b = asl_dataset.__getitem__(0)
num_hid = 980
num_head = 2
num_feed_forward = 96
source_maxlen = 96
target_maxlen = 96
num_layers_enc = 4
num_layers_dec = 1
num_classes = 59
learning_rate = 0.01

train_loader = DataLoader(asl_dataset, batch_size=1, shuffle = True, drop_last=True)

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
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
num_epochs = 10
# # create loss and optimizer
# num_epochs =  10
# batch_size = 32
# for epoch in range(num_epochs):
#     i = 0
#     model.train()
#     total_loss = 0.0
#     total_correct = 0.
i = 0
print()
for batch in train_loader:
    print(i)
    i = i + 1


# for epoch in range(num_epochs):
#     model.train()
#     total_loss = 0
#     for batch in train_loader:
#         optimizer.zero_grad()
#         outputs = model.training_step(batch)
#         loss = outputs["loss"]
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#     avg_loss = total_loss / len(train_loader)
#     print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss}")

# print(model(a["inputs_embeds"], b)[0].shape)


# # print(asl_dataset.__len__())
