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
npy_path = "/workspace/data/asl_numpy_dataset/train_landmarks/train_npy"
df = pd.read_csv("/workspace/data/asl_numpy_dataset/train.csv")
print("len data: ", df.shape[0])
character_to_prediction_index_path = "/workspace/data/asl_numpy_dataset/character_to_prediction_index.json"
asl_dataset = AslDataset(df, npy_path, character_to_prediction_index_path, config)
# a, b = asl_dataset.__getitem__(0)
num_hid = 980
num_head = 2
num_feed_forward = 96
source_maxlen = 96
target_maxlen = 96
num_layers_enc = 4
num_layers_dec = 1
num_classes = 59
learning_rate = 0.01
num_epochs = 100
train_loader = DataLoader(asl_dataset, batch_size=32, shuffle = True, drop_last=True)
# print(train_loader.__len__())
# i = 0
# for i, batch in enumerate(train_loader):
#     print(i)
#     # i = i + 1

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
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.CrossEntropyLoss()
for epoch in range(num_epochs):
    total_loss = 0.0
    total_correct = 0
    model.train()
    for j, batch in enumerate(train_loader):
        print("batch {}|{}".format(j, epoch))
        input, phrase = batch
        optimizer.zero_grad()
        # forward pass
        outputs = model(input['inputs_embeds'], phrase)
        one_hot = torch.nn.functional.one_hot(phrase, num_classes= 59).float()
        loss = loss_fn(outputs, one_hot)
        # backpropagation and optimization
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        print(loss)
    avg_loss = total_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_loss:.4f}")

    # accuracy = total_correct / len(train_loader.dataset)
    # model.eval()

