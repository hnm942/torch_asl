import numpy as np
import pandas as pd
# for develop
import sys
import torch
from tqdm import tqdm
import sys
sys.path.append("/workspace/src/torch_asl")


from models.utils.config import ASLConfig
from models.data.dataset import AslDataset
from models.transformer.als_transformer import Transformer
from torch.utils.data import DataLoader
import torch.nn.functional as F

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
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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
for epoch in range(num_epochs):
    total_loss = 0.0
    model.train()
    print("Epoch j: ", epoch)
    for j, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
        # Forward and backward pass using training_step function
        output = model.training_step(batch)
        loss = output["loss"]
        total_loss += loss

    avg_loss = total_loss / len(train_loader)  # Tránh chia số lượng batch, không phải dataset
    print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_loss:.4f}")

    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        for j, batch in enumerate(tqdm(val_loader, desc=f"Epoch {epoch}")):
            output = model.validation_step(batch)
            loss = output["loss"]
            val_loss += loss
    val_avg_loss = val_loss / len(val_loader)  # Tránh chia số lượng batch, không phải dataset
    print(f"Validation Loss: {val_avg_loss:.4f}")

    # Save checkpoint
    checkpoint = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': model.optimizer.state_dict(),
        'loss': avg_loss,
        'val_loss': val_avg_loss
    }
    checkpoint_path = f"/workspace/src/torch_asl/checkpoints/checkpoint_epoch_{epoch+1}.pth"
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")
