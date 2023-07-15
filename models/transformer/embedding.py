import torch
import torch.nn as nn

class TokenEmbedding(nn.Module):
    def __init__(self, num_vocab=1000, maxlen=100, num_hid=64):
        super().__init__()
        self.emb = nn.Embedding(num_vocab, num_hid)
        self.pos_emb = nn.Embedding(maxlen, num_hid)

    def forward(self, x):
        maxlen = x.size(-1)
        x = self.emb(x)
        positions = torch.arange(maxlen, device=x.device).unsqueeze(0)
        positions = self.pos_emb(positions)
        return x + positions

class LandmarkEmbedding(nn.Module):
    def __init__(self, num_hid=64, maxlen=100):
        super().__init__()
        self.conv1 = nn.Conv1d(num_hid, num_hid, kernel_size=11, padding=5)
        self.conv2 = nn.Conv1d(num_hid, num_hid, kernel_size=11, padding=5)
        self.conv3 = nn.Conv1d(num_hid, num_hid, kernel_size=11, padding=5)
        # self.lstm1 = nn.LSTM(num_hid, num_hid, batch_first=True, bidirectional=True)
        # self.lstm2 = nn.LSTM(num_hid, num_hid, batch_first=True, bidirectional=True)
        # self.lstm3 = nn.LSTM(num_hid, num_hid, batch_first=True, bidirectional=True)
        # self.dense1 = nn.Linear(num_hid, num_hid)
        # self.dense2 = nn.Linear(num_hid, num_hid)
        # self.dense3 = nn.Linear(num_hid, num_hid)

    def forward(self, x):
        # x = x.unsqueeze(2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # x = self.dense1(x)
        # x = self.dense2(x)
        # x = self.dense3(x)
        return x
